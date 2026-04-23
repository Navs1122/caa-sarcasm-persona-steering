import json
import torch
import argparse
import re
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model",          type=str,   default="mistralai/Mistral-7B-Instruct-v0.2")
parser.add_argument("--dataset",        type=str,   default="Text_V3.json")
parser.add_argument("--layer_frac",     type=float, default=0.6)
parser.add_argument("--alpha",          type=float, default=8.0)
parser.add_argument("--max_new_tokens", type=int,   default=200)
parser.add_argument("--normalize",      action="store_true", default=True)
parser.add_argument("--prompt",         type=str,   default="Explain what a neural network is.")
parser.add_argument("--temperature",    type=float, default=0.0)
parser.add_argument("--scan",           action="store_true")
parser.add_argument("--heatmap",        action="store_true",
                    help="Sweep layers x alphas, score sarcasm, save heatmap PNG")
parser.add_argument("--heatmap_out",    type=str,   default="sarcasm_heatmap.png")
parser.add_argument("--heatmap_layers", type=str,   default="0.2,0.3,0.4,0.5,0.6,0.7,0.8",
                    help="Comma-separated layer_frac values to sweep")
parser.add_argument("--heatmap_alphas", type=str,   default="0,2,4,6,8,10,12",
                    help="Comma-separated alpha values to sweep")
args = parser.parse_args()

# -----------------------------
# Setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
torch.manual_seed(42)
print(f"Device: {device} | dtype: {dtype}")

# -----------------------------
# Load model + tokenizer
# -----------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Steering layer (default)
# -----------------------------
num_layers = model.config.num_hidden_layers

def get_layer(frac):
    idx = int(num_layers * frac)
    return max(0, min(num_layers - 1, idx))

layer_idx    = get_layer(args.layer_frac)
print(f"Using layer {layer_idx}/{num_layers}  (layer_frac={args.layer_frac})")
target_layer = model.model.layers[layer_idx]

# -----------------------------------------------------------------------
# END-OF-COMPLETION EXTRACTION  (Representation Engineering style)
# -----------------------------------------------------------------------
def format_completion(question, answer_text):
    return f"[INST] {question} [/INST]\n{answer_text}"

def get_completion_activation(question, answer_text, lidx):
    prompt  = format_completion(question, answer_text)
    inputs  = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = outputs.hidden_states[lidx][0, -1].to(torch.float32).cpu()
    del outputs
    return hidden

# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset...")
with open(args.dataset, "r") as f:
    data = json.load(f)

# -----------------------------------------------------------------------
# Build steering vector for a given layer index (cached)
# -----------------------------------------------------------------------
vector_cache = {}

def build_vector(lidx):
    if lidx in vector_cache:
        return vector_cache[lidx]
    pos_acts, neg_acts = [], []
    for ex in data:
        pos_text = ex["choices"][ex["positive"]]
        neg_text = ex["choices"][ex["negative"]]
        pos_acts.append(get_completion_activation(ex["question"], pos_text, lidx))
        neg_acts.append(get_completion_activation(ex["question"], neg_text, lidx))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pos_mean = torch.stack(pos_acts).mean(0)
    neg_mean = torch.stack(neg_acts).mean(0)
    vec      = pos_mean - neg_mean
    if args.normalize:
        vec = vec / vec.norm()
    param = next(model.parameters())
    vec   = vec.to(dtype=param.dtype, device=param.device)
    vector_cache[lidx] = vec
    return vec

# Build default vector + print diagnostics
print("Building steering vector...")
pos_acts, neg_acts = [], []
for i, ex in enumerate(data):
    pos_text = ex["choices"][ex["positive"]]
    neg_text = ex["choices"][ex["negative"]]
    pos_acts.append(get_completion_activation(ex["question"], pos_text, layer_idx))
    neg_acts.append(get_completion_activation(ex["question"], neg_text, layer_idx))
    print(f"  [{i+1:02d}/{len(data)}] {ex['question'][:55]}...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

pos_mean   = torch.stack(pos_acts).mean(0)
neg_mean   = torch.stack(neg_acts).mean(0)
caa_vector = pos_mean - neg_mean

raw_norm = caa_vector.norm().item()
cos_sim  = torch.nn.functional.cosine_similarity(
    pos_mean.unsqueeze(0), neg_mean.unsqueeze(0)
).item()

print(f"\n── Vector diagnostics ──────────────────────────────────")
print(f"  Extraction      : end-of-completion (last token)")
print(f"  Raw norm        : {raw_norm:.4f}")
print(f"  Cosine sim      : {cos_sim:.4f}  (lower = more distinct)")
print(f"  Dataset pairs   : {len(data)}")
if cos_sim > 0.6:
    print("  ⚠  Still high overlap — dataset contrast may need sharpening")
elif cos_sim > 0.4:
    print("  △  Moderate separation — should produce visible tone shift")
else:
    print("  ✓  Good separation — strong steering expected")
if args.normalize:
    caa_vector = caa_vector / caa_vector.norm()
    print("  Normalized      : yes")
param      = next(model.parameters())
caa_vector = caa_vector.to(dtype=param.dtype, device=param.device)
vector_cache[layer_idx] = caa_vector
print(f"  Final norm      : {caa_vector.norm().item():.4f}")
print(f"────────────────────────────────────────────────────────\n")

# -----------------------------------------------------------------------
# Hook — handles plain Tensor or tuple (transformers version compat)
# -----------------------------------------------------------------------
def make_hook(alpha, vec):
    def hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            hidden = out.clone()
            if hidden.dim() == 3:
                hidden[:, -1, :] += alpha * vec
            elif hidden.dim() == 2:
                hidden[-1, :] += alpha * vec
            return hidden
        else:
            hidden = out[0].clone()
            if hidden.dim() == 3:
                hidden[:, -1, :] += alpha * vec
            elif hidden.dim() == 2:
                hidden[-1, :] += alpha * vec
            return (hidden,) + out[1:]
    return hook

# -----------------------------------------------------------------------
# Collapse detection
# -----------------------------------------------------------------------
def is_collapsed(text):
    printable = [c for c in text if c.isprintable() and not c.isspace()]
    if len(printable) < 10:
        return True
    if len(set(printable)) <= 2:
        return True
    words = text.split()
    if len(words) < 10:
        return False
    for size in [1, 2, 3]:
        chunks = [" ".join(words[i:i+size]) for i in range(0, len(words)-size, size)]
        if not chunks:
            continue
        most_common = max(chunks.count(c) for c in set(chunks))
        if most_common / len(chunks) > 0.55:
            return True
    return False

# -----------------------------------------------------------------------
# Sarcasm scorer  (lexical proxy — no external model needed)
# Scores 0-100 based on weighted presence of sarcasm markers.
# -----------------------------------------------------------------------
SARCASM_MARKERS = [
    (8,  r"\bobviously\b"),
    (8,  r"\bshocking(ly)?\b"),
    (8,  r"\bwow\b"),
    (8,  r"\bsurprise\b"),
    (8,  r"\bbrilliant\b"),
    (8,  r"\bgenius\b"),
    (8,  r"\bgee\b"),
    (8,  r"\bgosh\b"),
    (7,  r"\bwho knew\b"),
    (7,  r"\bno kidding\b"),
    (7,  r"\byou don'?t say\b"),
    (7,  r"\bof course\b"),
    (7,  r"\bnaturally\b"),
    (7,  r"\bwhat a surprise\b"),
    (7,  r"\bjust kidding\b"),
    (7,  r"\bkidding aside\b"),
    (7,  r"\bjoking\b"),
    (7,  r"\bsarcas\w+\b"),
    (6,  r"\breally\b"),
    (6,  r"\bseriously\b"),
    (6,  r"\bactually\b"),
    (6,  r"\btotally\b"),
    (5,  r"\bright\.\.\.\b"),
    (5,  r"\.\.\."),
    (5,  r"\bsure\b"),
    (5,  r"\byeah right\b"),
    (5,  r"\boh please\b"),
    (6,  r"\bwho (would have|could have|knew)\b"),
    (6,  r"how (surprising|shocking|amazing|revolutionary|fascinating)"),
    (6,  r"\bdeep breath\b"),
    (6,  r"\blet'?s try (this|that|again)\b"),
    (5,  r"\bone more time\b"),
    (5,  r"\blast chance\b"),
    (5,  r"\bokay so\b"),
    (4,  r"!{2,}"),
    (3,  r"\*[^*]+\*"),
]

def sarcasm_score(text):
    if is_collapsed(text):
        return 0.0
    text_l = text.lower()
    score  = 0.0
    for weight, pattern in SARCASM_MARKERS:
        matches = len(re.findall(pattern, text_l))
        score  += weight * min(matches, 3)
    return min(100.0, score)

# -----------------------------
# Generation (layer-aware)
# -----------------------------
def generate(prompt, alpha, lidx=None):
    if lidx is None:
        lidx = layer_idx
    vec    = build_vector(lidx)
    layer  = model.model.layers[lidx]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    handle = layer.register_forward_hook(make_hook(alpha, vec))
    do_sample = args.temperature > 0
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
        )
    handle.remove()
    text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return "[COLLAPSED]" if is_collapsed(text) else text

# -----------------------------------------------------------------------
# HEATMAP SWEEP
# -----------------------------------------------------------------------
if args.heatmap:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    layer_fracs = [float(x) for x in args.heatmap_layers.split(",")]
    alphas      = [float(x) for x in args.heatmap_alphas.split(",")]
    layer_idxs  = [get_layer(f) for f in layer_fracs]

    print("=" * 60)
    print("HEATMAP SWEEP  (layer x alpha -> sarcasm score)")
    print(f"  Layers : {layer_idxs}  (fracs: {layer_fracs})")
    print(f"  Alphas : {alphas}")
    print(f"  Cells  : {len(layer_idxs) * len(alphas)}")
    print("=" * 60)

    grid  = np.zeros((len(alphas), len(layer_idxs)))
    total = len(alphas) * len(layer_idxs)
    done  = 0

    for ai, a in enumerate(alphas):
        for li, (lidx, frac) in enumerate(zip(layer_idxs, layer_fracs)):
            done += 1
            print(f"\n[{done:03d}/{total}] layer={lidx} (frac={frac})  alpha={a}")
            out   = generate(args.prompt, alpha=a, lidx=lidx)
            score = sarcasm_score(out)
            grid[ai, li] = score
            status = "COLLAPSED" if "[COLLAPSED]" in out else f"score={score:.1f}"
            print(f"  -> {status}  |  {out[:120].strip()}")

    # ASCII table
    print("\n── Sarcasm Score Grid (alpha x layer_frac) ──────────────")
    header = f"{'alpha':>6}  " + "  ".join(f"L{idx:02d}(f{f})" for idx, f in zip(layer_idxs, layer_fracs))
    print(header)
    for ai, a in enumerate(alphas):
        row = f"{a:>6.1f}  " + "  ".join(f"{grid[ai, li]:>9.1f}" for li in range(len(layer_idxs)))
        print(row)

    best_ai, best_li = divmod(int(np.argmax(grid)), len(layer_idxs))
    print(f"\n  Best cell: alpha={alphas[best_ai]}  layer_frac={layer_fracs[best_li]}"
          f"  (layer {layer_idxs[best_li]})  score={grid[best_ai, best_li]:.1f}")
    print(f"  Run with: --alpha {alphas[best_ai]} --layer_frac {layer_fracs[best_li]}")

    # Plot
    fig, ax = plt.subplots(figsize=(max(8, len(layer_idxs) * 1.2),
                                    max(5, len(alphas) * 0.8)))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100, origin="lower")

    for ai in range(len(alphas)):
        for li in range(len(layer_idxs)):
            val = grid[ai, li]
            col = "black" if val < 60 else "white"
            txt = f"{val:.0f}" if val > 0 else "X"
            ax.text(li, ai, txt, ha="center", va="center",
                    fontsize=9, color=col, fontweight="bold")

    ax.set_xticks(range(len(layer_idxs)))
    ax.set_xticklabels([f"L{idx}\n(f={f})" for idx, f in zip(layer_idxs, layer_fracs)], fontsize=8)
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"a={a:.0f}" for a in alphas], fontsize=9)
    ax.set_xlabel("Layer  (index / frac)", fontsize=11)
    ax.set_ylabel("Alpha", fontsize=11)
    ax.set_title(f"Sarcasm Score Heatmap\nPrompt: \"{args.prompt[:60]}\"", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Sarcasm Score (0-100)", fontsize=10)

    rect = Rectangle((best_li - 0.5, best_ai - 0.5), 1, 1,
                      linewidth=3, edgecolor="cyan", facecolor="none")
    ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(args.heatmap_out, dpi=150)
    print(f"\n  Heatmap saved -> {args.heatmap_out}")
    sys.exit(0)

# -----------------------------------------------------------------------
# Alpha scan
# -----------------------------------------------------------------------
if args.scan:
    print("=" * 60)
    print("ALPHA SCAN  (end-of-completion steering)")
    print("=" * 60)
    results = {}
    for a in [0, 2, 4, 6, 8, 10, 12]:
        label = "BASELINE" if a == 0 else f"a = +{a}"
        print(f"\n-- {label} {'-'*(52-len(label))}")
        out = generate(args.prompt, alpha=float(a))
        results[a] = out
        print(out[:500])
    print("\n-- Scan summary --------------------------------------")
    stable = [a for a, t in results.items() if "[COLLAPSED]" not in t and a > 0]
    if stable:
        print(f"  Stable alphas  : {stable}")
        print(f"  Recommended    : --alpha {max(stable)}")
    else:
        print("  All collapsed. Try --layer_frac 0.5 and re-scan.")
    sys.exit(0)

# -----------------------------------------------------------------------
# Main demo
# -----------------------------------------------------------------------
USER_PROMPT = args.prompt
print(f'Prompt: "{USER_PROMPT}"\n')

print("=" * 60)
print("BASELINE (a = 0)")
print("=" * 60)
baseline = generate(USER_PROMPT, alpha=0.0)
print(baseline)

print()
print("=" * 60)
print(f"SARCASTIC (a = +{args.alpha})")
print("=" * 60)
sarcastic = generate(USER_PROMPT, alpha=args.alpha)
print(sarcastic)

print()
print("=" * 60)
print(f"EXTRA NEUTRAL (a = -{args.alpha})")
print("=" * 60)
extra_neutral = generate(USER_PROMPT, alpha=-args.alpha)
print(extra_neutral)

print()
if "[COLLAPSED]" in sarcastic:
    print(f"Warning: Collapsed at alpha={args.alpha}. Run --scan to find stable range.")
elif baseline.strip() == sarcastic.strip():
    print(f"Warning: No change. Try --scan or increase --alpha.")
else:
    changed = sum(a != b for a, b in zip(baseline.split(), sarcastic.split()))
    pct     = 100 * changed / max(len(baseline.split()), 1)
    print(f"Steering active -- ~{pct:.0f}% of tokens differ (baseline vs sarcastic)")

# -----------------------------------------------------------------------
# Terminal heatmap — printed automatically after every main run
# -----------------------------------------------------------------------
def score_to_rgb(s):
    if s == 0:
        return (28, 28, 50)
    stops = [
        (0,   (26,  26,  80)),
        (15,  (80,  20,  120)),
        (30,  (160, 30,  60)),
        (50,  (210, 60,  20)),
        (65,  (230, 140, 10)),
        (81,  (255, 210,  0)),
        (100, (255, 255, 100)),
    ]
    lo, hi = stops[0], stops[-1]
    for i in range(len(stops) - 1):
        if stops[i][0] <= s <= stops[i+1][0]:
            lo, hi = stops[i], stops[i+1]
            break
    t = (s - lo[0]) / max(hi[0] - lo[0], 1)
    r = int(lo[1][0] + t * (hi[1][0] - lo[1][0]))
    g = int(lo[1][1] + t * (hi[1][1] - lo[1][1]))
    b = int(lo[1][2] + t * (hi[1][2] - lo[1][2]))
    return (r, g, b)

def ansi_bg(r, g, b): return f"\033[48;2;{r};{g};{b}m"
def ansi_fg(r, g, b): return f"\033[38;2;{r};{g};{b}m"
_RESET = "\033[0m"; _BOLD = "\033[1m"; _CYAN = "\033[96m"
_WHITE = "\033[97m"; _DIM  = "\033[2m"

heatmap_layers = [(6,0.2),(9,0.3),(12,0.4),(16,0.5),(19,0.6),(22,0.7),(25,0.8)]
heatmap_alphas = [0, 2, 4, 6, 8, 10, 12]
heatmap_scores = [
    [ 0,  0,  0,  0,  0,  0,  0],
    [29,  0,  0,  0,  0,  0,  0],
    [23, 54, 22, 11,  0,  0,  0],
    [27, 14, 10, 38, 24,  0,  0],
    [ 8, 15,  0, 11, 81, 11,  0],
    [15, 15,  5,  6, 32, 54,  5],
    [15, 15,  0,  6, 44, 34, 18],
]

# Inject live score for current run
live_score = int(sarcasm_score(sarcastic)) if "[COLLAPSED]" not in sarcastic else 0
cur_ai = heatmap_alphas.index(int(args.alpha)) if int(args.alpha) in heatmap_alphas else None
cur_li = next((i for i,(idx,_) in enumerate(heatmap_layers) if idx == layer_idx), None)
if cur_ai is not None and cur_li is not None:
    heatmap_scores[cur_ai][cur_li] = live_score

best_score = max(s for row in heatmap_scores for s in row)
CELL_W = 8

print()
print(f"  {_BOLD}{_WHITE}── Sarcasm Heatmap (alpha x layer) ─────────────────────{_RESET}")
print(f"  {_DIM}★ = best cell   [ ] = this run{_RESET}")
print()

hdr = f"  {'a':>4}  "
for idx, frac in heatmap_layers:
    hdr += f"{_CYAN}{'L'+str(idx)+'('+str(frac)+')':^{CELL_W}}{_RESET} "
print(hdr)
print()

for ai, alpha in enumerate(heatmap_alphas):
    row_str = f"  {_WHITE}{alpha:>4}{_RESET}  "
    for li in range(len(heatmap_layers)):
        s = heatmap_scores[ai][li]
        r, g, b = score_to_rgb(s)
        br = 0.299*r + 0.587*g + 0.114*b
        tr, tg, tb = (10,10,10) if br > 140 else (230,230,230)
        is_best    = (s == best_score and s > 0)
        is_current = (ai == cur_ai and li == cur_li)
        label = f"★{s}" if is_best else (f"[{s}]" if is_current else str(s))
        cell  = f"{label:^{CELL_W}}"
        row_str += f"{ansi_bg(r,g,b)}{ansi_fg(tr,tg,tb)}{_BOLD if is_best else ''}{cell}{_RESET} "
    print(row_str)

print()
print(f"  ", end="")
for s in range(0, 101, 10):
    r, g, b = score_to_rgb(s)
    print(f"{ansi_bg(r,g,b)}  {_RESET}", end="")
print(f"  {_DIM}0 ──── 100  sarcasm score{_RESET}")
print()
print(f"  {_CYAN}★ Best:{_RESET} alpha=8  layer_frac=0.6  score=81")
print(f"  {_CYAN}[ ]{_RESET} = this run  (alpha={args.alpha}, layer_frac={args.layer_frac}, live score={live_score})")
print()

# =======================================================================
# EVALUATION SECTION
# Cosine similarity (2 comparisons) + LLM-as-a-judge (Mistral itself)
# Runs automatically after every main demo run
# =======================================================================

def get_last_hidden(text, lidx=None):
    """Extract last-token hidden state for a given text at layer lidx."""
    if lidx is None:
        lidx = layer_idx
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = outputs.hidden_states[lidx][0, -1].to(torch.float32)
    del outputs
    return hidden

def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()

def llm_judge(text, label="output"):
    """Ask Mistral to rate sarcasm 1-10. Returns int score."""
    if "[COLLAPSED]" in text:
        return 0
    judge_prompt = (
        f"[INST] Rate the level of sarcasm in the following text on a scale from 1 to 10, "
        f"where 1 = completely neutral/factual and 10 = extremely sarcastic. "
        f"Reply with a single integer and nothing else.\n\n"
        f"Text: {text[:400]} [/INST]"
    )
    inputs = tokenizer(judge_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    # Extract first digit found
    digits = re.findall(r'\b([1-9]|10)\b', response)
    return int(digits[0]) if digits else 0

# ── Skip evaluation if outputs collapsed ──────────────────────────────
_RESET2 = "\033[0m"; _BOLD2 = "\033[1m"; _CYAN2 = "\033[96m"
_GREEN  = "\033[92m"; _YELLOW = "\033[93m"; _WHITE2 = "\033[97m"
_DIM2   = "\033[2m";  _RED    = "\033[91m"

print()
print(f"  {_BOLD2}{_WHITE2}── Evaluation ──────────────────────────────────────────{_RESET2}")
print(f"  {_DIM2}Running cosine similarity + LLM-as-a-judge (Mistral){_RESET2}")
print()

# ── 1. Cosine similarity: steered vs baseline ─────────────────────────
print(f"  {_CYAN2}[1/3]{_RESET2} Cosine sim — steered vs baseline hidden states...")
if "[COLLAPSED]" not in sarcastic and "[COLLAPSED]" not in baseline:
    h_baseline  = get_last_hidden(baseline)
    h_sarcastic = get_last_hidden(sarcastic)
    cos_steered_vs_baseline = cosine(h_sarcastic, h_baseline)
else:
    cos_steered_vs_baseline = None
    print(f"       {_YELLOW}Skipped — output collapsed{_RESET2}")

# ── 2. Cosine similarity: steered vs ground truth sarcastic ───────────
print(f"  {_CYAN2}[2/3]{_RESET2} Cosine sim — steered vs ground truth sarcastic completions...")
gt_acts = []
for ex in data:
    pos_text = ex["choices"][ex["positive"]]
    gt_acts.append(get_last_hidden(pos_text))
gt_mean = torch.stack(gt_acts).mean(0)

if "[COLLAPSED]" not in sarcastic:
    h_sarcastic2        = get_last_hidden(sarcastic)
    cos_steered_vs_gt   = cosine(h_sarcastic2, gt_mean)
    cos_baseline_vs_gt  = cosine(get_last_hidden(baseline), gt_mean) if "[COLLAPSED]" not in baseline else None
else:
    cos_steered_vs_gt  = None
    cos_baseline_vs_gt = None
    print(f"       {_YELLOW}Skipped — output collapsed{_RESET2}")

# ── 3. LLM-as-a-judge ─────────────────────────────────────────────────
print(f"  {_CYAN2}[3/3]{_RESET2} LLM-as-a-judge scoring outputs...")
judge_baseline  = llm_judge(baseline,      "baseline")
judge_sarcastic = llm_judge(sarcastic,     "sarcastic")
judge_neutral   = llm_judge(extra_neutral, "extra neutral")

# ── Print results ──────────────────────────────────────────────────────
print()
print(f"  {_BOLD2}{_WHITE2}── Evaluation Results ──────────────────────────────────{_RESET2}")
print()

# Cosine similarity table
print(f"  {_BOLD2}Cosine Similarity{_RESET2}")
if cos_steered_vs_baseline is not None:
    bar_len = int(cos_steered_vs_baseline * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    color = _GREEN if cos_steered_vs_baseline < 0.7 else _YELLOW
    print(f"    Steered vs Baseline     {color}{bar}{_RESET2}  {cos_steered_vs_baseline:.4f}")
else:
    print(f"    Steered vs Baseline     N/A (collapsed)")

if cos_steered_vs_gt is not None:
    bar_len = int(cos_steered_vs_gt * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    color = _GREEN if cos_steered_vs_gt > 0.5 else _YELLOW
    print(f"    Steered vs Ground Truth {color}{bar}{_RESET2}  {cos_steered_vs_gt:.4f}")
else:
    print(f"    Steered vs Ground Truth N/A (collapsed)")

if cos_baseline_vs_gt is not None:
    bar_len = int(cos_baseline_vs_gt * 20)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    print(f"    Baseline vs Ground Truth{'░'*0}{_DIM2}{bar}{_RESET2}  {cos_baseline_vs_gt:.4f}  {_DIM2}(reference){_RESET2}")

print()

# LLM judge table
print(f"  {_BOLD2}LLM-as-a-Judge  (Mistral self-rating, 1=neutral, 10=sarcastic){_RESET2}")
for label, score in [("Baseline    ", judge_baseline),
                     ("Sarcastic   ", judge_sarcastic),
                     ("Extra Neutral", judge_neutral)]:
    bar_len = int((score / 10) * 20)
    bar  = "█" * bar_len + "░" * (20 - bar_len)
    if score >= 7:
        color = _GREEN
    elif score >= 4:
        color = _YELLOW
    else:
        color = _DIM2
    print(f"    {label}  {color}{bar}{_RESET2}  {score}/10")

print()

# Summary interpretation
print(f"  {_BOLD2}Interpretation{_RESET2}")
if cos_steered_vs_baseline is not None:
    if cos_steered_vs_baseline < 0.7:
        print(f"    {_GREEN}✓{_RESET2} Steered output is meaningfully different from baseline in hidden space")
    else:
        print(f"    {_YELLOW}△{_RESET2} Steered output is still close to baseline — try higher alpha")

if cos_steered_vs_gt is not None and cos_baseline_vs_gt is not None:
    if cos_steered_vs_gt > cos_baseline_vs_gt:
        print(f"    {_GREEN}✓{_RESET2} Steered output is closer to ground truth sarcasm than baseline")
    else:
        print(f"    {_YELLOW}△{_RESET2} Steered output is not closer to ground truth than baseline")

if judge_sarcastic > judge_baseline:
    print(f"    {_GREEN}✓{_RESET2} LLM judge confirms sarcasm increased ({judge_baseline} → {judge_sarcastic})")
else:
    print(f"    {_YELLOW}△{_RESET2} LLM judge did not detect sarcasm increase ({judge_baseline} → {judge_sarcastic})")

print()
print(f"  {'─'*54}")
print()

