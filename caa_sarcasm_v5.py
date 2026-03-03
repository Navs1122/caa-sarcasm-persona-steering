import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model",          type=str,   default="mistralai/Mistral-7B-Instruct-v0.2")
parser.add_argument("--dataset",        type=str,   default="text.json")
parser.add_argument("--layer_frac",     type=float, default=0.6)
parser.add_argument("--alpha",          type=float, default=8.0)   # much lower default
parser.add_argument("--max_new_tokens", type=int,   default=200)
parser.add_argument("--normalize",      action="store_true", default=True)
parser.add_argument("--prompt",         type=str,   default="Explain what a neural network is.")
parser.add_argument("--temperature",    type=float, default=0.0)
parser.add_argument("--scan",           action="store_true")
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
# Steering layer
# -----------------------------
num_layers   = model.config.num_hidden_layers
layer_idx    = int(num_layers * args.layer_frac)
layer_idx    = max(0, min(num_layers - 1, layer_idx))
print(f"Using layer {layer_idx}/{num_layers}  (layer_frac={args.layer_frac})")
target_layer = model.model.layers[layer_idx]

# -----------------------------------------------------------------------
# Positional extraction — no token search needed
#
# Format: "[INST] Q [/INST]\nChoices: (A) ... (B) ...\nAnswer: (X)"
# We tokenize the prefix (everything up to the answer letter) separately,
# so the answer token is always at position len(prefix_tokens).
# -----------------------------------------------------------------------
def get_answer_activation(question, choices, answer_letter):
    prefix = (
        f"[INST] {question} [/INST]\n"
        f"Choices: (A) {choices['A']} (B) {choices['B']}\n"
        f"Answer: ("
    )
    full = prefix + f"{answer_letter})"

    prefix_ids  = tokenizer.encode(prefix, add_special_tokens=True)
    full_inputs = tokenizer(full, return_tensors="pt").to(device)
    seq_len     = full_inputs["input_ids"].shape[1]

    # Clamp to valid range
    answer_pos = min(len(prefix_ids), seq_len - 1)

    with torch.no_grad():
        outputs = model(
            **full_inputs,
            output_hidden_states=True,
            use_cache=False,
        )

    return outputs.hidden_states[layer_idx][0, answer_pos].to(torch.float32)

# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset...")
with open(args.dataset, "r") as f:
    data = json.load(f)

# -----------------------------------------------------------------------
# Build CAA vector
# -----------------------------------------------------------------------
print("Building CAA steering vector...")
pos_acts, neg_acts = [], []

for i, ex in enumerate(data):
    pos_acts.append(get_answer_activation(ex["question"], ex["choices"], ex["positive"]))
    neg_acts.append(get_answer_activation(ex["question"], ex["choices"], ex["negative"]))
    print(f"  [{i+1:02d}/{len(data)}] {ex['question'][:55]}...")

pos_mean   = torch.stack(pos_acts).mean(0)
neg_mean   = torch.stack(neg_acts).mean(0)
caa_vector = pos_mean - neg_mean

# Diagnostics
raw_norm = caa_vector.norm().item()
cos_sim  = torch.nn.functional.cosine_similarity(
    pos_mean.unsqueeze(0), neg_mean.unsqueeze(0)
).item()

print(f"\n── Vector diagnostics ──────────────────────────────────")
print(f"  Raw norm        : {raw_norm:.4f}")
print(f"  Cosine sim      : {cos_sim:.4f}  (lower = more distinct)")
print(f"  Dataset pairs   : {len(data)}")

if cos_sim > 0.7:
    print("  ⚠  High overlap — dataset may not be contrastive enough at this layer")
    print("     Try --layer_frac 0.75 or improve dataset contrast")
elif cos_sim > 0.5:
    print("  △  Moderate separation")
else:
    print("  ✓  Good separation")

if args.normalize:
    caa_vector = caa_vector / caa_vector.norm()
    print("  Normalized      : yes")

param      = next(model.parameters())
caa_vector = caa_vector.to(dtype=param.dtype, device=param.device)
print(f"  Final norm      : {caa_vector.norm().item():.4f}")
print(f"────────────────────────────────────────────────────────\n")

# -----------------------------------------------------------------------
# Hook — LAST TOKEN ONLY during generation
#
# KEY INSIGHT from your scan results:
#   steer_all=True pushes EVERY token at every layer call → exponential
#   amplification → gibberish by alpha=15.
#
#   last-token-only steers only the CURRENT prediction token per step,
#   which is the correct CAA formulation for generation.
#   This lets you use higher alpha without collapse.
# -----------------------------------------------------------------------
def make_hook(alpha):
    def hook(module, inp, out):
        hidden = out[0].clone()
        hidden[:, -1, :] += alpha * caa_vector   # only the current token
        return (hidden,) + out[1:]
    return hook

# -----------------------------------------------------------------------
# Repetition detector — catches collapse before printing garbage
# -----------------------------------------------------------------------
def is_collapsed(text, window=6, threshold=0.6):
    """Return True if text has degenerated into repetition loops."""
    words = text.split()
    if len(words) < window * 2:
        return False
    # Check if any window-length phrase repeats > threshold fraction of the time
    for size in [1, 2, 3]:
        chunks = [" ".join(words[i:i+size]) for i in range(0, len(words)-size, size)]
        if len(chunks) == 0:
            continue
        most_common_count = max(chunks.count(c) for c in set(chunks))
        if most_common_count / len(chunks) > threshold:
            return True
    return False

# -----------------------------
# Generation
# -----------------------------
def generate(prompt, alpha):
    inputs    = tokenizer(prompt, return_tensors="pt").to(device)
    handle    = target_layer.register_forward_hook(make_hook(alpha))
    do_sample = args.temperature > 0

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,   # stronger penalty to resist collapse
        )
    handle.remove()

    text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    if is_collapsed(text):
        return f"[COLLAPSED — alpha too high, try lower value]"
    return text

# -----------------------------------------------------------------------
# Alpha scan — now with collapse detection
# -----------------------------------------------------------------------
if args.scan:
    print("=" * 60)
    print("ALPHA SCAN  (last-token steering, collapse detection)")
    print("=" * 60)
    results = {}
    for a in [0, 3, 5, 8, 10, 12, 15]:
        label = "BASELINE" if a == 0 else f"α = +{a}"
        print(f"\n── {label} {'─'*(52 - len(label))}")
        out = generate(args.prompt, alpha=float(a))
        results[a] = out
        print(out[:500])

    # Find sweet spot
    print("\n── Scan summary ─────────────────────────────────────")
    good_alphas = [a for a, t in results.items() if "[COLLAPSED" not in t and a > 0]
    if good_alphas:
        print(f"  Stable alphas  : {good_alphas}")
        print(f"  Recommended    : --alpha {max(good_alphas)}")
    else:
        print("  All alphas collapsed. Try --layer_frac 0.75 and re-scan.")
    import sys; sys.exit(0)

# -----------------------------------------------------------------------
# Main demo
# -----------------------------------------------------------------------
USER_PROMPT = args.prompt
print(f'Prompt: "{USER_PROMPT}"\n')

print("=" * 60)
print("BASELINE (α = 0)")
print("=" * 60)
baseline = generate(USER_PROMPT, alpha=0.0)
print(baseline)

print()
print("=" * 60)
print(f"SARCASTIC (α = +{args.alpha})")
print("=" * 60)
sarcastic = generate(USER_PROMPT, alpha=args.alpha)
print(sarcastic)

print()
print("=" * 60)
print(f"EXTRA NEUTRAL (α = -{args.alpha})")
print("=" * 60)
extra_neutral = generate(USER_PROMPT, alpha=-args.alpha)
print(extra_neutral)

print()
if "[COLLAPSED" in sarcastic:
    print(f"⚠  Collapsed at alpha={args.alpha}. Run --scan to find stable range.")
elif baseline.strip() == sarcastic.strip():
    print(f"⚠  No change at alpha={args.alpha}. Try higher or run --scan.")
else:
    changed = sum(a != b for a, b in zip(baseline.split(), sarcastic.split()))
    pct     = 100 * changed / max(len(baseline.split()), 1)
    print(f"✓  Steering active — ~{pct:.0f}% of tokens differ (baseline vs sarcastic)")
