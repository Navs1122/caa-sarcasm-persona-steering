import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
parser.add_argument("--dataset", type=str, default="caa_neural_networks_sarcasm_strong.json")
parser.add_argument("--layer_frac", type=float, default=0.65)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# -----------------------------
# Load model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else None
)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Choose layer (paper: mid–late)
# -----------------------------
num_layers = model.config.num_hidden_layers
layer_idx = int(num_layers * args.layer_frac)
layer_idx = max(0, min(num_layers - 1, layer_idx))

print(f"Using layer {layer_idx}/{num_layers}")

# -----------------------------
# Prompt formatting (CAA paper style)
# -----------------------------
def format_prompt(question, choices, answer):
    return (
        f"[INST] {question} [/INST]\n"
        f"Choices: (A) {choices['A']} (B) {choices['B']}\n"
        f"Answer: {answer}"
    )

# -----------------------------
# Extract activation at ANSWER LETTER token
# -----------------------------
def get_answer_activation(prompt, answer_letter):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False
        )

    answer_id = tokenizer.encode(answer_letter, add_special_tokens=False)[0]
    token_ids = inputs["input_ids"][0]
    pos = (token_ids == answer_id).nonzero(as_tuple=True)[0][-1]

    return out.hidden_states[layer_idx][0, pos].to(torch.float32)

# -----------------------------
# Load dataset
# -----------------------------
with open(args.dataset, "r") as f:
    data = json.load(f)

# -----------------------------
# Build CAA vector (Mean Difference)
# -----------------------------
pos_acts, neg_acts = [], []

for ex in data:
    pos_prompt = format_prompt(ex["question"], ex["choices"], ex["positive"])
    neg_prompt = format_prompt(ex["question"], ex["choices"], ex["negative"])

    pos_acts.append(get_answer_activation(pos_prompt, ex["positive"]))
    neg_acts.append(get_answer_activation(neg_prompt, ex["negative"]))

caa_vector = torch.stack(pos_acts).mean(0) - torch.stack(neg_acts).mean(0)

# 🔑 CRITICAL FIX: match model dtype + device
param = next(model.parameters())
caa_vector = caa_vector.to(dtype=param.dtype, device=param.device)

torch.save(caa_vector, "caa_sarcasm_vector.pt")
print("Saved CAA sarcasm vector:", caa_vector.shape, caa_vector.dtype)

# -----------------------------
# Steering hook (dtype-safe)
# -----------------------------
def make_hook(alpha):
    def hook(module, inp, out):
        if not isinstance(out, torch.Tensor) or alpha == 0.0:
            return out
        return out + alpha * caa_vector
    return hook

target_layer = model.model.layers[layer_idx]

# -----------------------------
# Generation
# -----------------------------
def generate(prompt, alpha):
    handle = target_layer.register_forward_hook(make_hook(alpha))

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    attn = (inputs["input_ids"] != tokenizer.pad_token_id).long()

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=attn,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    handle.remove()
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

# -----------------------------
# Demo
# -----------------------------
USER_PROMPT = "Explain what a neural network is so a beginner understands."

print("\n=== BASELINE (alpha=0) ===\n")
print(generate(USER_PROMPT, alpha=0.0))

print(f"\n=== SARCASTIC (alpha={args.alpha:+.2f}) ===\n")
print(generate(USER_PROMPT, alpha=args.alpha))

print(f"\n=== EXTRA NEUTRAL (alpha={-args.alpha:+.2f}) ===\n")
print(generate(USER_PROMPT, alpha=-args.alpha))
