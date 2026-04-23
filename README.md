# Contrastive Activation Addition (CAA) – Sarcasm Persona Steering

This repository demonstrates **Contrastive Activation Addition (CAA)** used to steer a model's persona (neutral ↔ sarcastic) at inference time. Rather than changing the prompt or retraining the model, behavior is controlled by directly injecting a persona vector into the model's hidden states during generation.

## Key Points

- This is **not prompt engineering** — the prompt stays identical across all runs
- This is **not fine-tuning** — model weights are never modified
- Behavior is controlled via an **activation vector** extracted from the model's residual stream
- Steering strength is continuously adjustable via `--alpha` — fully reversible
- Evaluated using three independent metrics: lexical scoring, cosine similarity, and LLM-as-a-judge

## How It Works

1. A contrastive dataset of 30 question-answer pairs is built — each with one sarcastic and one neutral completion
2. Both completions are run through Mistral-7B and the hidden state at the **last token** is extracted at a target layer
3. The mean difference between sarcastic and neutral hidden states produces a normalized **persona vector**
4. At inference time, a PyTorch forward hook injects the vector at the target layer — nudging the model toward sarcasm (`+alpha`) or extra neutrality (`-alpha`)

The cosine similarity between sarcastic and neutral activation clusters dropped from **0.779 → 0.298** across dataset iterations, indicating increasingly distinct behavioral separation.

## Results

| Metric | Value |
|--------|-------|
| LLM Judge — Baseline | 1/10 |
| LLM Judge — Steered | 8/10 |
| Steered vs Baseline cosine sim | 0.028 |
| Steered vs Ground Truth cosine sim | 0.242 |
| Token difference (baseline vs steered) | ~98% |
| Optimal configuration | layer_frac=0.4, extract_frac=0.5, alpha=4 |

## Files

| File | Description |
|------|-------------|
| `caa_eval.py` | Full evaluation pipeline — extraction, steering, heatmap, cosine similarity, LLM-as-a-judge |
| `caa_sarcasm_v5.py` | Original CAA pipeline — extraction, vector building, generation, collapse detection |
| `run_caa_sarcasm.py` | Simplified end-to-end CAA extraction + generation |
| `Text_V6.json` | Current dataset — longer, more stylistically extreme sarcastic completions |
| `Text_V5.json` | Earlier dataset iteration |
| `caa_neural_networks_sarcasm_strong.json` | Initial CAA dataset |

## Run

```bash
# Runs everything automatically — baseline, steered, extra neutral,
# terminal heatmap, cosine similarity, and LLM-as-a-judge evaluation
python caa_eval.py --dataset Text_V6.json --alpha 4 --layer_frac 0.4
```

### What it outputs:
1. **Vector diagnostics** — cosine similarity and norm of the steering vector
2. **Baseline output** — unsteered model response (α = 0)
3. **Steered output** — sarcastic response (α = +4)
4. **Extra neutral output** — opposite direction (α = -4)
5. **Terminal heatmap** — color-coded layer × alpha sarcasm scores
6. **Cosine similarity** — steered vs baseline and steered vs ground truth
7. **LLM-as-a-judge** — Mistral self-rates each output 1–10 for sarcasm

### Optional flags:
```bash
# Change the prompt
python caa_eval.py --dataset Text_V6.json --alpha 4 --layer_frac 0.4 --prompt "Why is sleep important?"

# Change steering strength
python caa_eval.py --dataset Text_V6.json --alpha 8 --layer_frac 0.4

# Run on a different dataset
python caa_eval.py --dataset Text_V5.json --alpha 8 --layer_frac 0.6
```

## Model

- **Model:** `mistralai/Mistral-7B-Instruct-v0.2`
- **Injection layer:** Layer 12 of 32 (`layer_frac=0.4`)
- **Extraction layer:** Layer 16 of 32 (`extract_frac=0.5`)
- **Extraction method:** End-of-completion (last token of full answer text)
- **Dataset pairs:** 30 contrastive question-answer pairs

## Evaluation

Three independent metrics are used:
- **Lexical sarcasm scoring** — weighted regex patterns (0–100)
- **Cosine similarity** — steered vs baseline and steered vs ground truth hidden states
- **LLM-as-a-judge** — Mistral self-rates each output 1–10 for sarcasm

## Requirements

```bash
pip install torch transformers accelerate
```
