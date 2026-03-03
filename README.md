# Contrastive Activation Addition (CAA) – Sarcasm Persona Steering

This repository demonstrates **Contrastive Activation Addition (CAA)** used to steer a model's persona (neutral ↔ sarcastic) when explaining neural networks. Rather than changing the prompt, behavior is controlled by directly manipulating the model's internal activation vectors at inference time.

## Key Points

- This is **not prompt engineering** — the prompt stays identical across all runs
- Behavior is controlled via an internal **activation vector** extracted from the model's residual stream
- The vector is built from contrastive A/B pairs (neutral vs sarcastic completions)
- Steering strength is continuously adjustable via `--alpha` — no retraining required

## How It Works

1. For each question-answer pair in the dataset, both the neutral and sarcastic completions are run through the model
2. The hidden state at **layer 19, last token position** is extracted for each completion
3. The mean sarcastic activation is subtracted from the mean neutral activation to produce a **persona vector**
4. During generation, this vector is added to the hidden state at every token step — nudging the model toward sarcasm (`+alpha`) or extra neutrality (`-alpha`)

The cosine similarity between sarcastic and neutral activation clusters dropped from **0.779 → 0.439** across dataset iterations, indicating stronger behavioral separation.

## Files

| File | Description |
|------|-------------|
| `caa_sarcasm_v5.py` | Full CAA pipeline — extraction, vector building, generation, collapse detection |
| `run_caa_sarcasm.py` | Simplified end-to-end CAA extraction + generation |
| `caa_neural_networks_sarcasm_strong.json` | Primary CAA dataset (neutral vs sarcastic) |
| `Text_V5.json` | Earlier dataset iteration |

## Run

```bash
# Baseline (no steering)
python caa_sarcasm_v5.py --dataset caa_neural_networks_sarcasm_strong.json --alpha 0

# Sarcastic
python caa_sarcasm_v5.py --dataset caa_neural_networks_sarcasm_strong.json --alpha 8

# Extra neutral
python caa_sarcasm_v5.py --dataset caa_neural_networks_sarcasm_strong.json --alpha -8

# Scan to find effective alpha range
python caa_sarcasm_v5.py --dataset caa_neural_networks_sarcasm_strong.json --scan
```

## Model

- **Model:** `mistralai/Mistral-7B-Instruct-v0.2`
- **Steering layer:** Layer 19 of 32 (`layer_frac=0.6`)
- **Extraction method:** End-of-completion (last token of full answer text)
- **Dataset pairs:** 30 contrastive question-answer pairs

## Requirements

```bash
pip install torch transformers accelerate
```
