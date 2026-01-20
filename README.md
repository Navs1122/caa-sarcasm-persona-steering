# Contrastive Activation Addition (CAA) – Sarcasm Persona Steering

This repository demonstrates **Contrastive Activation Addition (CAA)** used to steer
a model’s persona (neutral ↔ sarcastic) when explaining neural networks.

## Key points
- This is **not prompt engineering**
- The prompt stays identical across runs
- Behavior is controlled via an internal activation vector
- Vector is learned from A/B contrast pairs

## Files
- `run_caa_sarcasm.py` – End-to-end CAA extraction + generation
- `caa_neural_networks_sarcasm_strong.json` – CAA dataset (neutral vs sarcastic)

## Run

```bash
python run_caa_sarcasm.py --alpha 0.8
