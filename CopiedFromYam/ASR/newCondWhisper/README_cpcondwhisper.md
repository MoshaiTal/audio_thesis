# Pred-only CP-conditioned Whisper (latent Condformer-style)

This setup keeps **`pred`** as the only acoustic input and uses **CP outputs** (`lower`, `upper`) as an explicit prior.

## Files
- `cpcondwhisper_blocks.py` — model components
- `train_cpcondwhisper_latent.py` — train / validate / test script
- `run_cpcondwhisper_latent.sh` — example launch command

## Method
1. Run Whisper encoder on `pred`
2. Build a CP conditioner from `pred`, `lower`, `upper`, and `width = |upper-lower|`
3. Insert a stack of **latent conditional self-attention blocks** between the frozen Whisper encoder and decoder
4. Train with ASR loss, plus a small latent-delta regularizer to keep the conditioned latent state close to the original one

This is closer to the Condformer paper than the earlier reverb-fusion path, while still using only dereverb + CP at inference.

## Expected behavior at initialization
The conditioning blocks are initialized as near-identity, so the model starts close to **plain Whisper-on-pred**.

## What to monitor
- `initial_summary.json`
- `history.json`
- `final_summary.json`
- `final_report.txt`

Most important metric:
- `best_val_delta_vs_pred`

Negative is good: it means the conditioned model beats the plain `pred` baseline on validation WER.
