#!/usr/bin/env bash
set -euo pipefail

python3 -m CopiedFromYam.ASR.newCondWhisper.train_cpcondwhisper_latent \
  --train-manifest /storage/tal/thesis/condwhisper_manifests/train_cpwidth.jsonl \
  --val-manifest /storage/tal/thesis/condwhisper_manifests/val_cpwidth.jsonl \
  --test-manifest /storage/tal/thesis/condwhisper_manifests/test_cpwidth.jsonl \
  --model-name openai/whisper-small \
  --output-dir /storage/tal/thesis/cpcondwhisper_latent \
  --epochs 10 \
  --batch-size 6 \
  --lr 2e-4 \
  --num-latent-blocks 4 \
  --d-cond 128 \
  --num-heads 8 \
  --lambda-latent-delta 0.01 \
  --patience 4
