#!/bin/bash

DATASET_DIR=/home/plat/WDC20/data/audio/sticker-voice-3

uv run prepare_manifest.py \
  --dataset $DATASET_DIR \
  --split train \
  --audio-column audio \
  --text-column text \
  --image-column image \
  --output-manifest data/sticker-voice-3.jsonl \
  --latent-dir data/latents/sticker-voice-3 \
  --num-gpus 1 \
  --merge-output

