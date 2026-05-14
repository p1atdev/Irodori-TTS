#!/bin/bash

INIT_CHECKPOINT="models/500m_v3.safetensors"
# INIT_CHECKPOINT="models/500m_v3_character.safetensors"
OUTPUT_DIR="outputs/v3_character_38"

uv run train.py \
  --config configs/train_500m_v3_character.yaml \
  --manifest data/sticker-voice-3.jsonl \
  --init-checkpoint $INIT_CHECKPOINT \
  --output-dir $OUTPUT_DIR \
  $@
