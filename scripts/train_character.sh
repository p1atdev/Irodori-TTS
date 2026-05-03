#!/bin/bash

INIT_CHECKPOINT="models/500m_v2.safetensors"
# INIT_CHECKPOINT="models/500m_v2_character.safetensors"
# INIT_CHECKPOINT="outputs/character_07/checkpoint_best_val_loss_0003000_0.905405.pt"
OUTPUT_DIR="outputs/character_24"

uv run train.py \
  --config configs/train_500m_v2_character.yaml \
  --manifest data/sticker-voice-2.jsonl \
  --init-checkpoint $INIT_CHECKPOINT \
  --output-dir $OUTPUT_DIR \
  $@
