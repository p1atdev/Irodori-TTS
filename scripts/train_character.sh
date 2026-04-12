#!/bin/bash

uv run train.py \
  --config configs/train_500m_v2_character.yaml \
  --manifest data/sticker-voice.jsonl \
  --init-checkpoint models/500m_v2.safetensors \
  --output-dir outputs/character_01 \
  $@
