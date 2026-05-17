 #!/bin/bash

INIT_CHECKPOINT="models/500m_v3.safetensors"
OUTPUT_DIR="outputs/v3_zunsasa_03"
# MANIFEST="data/amitaro.jsonl"
MANIFEST="data/zunda-sasa.jsonl"

 
 uv run train.py \
    --config configs/train_500m_v3_speaker_inversion.yaml \
    --manifest $MANIFEST \
    --init-checkpoint $INIT_CHECKPOINT \
    --output-dir $OUTPUT_DIR
