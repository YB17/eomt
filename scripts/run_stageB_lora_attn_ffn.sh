#!/usr/bin/env bash
set -euo pipefail

python tools/train_net.py \
  --config-file configs/coco_panoptic_siglip2_eomt_ov.yaml \
  MODEL.BACKBONE.LORA.ENABLED true \
  MODEL.BACKBONE.LORA.LAST_N_LAYERS 12 \
  MODEL.BACKBONE.LORA.RANK_ATTN 16 \
  MODEL.BACKBONE.LORA.RANK_FFN 32 \
  OUTPUT_DIR "${1:-runs/coco_eomt_siglip2_ov/stageB}"
