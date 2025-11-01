#!/usr/bin/env bash
set -euo pipefail

python tools/train_net.py \
  --config-file configs/coco_panoptic_siglip2_eomt_ov.yaml \
  MODEL.BACKBONE.LORA.ENABLED true \
  MODEL.BACKBONE.LORA.RANK 8 \
  MODEL.BACKBONE.LORA.TARGET "['q_proj','k_proj','v_proj']" \
  OUTPUT_DIR "${1:-runs/coco_eomt_siglip2_ov/stageB_lora8}"
