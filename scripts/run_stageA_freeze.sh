#!/usr/bin/env bash
set -euo pipefail

python tools/train_net.py \
  --config-file configs/coco_panoptic_siglip2_eomt_ov.yaml \
  MODEL.BACKBONE.LORA.ENABLED false \
  MODEL.BACKBONE.FREEZE true \
  OUTPUT_DIR "${1:-runs/coco_eomt_siglip2_ov/stageA}"
