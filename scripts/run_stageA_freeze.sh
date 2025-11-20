#!/usr/bin/env bash
export CUDA_LAUNCH_BLOCKING=1  # ✅ 添加这行以获得准确的错误堆栈

python -m tools.train_net  \
 --config-file configs/stage_a.yaml \
 MODEL.BACKBONE.LORA.ENABLED false TRAINER.PRECISION "16-mixed"   \
 OUTPUT_DIR runs/coco_eomt_siglip2_ov/stageA   \
 LOGGING.WANDB.ENABLED true   \
 LOGGING.WANDB.PROJECT eomt-panoptic   \
 LOGGING.WANDB.NAME stageA-freeze   \
 LOGGING.WANDB.TAGS [stageA,coco]   \
 TRAINER.DEVICES 6   \
 TRAINER.STRATEGY "ddp_find_unused_parameters_true" \
#  TRAINER.LIMIT_VAL_BATCHES 10 \
#  DATA.BATCH_SIZE 10
