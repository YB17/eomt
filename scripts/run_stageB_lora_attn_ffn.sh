#!/usr/bin/env bash
python -m tools.train_net  \
 --config-file configs/stage_b.yaml \
 TRAINER.PRECISION "16-mixed"   \
 OUTPUT_DIR runs/coco_eomt_siglip2_ov/stageB  \
 LOGGING.WANDB.ENABLED true   \
 LOGGING.WANDB.PROJECT eomt-panoptic   \
 LOGGING.WANDB.NAME stageB-lora   \
 LOGGING.WANDB.TAGS [stageB,coco]   \
 TRAINER.DEVICES 8   \
 TRAINER.STRATEGY "ddp_find_unused_parameters_true" \
 RESUME_FROM_STAGE_A "/home/dockeruser/ybai_ws/eomt/runs/coco_eomt_siglip2_ov/stageA/logs/version_38/checkpoints/last.ckpt"