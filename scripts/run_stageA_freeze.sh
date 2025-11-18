#!/usr/bin/env bash
python -m tools.train_net  \
 --config-file configs/stage_a.yaml \
 MODEL.BACKBONE.LORA.ENABLED false TRAINER.PRECISION "16-mixed"   \
 OUTPUT_DIR runs/coco_eomt_siglip2_ov/stageA   \
 LOGGING.WANDB.ENABLED true   \
 LOGGING.WANDB.PROJECT eomt-panoptic   \
 LOGGING.WANDB.NAME stageA-freeze   \
 LOGGING.WANDB.TAGS [stageA,coco]   \
 TRAINER.DEVICES 8   \
 TRAINER.STRATEGY "ddp_find_unused_parameters_true"
