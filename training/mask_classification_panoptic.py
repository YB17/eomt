# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule

# T ≈ 6159 × 100 = 615 900 total steps
# start steps ≈ [0.2T, 0.5T, 0.8T] ≈ [123 180, 307 950, 492 720]
# end steps   ≈ [0.4T, 0.7T, 1.0T] ≈ [246 360, 431 130, 615 900]

class MaskClassificationPanoptic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        stuff_classes: list[int],
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        llrd: float = 0.8,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.2,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        load_ckpt_class_head: bool = True,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = stuff_classes

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            # num_labels= 2,
            no_object_coefficient=no_object_coefficient,
        )

        thing_classes = [i for i in range(num_classes) if i not in stuff_classes]
        self.init_metrics_panoptic(
            thing_classes, stuff_classes, self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1
        )

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(transformed_imgs)

        is_crowds = [target["is_crowd"] for target in targets]
        targets = self.to_per_pixel_targets_panoptic(targets)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits, img_sizes
            )
            preds = self.to_per_pixel_preds_panoptic(
                mask_logits,
                class_logits,
                self.stuff_classes,
                self.mask_thresh,
                self.overlap_thresh,
            )
            self.update_metrics_panoptic(preds, targets, is_crowds, i)
        if batch_idx % 5 == 0:  # 每5个batch可视化一次，或者完全禁用
            self.plot_panoptic(
                img=imgs[0],  # 第一张图像
                targets=targets[0],  # 对应的target
                mask_logits=mask_logits_per_layer[-1][0],  # 最后一个block的第一张图像的mask logits
                class_logits=class_logits_per_layer[-1][0],  # 最后一个block的第一张图像的class logits
                log_prefix="val",
                block_idx=len(mask_logits_per_layer) - 1,  # 最后一个block
                batch_idx=batch_idx,
            )
    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_panoptic("val")

    def on_validation_end(self):
        self._on_eval_end_panoptic("val")
