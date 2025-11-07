# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule


class MaskClassificationLossCoarse(MaskClassificationLoss):
    """使用粗粒度分类的Loss函数（things/stuff二分类）"""
    
    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
        open_vocab_logits: Optional[torch.Tensor] = None,
        text_priors: Optional[torch.Tensor] = None,
        seen_mask: Optional[List[torch.Tensor]] = None,
    ):
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        # 使用is_thing作为标签：thing=1, stuff=0
        class_labels = [target["is_thing"].long() for target in targets]

        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)

        return {**loss_masks, **loss_classes}


class MaskClassificationPanopticCoarse(LightningModule):
    """粗粒度全景分割训练模块（things/stuff二分类）"""
    
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,  # 保留原参数但实际使用2类
        stuff_classes: list[int],
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        lr_lora: Optional[float] = None,
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
        load_ckpt_class_head: bool = False,  # 默认不加载类别头权重
    ):
        # 修改网络的类别头输出维度为2（thing/stuff）
        if hasattr(network, 'class_head'):
            # 保存原来的权重维度信息
            original_out_features = network.class_head.out_features
            # 重新初始化为2类+1个no-object类
            network.class_head = nn.Linear(network.class_head.in_features, 3)
            print(f"修改类别头输出维度：{original_out_features} -> 3 (thing/stuff/no-object)")
        
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=2,  # 强制使用2类：thing=1, stuff=0
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            lr_lora=lr_lora,
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
        self.stuff_classes = [0]  # stuff对应类别0
        self.thing_classes = [1]  # thing对应类别1

        # 使用粗粒度Loss函数，只有2个类别
        self.criterion = MaskClassificationLossCoarse(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=2,  # 只有2个类别：thing(1) 和 stuff(0)
            no_object_coefficient=no_object_coefficient,
        )

        # 初始化粗粒度评估指标
        self.init_metrics_panoptic(
            self.thing_classes, 
            self.stuff_classes, 
            self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1
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
        outputs = self(transformed_imgs)
        mask_logits_per_layer = outputs["mask_logits"]
        class_logits_per_layer = outputs["class_logits"]

        is_crowds = [target["is_crowd"] for target in targets]
        targets = self.to_per_pixel_targets_panoptic_coarse(targets)

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

    def to_per_pixel_targets_panoptic_coarse(self, targets):
        """将targets转换为粗粒度的per-pixel目标"""
        coarse_targets = []
        for target in targets:
            # 将原始标签转换为粗粒度标签
            labels = target["labels"]
            is_thing = target["is_thing"]
            
            # 创建新的粗粒度标签：thing=1, stuff=0
            coarse_labels = is_thing.long()
            
            coarse_target = target.copy()
            coarse_target["labels"] = coarse_labels
            coarse_targets.append(coarse_target)
        
        # 调用父类方法处理per-pixel转换
        return super().to_per_pixel_targets_panoptic(coarse_targets)

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_panoptic("val")

    def on_validation_end(self):
        self._on_eval_end_panoptic("val") 