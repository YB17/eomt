# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import Dict, List, Optional
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)

from engine.matcher import OpenVocabHungarianMatcher


class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
        open_vocab_enabled: bool = False,
        open_vocab_kl_weight: float = 0.0,
        open_vocab_kl_teacher_temp: float = 2.0,
        matcher_weights: Optional[dict[str, float]] = None,
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient
        self.open_vocab_enabled = open_vocab_enabled
        self.open_vocab_kl_weight = open_vocab_kl_weight
        self.open_vocab_kl_teacher_temp = open_vocab_kl_teacher_temp
        self.matcher_weights = matcher_weights or {}
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        if self.open_vocab_enabled:
            self.matcher: Mask2FormerHungarianMatcher = OpenVocabHungarianMatcher(
                matcher_weights=self.matcher_weights,
                kl_weight=open_vocab_kl_weight,
                num_points=num_points,
            )
        else:
            self.matcher = Mask2FormerHungarianMatcher(
                num_points=num_points,
                cost_mask=mask_coefficient,
                cost_dice=dice_coefficient,
                cost_class=class_coefficient,
            )

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
        mask_labels = [target["masks"].to(masks_queries_logits.dtype) for target in targets]
        class_labels = [target["labels"].long() for target in targets]
        # 只用is_thing作为标签，thing=1, stuff=0
        # class_labels = [target["is_thing"].long() for target in targets]

        if isinstance(self.matcher, OpenVocabHungarianMatcher):
            indices = self.matcher(
                masks_queries_logits=masks_queries_logits,
                mask_labels=mask_labels,
                class_queries_logits=class_queries_logits,
                class_labels=class_labels,
                open_vocab_logits=open_vocab_logits,
                text_priors=text_priors,
                seen_mask=seen_mask,
            )
        else:
            indices = self.matcher(
                masks_queries_logits=masks_queries_logits,
                mask_labels=mask_labels,
                class_queries_logits=class_queries_logits,
                class_labels=class_labels,
            )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)
        ov_losses = {}
        if open_vocab_logits is not None and class_queries_logits is not None:
            ov_losses = self._open_vocab_losses(
                open_vocab_logits=open_vocab_logits,
                targets=targets,
                matched_indices=indices,
                text_priors=text_priors,
            )

        return {**loss_masks, **loss_classes, **ov_losses}

    def _open_vocab_losses(
        self,
        open_vocab_logits: torch.Tensor,
        targets: List[dict],
        matched_indices: List[tuple[torch.Tensor, torch.Tensor]],
        text_priors: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        device = open_vocab_logits.device
        ce_loss = torch.tensor(0.0, device=device)
        kl_loss = torch.tensor(0.0, device=device)
        ce_count = 0
        kl_count = 0

        for batch_idx, (src_idx, tgt_idx) in enumerate(matched_indices):
            if src_idx.numel() == 0:
                continue
            logits = open_vocab_logits[batch_idx, src_idx]
            labels = targets[batch_idx]["labels"][tgt_idx].to(device)
            
            # ✅ 关键修复：确保 labels 在有效范围内
            num_classes = logits.shape[-1]
            invalid_mask = (labels < 0) | (labels >= num_classes)
            if invalid_mask.any():
                import logging
                logger = logging.getLogger(__name__)
                invalid_labels = labels[invalid_mask].tolist()
                logger.error(
                    f"Batch {batch_idx}: Found {invalid_mask.sum().item()} invalid labels "
                    f"{invalid_labels} (valid range: [0, {num_classes-1}]). Clamping to valid range."
                )
                # 将无效 labels 映射到0（背景）
                labels = torch.where(invalid_mask, torch.zeros_like(labels), labels)
        
            ce_loss = ce_loss + F.cross_entropy(logits, labels, reduction="sum")
            ce_count += len(src_idx)

            if text_priors is not None and self.open_vocab_kl_weight > 0:
                priors = text_priors[batch_idx]
                # ✅ 确保 priors 的维度正确
                if priors.shape[-1] != num_classes:
                    logger.warning(
                        f"text_priors shape mismatch: {priors.shape} vs expected (..., {num_classes})"
                    )
                    continue
                if priors.dim() == 3:
                    priors = priors[src_idx]
                elif priors.dim() == 2:
                    priors = priors[tgt_idx]
                elif priors.dim() == 1:
                    priors = priors.unsqueeze(0).expand_as(logits)
                else:
                    priors = priors.expand_as(logits)
                priors = priors.detach()
                tau_text = getattr(self, "open_vocab_kl_teacher_temp", 2.0)
                priors = priors / max(tau_text, 1e-6)
                reject_mask = torch.isinf(logits)
                if reject_mask.any():
                    priors = priors.masked_fill(reject_mask, float("-inf"))
                teacher_probs = F.softmax(priors, dim=-1)
                student_log = F.log_softmax(logits, dim=-1)
                kl_loss = kl_loss + F.kl_div(student_log, teacher_probs, reduction="sum")
                kl_count += logits.size(0)

        if ce_count > 0:
            ce_loss = ce_loss / ce_count
        if kl_count > 0:
            kl_loss = kl_loss / kl_count

        losses: dict[str, torch.Tensor] = {
            "loss_open_vocab_ce": ce_loss,
        }

        if self.open_vocab_kl_weight > 0:
            losses["loss_open_vocab_kl"] = kl_loss * self.open_vocab_kl_weight

        return losses

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            elif loss_key.startswith("loss_open_vocab"):
                weighted_loss = loss * self.class_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total  # type: ignore
