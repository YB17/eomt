from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matcher import OpenVocabHungarianMatcher


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.sigmoid()
    numerator = 2 * (pred * target).sum(dim=(-1, -2))
    denominator = pred.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) + 1e-6
    loss = 1 - numerator / denominator
    return loss.mean()


def bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(pred, target, reduction="mean")


class OpenVocabCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        no_object_weight: float,
        matcher: OpenVocabHungarianMatcher,
        mask_weight: float,
        dice_weight: float,
        cls_weight: float,
        ov_kl_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.cls_weight = cls_weight
        self.ov_kl_weight = ov_kl_weight

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = no_object_weight
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self,
        outputs: Dict[str, List[torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        mask_logits = outputs["mask_logits"][-1]
        class_logits = outputs["class_logits"][-1]
        ov_logits = outputs.get("open_vocab_logits", [None])[-1]

        mask_stack = torch.stack([t["masks"].float() for t in targets])
        label_stack = torch.stack([t["labels"].long() for t in targets])

        matched = self.matcher(
            masks_queries_logits=mask_logits,
            class_queries_logits=class_logits,
            mask_labels=mask_stack,
            class_labels=label_stack,
            open_vocab_logits=ov_logits,
        )

        loss_mask = mask_logits.new_tensor(0.0)
        loss_dice = mask_logits.new_tensor(0.0)
        for batch_idx, ((src_idx, tgt_idx), tgt) in enumerate(zip(matched, targets)):
            if len(src_idx) == 0:
                continue
            pred_masks = mask_logits[batch_idx, src_idx]
            target_masks = tgt["masks"].float()[tgt_idx]
            loss_mask = loss_mask + bce_loss(pred_masks, target_masks)
            loss_dice = loss_dice + dice_loss(pred_masks, target_masks)
        if len(matched) > 0:
            loss_mask = loss_mask / max(len(matched), 1)
            loss_dice = loss_dice / max(len(matched), 1)

        loss_cls = self._classification_loss(class_logits, targets, matched, ov_logits)

        return {
            "loss_mask": self.mask_weight * loss_mask,
            "loss_dice": self.dice_weight * loss_dice,
            "loss_cls": self.cls_weight * loss_cls,
        }

    def _classification_loss(
        self,
        class_logits: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        matched_indices: List[Tuple[torch.Tensor, torch.Tensor]],
        ov_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bs, num_queries, num_classes_plus_one = class_logits.shape
        target_classes = torch.full(
            (bs, num_queries),
            fill_value=self.num_classes,
            dtype=torch.long,
            device=class_logits.device,
        )
        for batch_idx, (src_idx, tgt_idx) in enumerate(matched_indices):
            if len(src_idx) == 0:
                continue
            target_classes[batch_idx, src_idx] = targets[batch_idx]["labels"][tgt_idx]

        loss = F.cross_entropy(
            class_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
        )

        if ov_logits is not None:
            ov_classes = torch.full(
                (bs, ov_logits.size(1)),
                fill_value=-1,
                dtype=torch.long,
                device=ov_logits.device,
            )
            for batch_idx, (src_idx, tgt_idx) in enumerate(matched_indices):
                if len(src_idx) == 0:
                    continue
                ov_classes[batch_idx, src_idx] = targets[batch_idx]["labels"][tgt_idx]
            mask = ov_classes >= 0
            if mask.any():
                selected = ov_logits[mask]
                target_ids = ov_classes[mask]
                loss = loss + F.cross_entropy(selected, target_ids, reduction="mean")

        return loss
