from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerHungarianMatcher,
    pair_wise_dice_loss,
    pair_wise_sigmoid_cross_entropy_loss,
    sample_point,
)


class OpenVocabHungarianMatcher(Mask2FormerHungarianMatcher):
    """Hungarian matcher that combines closed and open vocabulary costs."""

    def __init__(
        self,
        lambda_mask_iou: float = 2.0,
        lambda_bce: float = 2.0,
        lambda_cls: float = 1.0,
        kl_weight: float = 0.0,
        num_points: int = 12544,
    ) -> None:
        super().__init__(cost_class=lambda_cls, cost_mask=lambda_bce, cost_dice=lambda_mask_iou, num_points=num_points)
        self.lambda_mask_iou = lambda_mask_iou
        self.lambda_bce = lambda_bce
        self.lambda_cls = lambda_cls
        self.kl_weight = kl_weight

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        class_labels: List[torch.Tensor],
        open_vocab_logits: Optional[torch.Tensor] = None,
        text_priors: Optional[torch.Tensor] = None,
        seen_mask: Optional[List[torch.Tensor]] = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        indices: list[tuple[torch.Tensor, torch.Tensor]] = []
        batch_size = masks_queries_logits.shape[0]

        for i in range(batch_size):
            pred_mask = masks_queries_logits[i]
            target_mask = mask_labels[i].to(pred_mask)
            if target_mask.dim() == 3:
                target_mask = target_mask[:, None]
            pred_mask = pred_mask[:, None]

            if target_mask.shape[0] == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            point_coordinates = torch.rand(1, self.num_points, 2, device=pred_mask.device)
            target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1)
            sampled_target_mask = sample_point(target_mask, target_coordinates, align_corners=False).squeeze(1)
            pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
            sampled_pred_mask = sample_point(pred_mask, pred_coordinates, align_corners=False).squeeze(1)

            cost_mask = pair_wise_sigmoid_cross_entropy_loss(sampled_pred_mask, sampled_target_mask)
            cost_dice = pair_wise_dice_loss(sampled_pred_mask, sampled_target_mask)

            prob_closed = class_queries_logits[i].softmax(-1)
            gt_classes = class_labels[i]
            num_targets = gt_classes.numel()
            if num_targets == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            cost_class = -prob_closed[:, gt_classes]

            if open_vocab_logits is not None:
                prob_open = open_vocab_logits[i].softmax(-1)
                gather_ids = torch.clamp(gt_classes, max=prob_open.shape[-1] - 1)
                cost_ov = -prob_open[:, gather_ids]
                cost_class = (cost_class + cost_ov) * 0.5
                if text_priors is not None and self.kl_weight > 0:
                    priors = text_priors[i]
                    if priors.dim() == 2:
                        priors = priors[gt_classes]
                    else:
                        priors = priors[:, gt_classes]
                    priors = priors.clamp(min=1e-6)
                    kl_source = prob_open[:, gather_ids].clamp(min=1e-6)
                    kl_loss = (priors * (priors.log() - kl_source.log())).sum(dim=-1)
                    cost_class = cost_class + self.kl_weight * kl_loss

            if seen_mask is not None and i < len(seen_mask):
                seen_flags = seen_mask[i].to(gt_classes.device).bool()
                if seen_flags.numel() != num_targets:
                    seen_flags = torch.ones_like(gt_classes, dtype=torch.bool)
                unseen_flags = ~seen_flags
                if unseen_flags.any() and open_vocab_logits is not None:
                    prob_open = open_vocab_logits[i].softmax(-1)
                    gather_ids = torch.clamp(gt_classes[unseen_flags], max=prob_open.shape[-1] - 1)
                    cost_class[:, unseen_flags] = -prob_open[:, gather_ids]

            total_cost = self.lambda_bce * cost_mask + self.lambda_mask_iou * cost_dice + self.lambda_cls * cost_class
            total_cost = torch.nan_to_num(total_cost, 0.0, 0.0, 0.0)
            row_ind, col_ind = linear_sum_assignment_safe(total_cost)
            indices.append((row_ind, col_ind))

        return indices


def linear_sum_assignment_safe(cost_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10, device=cost_matrix.device))
    cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10, device=cost_matrix.device))
    cost = cost_matrix.detach().cpu().numpy()
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(cost)
    return (
        torch.as_tensor(row_ind, dtype=torch.int64, device=cost_matrix.device),
        torch.as_tensor(col_ind, dtype=torch.int64, device=cost_matrix.device),
    )
