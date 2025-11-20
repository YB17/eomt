from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerHungarianMatcher,
    pair_wise_sigmoid_cross_entropy_loss,
    sample_point,
)


class OpenVocabHungarianMatcher(Mask2FormerHungarianMatcher):
    """Hungarian matcher that combines closed and open vocabulary costs."""

    def __init__(
        self,
        matcher_weights: Optional[dict[str, float]] = None,
        kl_weight: float = 0.0,
        num_points: int = 12544,
    ) -> None:
        weights = matcher_weights or {}
        lambda_mask_iou = float(weights.get("LAMBDA_MASK_IOU", 2.0))
        lambda_bce = float(weights.get("LAMBDA_BCE", 1.0))
        lambda_cls = float(weights.get("LAMBDA_CLS", 0.5))
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
                # ✅ 修复：确保空索引在正确的设备上
                indices.append((
                    torch.empty(0, dtype=torch.int64, device=pred_mask.device),
                    torch.empty(0, dtype=torch.int64, device=pred_mask.device)
                ))
                continue

            point_coordinates = torch.rand(1, self.num_points, 2, device=pred_mask.device)
            target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1)
            sampled_target_mask = sample_point(target_mask, target_coordinates, align_corners=False).squeeze(1)
            pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
            sampled_pred_mask = sample_point(pred_mask, pred_coordinates, align_corners=False).squeeze(1)

            cost_mask = pair_wise_sigmoid_cross_entropy_loss(sampled_pred_mask, sampled_target_mask)
            prob_pred = sampled_pred_mask.sigmoid()
            prob_target = sampled_target_mask
            intersection = torch.einsum("qn,tn->qt", prob_pred, prob_target)
            pred_area = prob_pred.sum(-1).unsqueeze(1)
            tgt_area = prob_target.sum(-1).unsqueeze(0)
            union = pred_area + tgt_area - intersection
            cost_iou = 1.0 - (intersection / (union + 1e-6))

            prob_closed = class_queries_logits[i].softmax(-1)
            gt_classes = class_labels[i]
            num_targets = gt_classes.numel()
            if num_targets == 0:
                # ✅ 修复：确保空索引在正确的设备上
                indices.append((
                    torch.empty(0, dtype=torch.int64, device=pred_mask.device),
                    torch.empty(0, dtype=torch.int64, device=pred_mask.device)
                ))
                continue

            cost_class = -prob_closed[:, gt_classes]

            if open_vocab_logits is not None:
                prob_open = open_vocab_logits[i].softmax(-1)
                gather_ids = torch.clamp(gt_classes, max=prob_open.shape[-1] - 1)
                cost_ov = -prob_open[:, gather_ids]
                cost_class = (cost_class + cost_ov) * 0.5
                if text_priors is not None and self.kl_weight > 0:
                    # 1. 获取教师的先验 logits（相似度）
                    priors = text_priors[i]  # (num_queries, num_classes) = (150, 133)
                 
                    # 2. 将教师的先验转换为概率分布
                    #    可以使用温度系数来控制分布的"软硬"程度
                    tau_teacher = 2.0  # 温度系数，越大分布越平滑
                    teacher_probs = F.softmax(priors / tau_teacher, dim=-1)  # (150, 133)
                    
                    # 3. 学生的预测概率（已经是归一化的）
                    student_probs = prob_open  # (150, 133)
                    
                    # 4. 计算标准 KL 散度：D_KL(teacher || student)
                    #    在类别维度（最后一维）上求和
                    #    使用 PyTorch 的 kl_div：需要 log(student) 和 teacher
                    student_log_probs = torch.log(student_probs.clamp(min=1e-8))  # (150, 133)
                    
                    # F.kl_div 计算 KL(target || input)，所以参数顺序是 (student_log, teacher)
                    # reduction='batchmean' 会自动在最后一维求和
                    kl_per_query = F.kl_div(
                        student_log_probs, 
                        teacher_probs, 
                        reduction='none'  # 不自动求和，手动控制
                    ).sum(dim=-1)  # 在类别维度求和 → (150,)
                    
                    # 5. 现在 kl_per_query 是每个 query 的总体 KL 散度
                    #    它表示该 query 的预测与教师先验的差异程度
                    #    需要扩展到 (num_queries, num_targets) 以匹配 cost_class 的形状
                    num_targets = gt_classes.numel()
                    kl_cost = kl_per_query.unsqueeze(1).expand(-1, num_targets)  # (150, 11)
                    
                    # 6. 添加到分类成本中
                    cost_class = cost_class + self.kl_weight * kl_cost  # (150, 11) + (150, 11)

            if seen_mask is not None and i < len(seen_mask):
                seen_flags = seen_mask[i].to(gt_classes.device).bool()
                if seen_flags.numel() != num_targets:
                    seen_flags = torch.ones_like(gt_classes, dtype=torch.bool)
                unseen_flags = ~seen_flags
                if unseen_flags.any() and open_vocab_logits is not None:
                    prob_open = open_vocab_logits[i].softmax(-1)
                    gather_ids = torch.clamp(gt_classes[unseen_flags], max=prob_open.shape[-1] - 1)
                    cost_class[:, unseen_flags] = -prob_open[:, gather_ids]

            total_cost = self.lambda_bce * cost_mask + self.lambda_mask_iou * cost_iou + self.lambda_cls * cost_class
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
