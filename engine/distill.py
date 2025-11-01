from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F


def compute_feat_align(
    student_feats: Iterable[torch.Tensor],
    teacher_feats: Iterable[torch.Tensor],
    weight: float,
) -> torch.Tensor:
    """Align student features with frozen teacher features using L2 distance."""

    student_feats = list(student_feats)
    teacher_feats = list(teacher_feats)
    if weight <= 0 or not student_feats:
        device = student_feats[0].device if student_feats else (teacher_feats[0].device if teacher_feats else "cpu")
        return torch.tensor(0.0, device=device)

    losses: List[torch.Tensor] = []
    for sf, tf in zip(student_feats, teacher_feats):
        sf_norm = F.normalize(sf, dim=-1)
        tf_norm = F.normalize(tf, dim=-1)
        losses.append(F.mse_loss(sf_norm, tf_norm))
    if not losses:
        return torch.tensor(0.0, device=student_feats[0].device)
    return weight * torch.stack(losses).mean()


def compute_itc(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    temperature: float,
    weight: float,
) -> torch.Tensor:
    """Compute a lightweight image-text contrastive loss."""

    if weight <= 0 or image_embs.numel() == 0 or text_embs.numel() == 0:
        return torch.tensor(0.0, device=image_embs.device if image_embs.numel() else text_embs.device)

    img = F.normalize(image_embs, dim=-1)
    txt = F.normalize(text_embs, dim=-1)
    logits = img @ txt.t() / max(temperature, 1e-6)
    labels = torch.arange(img.size(0), device=img.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return weight * 0.5 * (loss_i + loss_t)
