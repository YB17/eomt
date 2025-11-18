import sys
from pathlib import Path

import torch

# Ensure the package root is importable when tests are executed from repo root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from training.mask_classification_loss import MaskClassificationLoss  # noqa: E402


def test_sanitize_labels_clamps_invalid_indices():
    loss = MaskClassificationLoss(
        num_points=4,
        oversample_ratio=1.0,
        importance_sample_ratio=1.0,
        mask_coefficient=1.0,
        dice_coefficient=1.0,
        class_coefficient=1.0,
        num_labels=3,
        no_object_coefficient=0.1,
    )

    labels = torch.tensor([0, 5, -2, 2])
    cleaned = loss._sanitize_labels(labels.clone(), batch_idx=7)

    assert cleaned.tolist() == [0, 2, 0, 2]
    assert torch.all((cleaned >= 0) & (cleaned < loss.num_labels))


def test_sanitize_labels_can_use_custom_limit():
    loss = MaskClassificationLoss(
        num_points=4,
        oversample_ratio=1.0,
        importance_sample_ratio=1.0,
        mask_coefficient=1.0,
        dice_coefficient=1.0,
        class_coefficient=1.0,
        num_labels=3,
        no_object_coefficient=0.1,
    )

    labels = torch.tensor([0, 4, -5, 8])
    cleaned = loss._sanitize_labels(labels.clone(), batch_idx=0, num_classes=5, context="open-vocab")

    assert cleaned.tolist() == [0, 4, 0, 4]
    assert torch.all((cleaned >= 0) & (cleaned < 5))
