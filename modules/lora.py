from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Light-weight Low-Rank Adapter for linear layers."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.weight = linear.weight
        self.bias = linear.bias

        self.A = nn.Parameter(linear.weight.new_zeros((linear.in_features, rank)))
        self.B = nn.Parameter(linear.weight.new_zeros((rank, linear.out_features)))
        self.A._lora_param = True  # type: ignore[attr-defined]
        self.B._lora_param = True  # type: ignore[attr-defined]

        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.zeros_(self.B)

        self.linear.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = torch.nn.functional.linear(x, self.weight, self.bias)
        lora_update = (x @ self.A) @ self.B * self.scaling
        return original + lora_update


@dataclass
class LoRAConfig:
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    target: Tuple[str, ...] = ("q", "k", "v")
    layers_last_n: int = 8


def _match_target(name: str, targets: Iterable[str]) -> bool:
    return any(name.endswith(target) or f".{target}" in name for target in targets)


def _extract_layer_index(name: str) -> Optional[int]:
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part.isdigit() and i > 0 and parts[i - 1] in {"layers", "blocks"}:
            try:
                return int(part)
            except ValueError:  # pragma: no cover - defensive
                continue
    return None


def inject_lora(
    module: nn.Module,
    target_names: Iterable[str] = ("q", "k", "v"),
    last_n_layers: int = 8,
    rank: int = 8,
    alpha: float = 16.0,
) -> None:
    """Replace matching Linear modules with LoRA-adapted variants."""

    linear_modules = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear) and _match_target(name, target_names):
            linear_modules.append((name, child))

    layer_indices = [_extract_layer_index(name) for name, _ in linear_modules]
    valid_indices = [idx for idx in layer_indices if idx is not None]
    max_idx = max(valid_indices) if valid_indices else -1

    for name, child in linear_modules:
        layer_idx = _extract_layer_index(name)
        if layer_idx is not None and max_idx >= 0:
            if layer_idx < max_idx - last_n_layers + 1:
                continue
        parent = module
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoRALinear(child, rank=rank, alpha=alpha))


def mark_only_lora_as_trainable(module: nn.Module) -> None:
    for param in module.parameters():
        if isinstance(param, nn.Parameter):
            param.requires_grad = getattr(param, "_lora_param", False)
