from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer augmented with a LoRA adapter.

    The original ``nn.Linear`` module is frozen and wrapped with a low-rank
    residual branch parameterised by matrices ``A`` and ``B``. During the
    forward pass the adapter output is scaled by ``alpha / rank`` and added to
    the frozen projection. Optionally a dropout can be applied before the
    adapter to regularise training.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        bias: str = "none",
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.weight = linear.weight
        self.weight.requires_grad_(False)

        self.base_bias: Optional[nn.Parameter]
        self.adapter_bias: Optional[nn.Parameter]
        if linear.bias is not None:
            if bias == "all":
                linear.bias.requires_grad_(True)
                self.base_bias = linear.bias
                self.adapter_bias = None
            elif bias == "lora_only":
                linear.bias.requires_grad_(False)
                self.base_bias = linear.bias
                self.adapter_bias = nn.Parameter(torch.zeros_like(linear.bias))
                self.adapter_bias._is_lora_param = True  # type: ignore[attr-defined]
            else:
                linear.bias.requires_grad_(False)
                self.base_bias = linear.bias
                self.adapter_bias = None
        else:
            self.base_bias = None
            self.adapter_bias = None

        self.A = nn.Parameter(linear.weight.new_zeros((linear.in_features, rank)))
        self.B = nn.Parameter(linear.weight.new_zeros((rank, linear.out_features)))
        self.A._is_lora_param = True  # type: ignore[attr-defined]
        self.B._is_lora_param = True  # type: ignore[attr-defined]

        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.zeros_(self.B)

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute ``x @ W + b`` plus the LoRA residual branch."""

        residual = F.linear(x, self.weight, self.base_bias)
        lora_input = self.dropout(x)
        update = (lora_input @ self.A) @ self.B
        update = update * self.scaling
        if self.adapter_bias is not None:
            update = update + self.adapter_bias
        return residual + update


@dataclass
class LoRAInjectionStats:
    """Summary of the injected LoRA layers."""

    total_trainable: int
    per_layer: List[Dict[str, bool]]


def _get_transformer_blocks(module: nn.Module) -> List[nn.Module]:
    if hasattr(module, "encoder") and hasattr(module.encoder, "layers"):
        return list(module.encoder.layers)
    if hasattr(module, "blocks"):
        return list(module.blocks)
    return []


def _replace_linear(
    parent: nn.Module,
    name: str,
    rank: int,
    alpha_scale: float,
    dropout: float,
    bias: str,
) -> bool:
    if not hasattr(parent, name):
        return False
    layer = getattr(parent, name)
    if isinstance(layer, LoRALinear):
        return True
    if not isinstance(layer, nn.Linear):
        return False
    alpha = alpha_scale * float(rank)
    lora_layer = LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout, bias=bias)
    setattr(parent, name, lora_layer)
    return True


def inject_lora(
    module: nn.Module,
    last_n_layers: int = 12,
    r_attn: int = 16,
    r_ffn: int = 32,
    alpha_scale: float = 2.0,
    dropout: float = 0.05,
    bias: str = "none",
    include_proj: bool = False,
) -> LoRAInjectionStats:
    """Inject LoRA adapters into the last ``n`` Transformer blocks.

    Parameters
    ----------
    module:
        Module containing Transformer blocks (e.g. SigLIP2 vision tower).
    last_n_layers:
        Number of final blocks receiving LoRA adapters.
    r_attn:
        Rank applied to attention projections (q/k/v and optionally out proj).
    r_ffn:
        Rank applied to feed-forward layers (fc1 and fc2).
    alpha_scale:
        Scaling factor multiplied by the rank to compute ``alpha``.
    dropout:
        Dropout probability applied before the LoRA branch.
    bias:
        Bias training strategy (``"none"``, ``"all"``, ``"lora_only"``).
    include_proj:
        If ``True`` also adapts the attention output projection.
    """

    blocks = _get_transformer_blocks(module)
    if not blocks:
        raise ValueError("No transformer blocks found for LoRA injection")

    start_idx = max(0, len(blocks) - last_n_layers)
    coverage: List[Dict[str, bool]] = []

    for idx, block in enumerate(blocks):
        info: Dict[str, bool] = {
            "q": False,
            "k": False,
            "v": False,
            "proj": False,
            "fc1": False,
            "fc2": False,
        }
        if idx < start_idx:
            coverage.append(info)
            continue

        attn = getattr(block, "self_attn", getattr(block, "attn", None))
        if attn is None:
            raise AttributeError(f"Block {idx} has no attention module")
        info["q"] = _replace_linear(attn, "q_proj", r_attn, alpha_scale, dropout, bias)
        info["k"] = _replace_linear(attn, "k_proj", r_attn, alpha_scale, dropout, bias)
        info["v"] = _replace_linear(attn, "v_proj", r_attn, alpha_scale, dropout, bias)
        if include_proj:
            info["proj"] = _replace_linear(attn, "out_proj", r_attn, alpha_scale, dropout, bias)

        mlp = getattr(block, "mlp", None)
        if mlp is None:
            raise AttributeError(f"Block {idx} has no MLP module")
        info["fc1"] = _replace_linear(mlp, "fc1", r_ffn, alpha_scale, dropout, bias)
        info["fc2"] = _replace_linear(mlp, "fc2", r_ffn, alpha_scale, dropout, bias)
        if not info["fc1"] or not info["fc2"]:
            raise RuntimeError("LoRA must cover both fc1 and fc2 in each block")

        coverage.append(info)

    for param in module.parameters():
        if isinstance(param, nn.Parameter) and not getattr(param, "_is_lora_param", False):
            param.requires_grad = False

    return summarize_lora(module, coverage=coverage)


def summarize_lora(
    module: nn.Module,
    coverage: Optional[List[Dict[str, bool]]] = None,
) -> LoRAInjectionStats:
    """Collect trainable parameter counts and per-layer coverage."""

    if coverage is None:
        blocks = _get_transformer_blocks(module)
        coverage = []
        for block in blocks:
            attn = getattr(block, "self_attn", getattr(block, "attn", None))
            mlp = getattr(block, "mlp", None)
            info = {
                "q": isinstance(getattr(attn, "q_proj", None), LoRALinear),
                "k": isinstance(getattr(attn, "k_proj", None), LoRALinear),
                "v": isinstance(getattr(attn, "v_proj", None), LoRALinear),
                "proj": isinstance(getattr(attn, "out_proj", None), LoRALinear),
                "fc1": isinstance(getattr(mlp, "fc1", None), LoRALinear),
                "fc2": isinstance(getattr(mlp, "fc2", None), LoRALinear),
            }
            coverage.append(info)

    total_trainable = sum(p.numel() for p in module.parameters() if getattr(p, "_is_lora_param", False))
    return LoRAInjectionStats(total_trainable=total_trainable, per_layer=coverage)
