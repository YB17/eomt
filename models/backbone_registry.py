from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .backbones.siglip2_vit import SigLIP2ViTBackbone
from eomt.modules.lora import inject_lora


def _pop(d: Dict[str, Any], *keys: str, default=None):
    value = d
    for key in keys[:-1]:
        if isinstance(value, dict):
            value = value.get(key, {})
        else:
            value = getattr(value, key, {})
    last = keys[-1]
    if isinstance(value, dict):
        return value.get(last, default)
    return getattr(value, last, default)


def build_backbone(name: str, **cfg: Any) -> SigLIP2ViTBackbone:
    name = name.lower()
    backbone_cfg = cfg.copy()
    model_id = backbone_cfg.pop("MODEL_ID", backbone_cfg.pop("model_id", None))
    drop_path = backbone_cfg.pop("DROP_PATH", backbone_cfg.pop("drop_path", 0.0))
    naflex = backbone_cfg.pop("NAFLEX", backbone_cfg.pop("naflex", True))
    img_size = backbone_cfg.pop("IMG_SIZE", backbone_cfg.pop("img_size", None))
    fp16 = backbone_cfg.pop("FP16", backbone_cfg.pop("fp16", True))
    out_indices = backbone_cfg.pop("OUT_INDICES", backbone_cfg.pop("out_indices", (-1,)))

    backbone = SigLIP2ViTBackbone(
        model_id=model_id,
        out_indices=out_indices,
        drop_path=drop_path,
        naflex=naflex,
        img_size=img_size,
        fp16=fp16,
    )

    lora_cfg = backbone_cfg.pop("LORA", backbone_cfg.pop("lora", None))
    if isinstance(lora_cfg, dict) and lora_cfg.get("ENABLED", lora_cfg.get("enabled", False)):
        target_names: Iterable[str] = tuple(lora_cfg.get("TARGET", lora_cfg.get("target", ("q", "k", "v"))))
        rank = int(lora_cfg.get("RANK", lora_cfg.get("rank", 8)))
        alpha = float(lora_cfg.get("ALPHA", lora_cfg.get("alpha", 16.0)))
        last_n = int(lora_cfg.get("LAYERS_LAST_N", lora_cfg.get("layers_last_n", 8)))
        inject_lora(backbone, target_names=target_names, last_n_layers=last_n, rank=rank, alpha=alpha)
        for param in backbone.parameters():
            param.requires_grad = getattr(param, "_lora_param", False)

    return backbone
