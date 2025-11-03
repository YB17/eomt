from __future__ import annotations

from typing import Any, Dict, Iterable

from .backbones.siglip2_vit import SigLIP2ViTBackbone


_SIGLIP_ALIASES: Dict[str, str] = {
    "siglip2_vit_b_16_naflex": "google/siglip-so400m-patch16-384",
    "siglip2_vit_l_16_naflex": "google/siglip-large-patch16-384",
    "siglip2_vit_g_14_naflex": "google/siglip-large-patch14-384",
}


def build_backbone(name: str, **cfg: Any) -> SigLIP2ViTBackbone:
    """Instantiate a SigLIP2 backbone registered under ``name``."""

    name = name.lower()
    params = dict(cfg)

    model_id = params.pop("MODEL_ID", params.pop("model_id", None))
    if model_id is None:
        model_id = _SIGLIP_ALIASES.get(name)
    if model_id is None:
        raise ValueError("MODEL_ID must be provided for SigLIP2 backbones")

    out_indices = params.pop("OUT_INDICES", params.pop("out_indices", (-1,)))
    if isinstance(out_indices, Iterable) and not isinstance(out_indices, (str, bytes)):
        out_indices = tuple(out_indices)
    else:
        out_indices = (int(out_indices),)

    lora_cfg = params.pop("LORA", params.pop("lora", None))

    backbone = SigLIP2ViTBackbone(
        model_id=model_id,
        out_indices=out_indices,
        drop_path=float(params.pop("DROP_PATH", params.pop("drop_path", 0.0))),
        naflex=bool(params.pop("NAFLEX", params.pop("naflex", True))),
        img_size=params.pop("IMG_SIZE", params.pop("img_size", None)),
        fp16=bool(params.pop("FP16", params.pop("fp16", True))),
        lora_cfg=lora_cfg,
    )

    return backbone
