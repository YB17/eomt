from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path
from typing import Optional

import torch

base_path = Path(__file__).resolve()
sys.path.append(str(base_path.parents[2]))
sys.path.append(str(base_path.parents[1]))

if "cv2" not in sys.modules:
    cv2_stub = types.SimpleNamespace(
        RETR_EXTERNAL=0,
        CHAIN_APPROX_SIMPLE=0,
        FONT_HERSHEY_SIMPLEX=0,
        COLOR_BGR2RGB=0,
        LINE_AA=0,
        moments=lambda *args, **kwargs: {},
        findContours=lambda *args, **kwargs: ([], None),
        drawContours=lambda *args, **kwargs: None,
        getTextSize=lambda *args, **kwargs: ((0, 0), None),
        rectangle=lambda *args, **kwargs: None,
        putText=lambda *args, **kwargs: None,
        cvtColor=lambda *args, **kwargs: None,
    )
    cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)  # type: ignore[attr-defined]
    sys.modules["cv2"] = cv2_stub

from eomt.models.backbones.siglip2_vit import SigLIP2ViTBackbone
from eomt.models.eomt import EoMT
from eomt.models.open_vocab_head import OpenVocabHead
from training.lightning_module import LightningModule


class DummyTokenizer:
    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        max_len = max(len(t) for t in texts)
        input_ids = torch.zeros((len(texts), max_len), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DummyTextModel(torch.nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(32, hidden_dim)
        self.proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.config = type("Cfg", (), {"hidden_size": hidden_dim})()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):  # type: ignore[override]
        embeds = self.emb(input_ids)
        pooled = embeds.mean(dim=1)
        pooled = self.proj(pooled)
        return type("Outputs", (), {"pooler_output": pooled, "last_hidden_state": embeds})()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_siglip2_backbone_open_vocab_and_utils():
    backbone = SigLIP2ViTBackbone(model_id=None, img_size=224, fp16=False)
    dummy = torch.randn(2, 3, 512, 512)
    features = backbone.forward_features(dummy)
    assert features[-1].shape[0] == 2

    text_model = DummyTextModel(hidden_dim=64)
    head = OpenVocabHead(
        text_model_id="dummy",
        templates_things=["a {}"],
        templates_stuff=["a patch of {}"],
        class_names_things=["person", "car"],
        class_names_stuff=["sky"],
        text_model=text_model,
        tokenizer=DummyTokenizer(),
    )

    mask_logits = torch.randn(2, 10, 32, 32)
    patch_tokens = torch.randn(2, 1024, 64)
    ov_out = head(mask_logits, patch_tokens)
    assert ov_out["logits"].shape[:2] == (2, 10)
    assert ov_out["similarity"].shape == ov_out["logits"].shape

    # Inject LoRA and ensure final blocks include adapters for q/k/v and FFNs
    lora_backbone = SigLIP2ViTBackbone(
        model_id=None,
        img_size=224,
        fp16=False,
        lora_cfg={
            "ENABLED": True,
            "LAST_N_LAYERS": 2,
            "RANK_ATTN": 4,
            "RANK_FFN": 8,
            "ALPHA_SCALE": 2.0,
        },
    )
    summary = lora_backbone.get_lora_summary()
    assert summary is not None
    assert summary.total_trainable > 0
    assert summary.per_layer[-1]["q"] and summary.per_layer[-1]["k"] and summary.per_layer[-1]["v"]
    assert summary.per_layer[-1]["fc1"] and summary.per_layer[-1]["fc2"]

    # Polynomial mask annealing with factor 0.9
    annealer = types.SimpleNamespace(
        poly_power=0.9,
        device=torch.device("cpu"),
        network=types.SimpleNamespace(attn_mask_probs=torch.ones(4, dtype=torch.float32)),
    )
    start = LightningModule.mask_annealing(annealer, 0, 0, 10)
    mid = LightningModule.mask_annealing(annealer, 0, 5, 10)
    end = LightningModule.mask_annealing(annealer, 0, 10, 10)
    assert torch.allclose(start, torch.ones_like(start))
    assert end.item() == 0.0
    assert mid.item() < 1.0

    # Inference should disable masked attention entirely
    encoder = types.SimpleNamespace(backbone=backbone, pixel_mean=backbone.pixel_mean, pixel_std=backbone.pixel_std)
    model = EoMT(encoder=encoder, num_classes=3, num_q=8, num_blocks=2, masked_attn_enabled=True)
    model.eval()
    calls: list[Optional[torch.Tensor]] = []
    original_attn = model._attn

    def spy(self, module, x, mask):
        calls.append(mask)
        return original_attn(module, x, mask)

    model._attn = types.MethodType(spy, model)
    with torch.no_grad():
        model(torch.randn(1, 3, 224, 224))
    assert calls, "attention should be invoked"
    assert all(mask is None for mask in calls)
