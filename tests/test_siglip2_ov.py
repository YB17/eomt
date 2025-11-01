import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch

from eomt.models.backbones.siglip2_vit import SigLIP2ViTBackbone
from eomt.models.open_vocab_head import OpenVocabHead


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
        self.config = type("Cfg", (), {"hidden_size": hidden_dim, "projection_size": hidden_dim})()

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


def test_siglip2_backbone_and_open_vocab_head():
    backbone = SigLIP2ViTBackbone(model_id=None, img_size=224, fp16=False)
    dummy = torch.randn(2, 3, 512, 512)
    features = backbone.forward_features(dummy)
    assert features[-1].shape[0] == 2

    text_model = DummyTextModel(hidden_dim=64)

    head = OpenVocabHead(
        text_model_id="dummy",
        templates_things=["a {}"],
        templates_stuff=["a {}"],
        class_names_things=["person"],
        class_names_stuff=["sky"],
        text_model=text_model,
        tokenizer=DummyTokenizer(),
    )

    mask_logits = torch.randn(2, 10, 32, 32)
    patch_tokens = features[-1]
    ov_logits, sims = head(mask_logits, patch_tokens)
    assert ov_logits.shape[:2] == (2, 10)
    assert ov_logits.shape[-1] == head.text_features.shape[0]

    # Inject LoRA and ensure the number of trainable params drops
    backbone_lora = SigLIP2ViTBackbone(
        model_id=None,
        img_size=224,
        fp16=False,
    )
    from eomt.modules.lora import inject_lora

    total_before = count_trainable_parameters(backbone_lora)
    inject_lora(backbone_lora, target_names=("q_proj", "k_proj", "v_proj"), last_n_layers=1, rank=4, alpha=8)
    total_after = count_trainable_parameters(backbone_lora)
    assert total_after < total_before
    assert total_after > 0
