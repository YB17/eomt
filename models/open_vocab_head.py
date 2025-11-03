from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore


@dataclass
class VocabularyTemplates:
    things: Sequence[str]
    stuff: Sequence[str]
    multilingual: Optional[Sequence[str]] = None


class OpenVocabHead(nn.Module):
    """Open-vocabulary classification with SigLIP2 text embeddings."""

    def __init__(
        self,
        text_model_id: str,
        templates_things: Sequence[str],
        templates_stuff: Sequence[str],
        temperature: float = 0.02,
        gamma: float = 1.0,
        calibration_bias: float = 0.0,
        energy_reject_thr: float = float("-inf"),
        energy_type: str = "max",
        class_names_things: Optional[Sequence[str]] = None,
        class_names_stuff: Optional[Sequence[str]] = None,
        synonyms: Optional[Dict[str, Sequence[str]]] = None,
        multilingual: bool = False,
        multilingual_templates: Optional[Sequence[str]] = None,
        per_class_bias: Optional[torch.Tensor] = None,
        text_model: Optional[nn.Module] = None,
        tokenizer: Optional[object] = None,
    ) -> None:
        super().__init__()
        if text_model is None and tokenizer is None and (AutoModel is None or AutoTokenizer is None):
            raise ImportError(
                "transformers>=4.43.0 is required for OpenVocabHead. Install via requirements_extra.txt."
            )

        self.text_model_id = text_model_id
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.logit_bias = nn.Parameter(torch.tensor(float(calibration_bias)))
        self.gamma = float(gamma)
        self.energy_reject_thr = float(energy_reject_thr)
        self.energy_type = energy_type
        self.synonyms = {k: list(v) for k, v in (synonyms or {}).items()}
        self.multilingual = multilingual
        self.multilingual_templates = list(multilingual_templates or [])

        if per_class_bias is not None:
            if not isinstance(per_class_bias, torch.Tensor):
                per_class_bias = torch.tensor(per_class_bias, dtype=torch.float32)
            self.register_buffer("per_class_bias", per_class_bias.float())
        else:
            self.register_buffer("per_class_bias", None)

        self.text_model = text_model or AutoModel.from_pretrained(text_model_id, trust_remote_code=True)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_id, trust_remote_code=True)

        self.class_names_things = list(class_names_things or [])
        self.class_names_stuff = list(class_names_stuff or [])
        self.templates_things = list(templates_things)
        self.templates_stuff = list(templates_stuff)

        self.register_buffer("text_features", torch.empty(0), persistent=True)
        self.register_buffer("class_splits", torch.zeros(2, dtype=torch.long), persistent=True)
        self.text_projection: Optional[nn.Linear] = None
        self.class_name_list: List[str] = []
        if self.class_names_things or self.class_names_stuff:
            self._build_text_features()

    # ------------------------------------------------------------------
    # text encoding helpers
    # ------------------------------------------------------------------
    def encode_text(self, names: Sequence[str], templates: Sequence[str]) -> torch.Tensor:
        prompts: List[str] = []
        for name in names:
            prompts.extend(template.format(name) for template in templates)
        if self.multilingual and self.multilingual_templates:
            for name in names:
                prompts.extend(template.format(name) for template in self.multilingual_templates)
        tokenized = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.text_model.device) for k, v in tokenized.items()}
        with torch.no_grad():
            outputs = self.text_model(**tokenized)
        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            feats = outputs.text_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        else:
            hidden = outputs.last_hidden_state
            feats = hidden[:, -1, :]
        feats = F.normalize(feats, dim=-1)
        num_templates = len(prompts) // len(names)
        feats = feats.view(len(names), num_templates, -1).mean(dim=1)
        return feats

    def _encode_with_synonyms(self, name: str, templates: Sequence[str]) -> torch.Tensor:
        variants = [name] + self.synonyms.get(name, [])
        variant_feats = []
        for variant in variants:
            encoded = self.encode_text([variant], templates)[0]
            variant_feats.append(encoded)
        stacked = torch.stack(variant_feats)
        return F.normalize(stacked.mean(dim=0, keepdim=True), dim=-1)[0]

    def _build_text_features(self) -> None:
        device = self.temperature.device
        all_features: List[torch.Tensor] = []
        class_order: List[str] = []

        def encode_group(names: Sequence[str], templates: Sequence[str]) -> torch.Tensor:
            group_features: List[torch.Tensor] = []
            for base_name in names:
                group_features.append(self._encode_with_synonyms(base_name, templates))
                class_order.append(base_name)
            if group_features:
                return torch.stack(group_features)
            return torch.empty(0, self.text_model.config.hidden_size, device=device)

        things = encode_group(self.class_names_things, self.templates_things)
        stuff = encode_group(self.class_names_stuff, self.templates_stuff)

        if things.numel() > 0:
            all_features.append(things)
        if stuff.numel() > 0:
            all_features.append(stuff)

        if all_features:
            text_features = torch.cat(all_features, dim=0).to(device)
            text_features = F.normalize(text_features, dim=-1)
            self.register_buffer("text_features", text_features, persistent=True)
            self.register_buffer(
                "class_splits",
                torch.tensor([things.shape[0], text_features.shape[0]], dtype=torch.long, device=device),
                persistent=True,
            )
            self.class_name_list = class_order
        else:
            self.register_buffer("text_features", torch.empty(0, device=device), persistent=True)
            self.register_buffer("class_splits", torch.tensor([0, 0], dtype=torch.long, device=device), persistent=True)
            self.class_name_list = []

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        mask_logits: torch.Tensor,
        patch_tokens: torch.Tensor,
        temperature: Optional[float] = None,
        calibration_bias: Optional[float] = None,
        per_class_bias: Optional[torch.Tensor] = None,
        energy_threshold: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.text_features.numel() == 0:
            raise RuntimeError("Text features have not been initialised. Provide class names at construction time.")

        B, Q = mask_logits.shape[:2]
        spatial = mask_logits.shape[2:]
        masks_flat = mask_logits.view(B, Q, -1)
        gamma_val = float(gamma) if gamma is not None else self.gamma
        weights = torch.softmax(gamma_val * masks_flat, dim=-1)
        patch_tokens = patch_tokens.view(B, -1, patch_tokens.shape[-1])
        vision_feats = weights @ patch_tokens
        vision_feats = F.normalize(vision_feats, dim=-1)

        text_feats = self.text_features.to(patch_tokens.device, dtype=patch_tokens.dtype)
        if text_feats.shape[-1] != vision_feats.shape[-1]:
            if self.text_projection is None:
                self.text_projection = nn.Linear(text_feats.shape[-1], vision_feats.shape[-1], bias=False)
                nn.init.normal_(self.text_projection.weight, std=0.02)
                self.text_projection.to(patch_tokens.device, dtype=patch_tokens.dtype)
            text_feats = self.text_projection(text_feats)
        text_feats = F.normalize(text_feats, dim=-1)

        sims = vision_feats @ text_feats.t()
        tau = (self.temperature if temperature is None else torch.tensor(temperature, device=sims.device)).clamp(min=1e-6)
        logits = sims / tau

        bias_term = self.logit_bias if calibration_bias is None else torch.tensor(calibration_bias, device=logits.device)
        logits = logits + bias_term

        if per_class_bias is None:
            bias_vec = self.per_class_bias
        else:
            bias_vec = per_class_bias.to(logits.device)
        if bias_vec is not None:
            logits = logits + bias_vec

        energy_thr = self.energy_reject_thr if energy_threshold is None else energy_threshold
        if self.energy_type == "logsumexp":
            energy = torch.logsumexp(logits, dim=-1)
        else:
            energy = logits.max(dim=-1).values
        if energy_thr != float("-inf"):
            reject_mask = energy < energy_thr
            logits = logits.masked_fill(reject_mask.unsqueeze(-1), float("-inf"))

        return {
            "logits": logits,
            "similarity": sims,
            "tau": tau,
            "bias": bias_term,
            "energy": energy,
            "mask_weights": weights.view(B, Q, *spatial),
        }
