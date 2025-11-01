from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
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
    """Open-vocabulary classification via vision-text similarity."""

    def __init__(
        self,
        text_model_id: str,
        templates_things: Sequence[str],
        templates_stuff: Sequence[str],
        temperature: float = 0.02,
        class_names_things: Optional[Sequence[str]] = None,
        class_names_stuff: Optional[Sequence[str]] = None,
        synonyms: Optional[Dict[str, Sequence[str]]] = None,
        multilingual: bool = False,
        multilingual_templates: Optional[Sequence[str]] = None,
        min_sim_threshold: Optional[float] = None,
        logit_bias: float = 0.0,
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
        self.logit_bias = nn.Parameter(torch.tensor(float(logit_bias)))
        if per_class_bias is not None:
            if not isinstance(per_class_bias, torch.Tensor):
                per_class_bias = torch.tensor(per_class_bias, dtype=torch.float32)
            self.register_buffer("per_class_bias", per_class_bias.float())
        else:
            self.register_buffer("per_class_bias", None)

        self.synonyms = {k: list(v) for k, v in (synonyms or {}).items()}
        self.multilingual = multilingual
        self.multilingual_templates = list(multilingual_templates or [])
        self.min_sim_threshold = min_sim_threshold

        self.text_model = text_model or AutoModel.from_pretrained(text_model_id, trust_remote_code=True)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_id, trust_remote_code=True)

        self.class_names_things = list(class_names_things or [])
        self.class_names_stuff = list(class_names_stuff or [])
        self.templates_things = list(templates_things)
        self.templates_stuff = list(templates_stuff)

        self.register_buffer("text_features", torch.empty(0))
        self.register_buffer("class_splits", torch.tensor([0, 0], dtype=torch.long))
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

    def _build_text_features(self) -> None:
        all_features: List[torch.Tensor] = []
        class_order: List[str] = []

        hidden_dim = None
        for attr in ("projection_dim", "projection_size", "hidden_size"):
            if hasattr(self.text_model.config, attr):
                hidden_dim = getattr(self.text_model.config, attr)
                break
        if hidden_dim is None and hasattr(self.text_model, "config") and hasattr(self.text_model.config, "text_config"):
            hidden_dim = getattr(self.text_model.config.text_config, "projection_size", None) or getattr(
                self.text_model.config.text_config, "hidden_size", None
            )
        if hidden_dim is None:
            raise AttributeError("Unable to infer hidden dimension from text model configuration.")

        def encode_group(names: Sequence[str], templates: Sequence[str]) -> torch.Tensor:
            group_features: List[torch.Tensor] = []
            for base_name in names:
                variants = [base_name] + self.synonyms.get(base_name, [])
                variant_features = []
                for variant in variants:
                    variant_features.append(self.encode_text([variant], templates)[0])
                stacked = torch.stack(variant_features)
                group_features.append(F.normalize(stacked.mean(dim=0, keepdim=True), dim=-1)[0])
                class_order.append(base_name)
            if group_features:
                return torch.stack(group_features)
            return torch.empty(0, hidden_dim, device=self.temperature.device)

        things = encode_group(self.class_names_things, self.templates_things)
        stuff = encode_group(self.class_names_stuff, self.templates_stuff)

        if things.numel() > 0:
            all_features.append(things)
        if stuff.numel() > 0:
            all_features.append(stuff)

        if all_features:
            text_features = torch.cat(all_features, dim=0).to(self.temperature.device)
            self.register_buffer("text_features", text_features, persistent=True)
            split_point = things.shape[0]
            self.register_buffer(
                "class_splits",
                torch.tensor([split_point, text_features.shape[0]], dtype=torch.long),
                persistent=True,
            )
            self.class_name_list = class_order
        else:
            self.register_buffer("text_features", torch.empty(0), persistent=True)
            self.register_buffer("class_splits", torch.tensor([0, 0], dtype=torch.long), persistent=True)
            self.class_name_list = []

    # ------------------------------------------------------------------
    # calibration helpers
    # ------------------------------------------------------------------
    def set_temperature(self, new_tau: float) -> None:
        self.temperature.data = torch.tensor(float(new_tau), device=self.temperature.device)

    def set_bias(self, bias_vec: torch.Tensor) -> None:
        self.register_buffer("per_class_bias", bias_vec.to(self.temperature.device))

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        mask_logits: torch.Tensor,
        patch_tokens: torch.Tensor,
        min_sim_threshold: Optional[float] = None,
        logit_bias: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.text_features.numel() == 0:
            raise RuntimeError("Text features have not been initialised. Provide class names at construction time.")

        probs = mask_logits.sigmoid().view(mask_logits.shape[0], mask_logits.shape[1], -1)
        weights = probs / (probs.sum(dim=-1, keepdim=True) + 1e-6)
        vision_feats = weights @ patch_tokens
        vision_feats = F.normalize(vision_feats, dim=-1)

        text_feats = F.normalize(self.text_features.to(vision_feats.dtype), dim=-1)
        sims = vision_feats @ text_feats.t()
        tau = self.temperature.clamp(min=1e-6)
        logits = sims / tau
        bias_term = self.logit_bias if logit_bias is None else logit_bias
        logits = logits + bias_term
        if self.per_class_bias is not None:
            logits = logits + self.per_class_bias
        if min_sim_threshold is None:
            min_sim_threshold = self.min_sim_threshold
        if min_sim_threshold is not None:
            logits = logits.masked_fill(sims < min_sim_threshold, float("-inf"))

        return logits, sims
