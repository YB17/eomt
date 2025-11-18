from __future__ import annotations

import logging  # ← 添加这一行
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

# ← 添加这一行
LOGGER = logging.getLogger(__name__)

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

        # 如果是分布式训练，只在 rank 0 编码
        if self.class_names_things or self.class_names_stuff:
            # 检查是否分布式
            is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            rank = torch.distributed.get_rank() if is_distributed else 0
            
            if rank == 0:
                # 只有 rank 0 进行编码
                if self.text_model is not None and torch.cuda.is_available():
                    self.text_model = self.text_model.to('cuda:0')
                    LOGGER.info("Rank 0: Encoding text features on GPU")
                self._build_text_features()
                if self.text_model is not None:
                    del self.text_model
                    del self.tokenizer
                    self.text_model = None
                    self.tokenizer = None
            
            # 所有进程等待 rank 0 完成
            if is_distributed:
                torch.distributed.barrier()
                # ✅ 广播形状信息
                if rank == 0:
                    shape_info = torch.tensor([self.text_features.shape[0], self.text_features.shape[1]], dtype=torch.long)
                else:
                    shape_info = torch.tensor([0, 0], dtype=torch.long)
                
                torch.distributed.broadcast(shape_info, src=0)
                
                # ✅ 其他 ranks 根据形状信息创建相同大小的张量
                if rank != 0:
                    num_classes = int(shape_info[0].item())
                    hidden_size = int(shape_info[1].item())
                    self.text_features = torch.zeros(num_classes, hidden_size, dtype=torch.float32)
                    LOGGER.info(f"Rank {rank}: Allocated text_features with shape ({num_classes}, {hidden_size})")
                
                # ✅ 广播 text_features 和 class_splits
                torch.distributed.broadcast(self.text_features, src=0)
                torch.distributed.broadcast(self.class_splits, src=0)
                
                LOGGER.info(f"Rank {rank}: Received text_features with shape {self.text_features.shape}")


    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """
        自定义 state_dict，排除 text_model 的参数。
        text_model 是冻结的预训练模型，每次都从相同路径加载。
        """
        # 调用父类方法获取完整的 state_dict
        state = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        # 过滤掉 text_model 的键
        keys_to_remove = [k for k in state.keys() if k.startswith(f'{prefix}text_model.')]
        for key in keys_to_remove:
            del state[key]
        
        return state
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        自定义加载逻辑，允许 text_model 的参数缺失。
        这些参数会在初始化时自动加载。
        """
        # 记录缺失的 text_model 键（这些是预期的）
        text_model_missing = [k for k in missing_keys if 'text_model' in k]
        
        # 从 missing_keys 中移除 text_model 相关的键
        missing_keys[:] = [k for k in missing_keys if 'text_model' not in k]
        
        # 调用父类方法
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        
        # 可选：记录日志
        if text_model_missing:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Skipped loading {len(text_model_missing)} text_model parameters (will be loaded from {self.text_model_id})")

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

        # ✅ 获取设备（兼容 OpenCLIP 和 Transformers）
        if hasattr(self.text_model, 'device'):
            device = self.text_model.device
        else:
            # OpenCLIP text_model 没有 .device，从参数中获取
            device = next(self.text_model.parameters()).device
        
        # ✅ 检测 tokenizer 类型并适配
        tokenizer_type = type(self.tokenizer).__name__
        
        if "HFTokenizer" in tokenizer_type or "SimpleTokenizer" in tokenizer_type:
            # OpenCLIP tokenizer: 直接调用，返回 tensor
            tokenized = self.tokenizer(prompts)
            if not isinstance(tokenized, dict):
                tokenized = {"input_ids": tokenized}
            tokenized = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in tokenized.items()}
        else:
            # Transformers tokenizer: 使用 padding 等参数
            tokenized = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
          
        with torch.no_grad():
            # ✅ 适配不同的 text_model 调用方式
            if hasattr(self.text_model, '__call__'):
                # OpenCLIP text encoder: 直接调用
                if "input_ids" in tokenized and len(tokenized) == 1:
                    # OpenCLIP 只需要 input_ids
                    outputs = self.text_model(tokenized["input_ids"])
                else:
                    outputs = self.text_model(**tokenized)
            else:
                outputs = self.text_model(**tokenized)
            
            # ✅ 适配不同的输出格式
            if isinstance(outputs, torch.Tensor):
                # OpenCLIP 直接返回 tensor
                feats = outputs
            elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
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
