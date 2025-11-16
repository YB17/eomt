from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from eomt.modules.lora import LoRAInjectionStats, inject_lora, summarize_lora

try:
    from transformers import AutoConfig, AutoModel
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
except Exception:  # pragma: no cover - optional dependency
    AutoConfig = None  # type: ignore
    AutoModel = None  # type: ignore
    SiglipVisionModel = None  # type: ignore
    SiglipVisionConfig = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import open_clip
except Exception:  # pragma: no cover - optional dependency
    open_clip = None  # type: ignore


@dataclass
class Siglip2BackboneConfig:
    model_id: Optional[str] = None
    drop_path_rate: float = 0.0
    naflex: bool = True
    img_size: Optional[int] = None
    fp16: bool = True
    out_indices: Iterable[int] = (-1,)


class _SiglipPatchEmbed(nn.Module):
    def __init__(self, embeddings: nn.Module):
        super().__init__()
        self.embeddings = embeddings
        self.proj = embeddings.patch_embedding
        patch_size = getattr(embeddings, "patch_size", 16)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        
        # 修复：处理 image_size 可能是 tuple 或 int 的情况
        image_size = embeddings.image_size
        if isinstance(image_size, (tuple, list)):
            # OpenCLIP 格式：image_size 是 tuple (H, W)
            self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        else:
            # HuggingFace Transformers 格式：image_size 是 int
            self.grid_size = (image_size // patch_size[0], image_size // patch_size[1])
        
        self.num_patches = getattr(embeddings, "num_patches", self.grid_size[0] * self.grid_size[1])
        self.last_hw: Optional[tuple[int, int]] = None
        
        # 修复：安全地获取和保存 dtype
        self._cached_dtype = self._get_proj_dtype()

    def _get_proj_dtype(self) -> torch.dtype:
        """安全地获取 projection 层的 dtype"""
        # 尝试直接访问 weight (HuggingFace Transformers)
        if hasattr(self.proj, 'weight'):
            return self.proj.weight.dtype
        # 尝试访问 proj 属性 (OpenCLIP PatchEmbed)
        elif hasattr(self.proj, 'proj') and hasattr(self.proj.proj, 'weight'):
            return self.proj.proj.weight.dtype
        # 遍历查找第一个有 weight 的参数
        for param in self.proj.parameters():
            return param.dtype
        # 默认返回 float32
        return torch.float32

    @property
    def weight_dtype(self) -> torch.dtype:
        """提供一个属性来访问 dtype"""
        return self._cached_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H % self.patch_size[0] != 0 or W % self.patch_size[1] != 0:
            raise ValueError(
                "Input height/width must be divisible by the patch size for SigLIP2 patch embedding. "
                f"Got {(H, W)} with patch size {self.patch_size}."
            )
        out = self.proj(x)
        
        # 计算 grid_size
        patch_h = H // self.patch_size[0]
        patch_w = W // self.patch_size[1]
        self.grid_size = (patch_h, patch_w)
        self.last_hw = (H, W)
        
        # 根据输出维度决定是否需要 flatten
        if out.dim() == 4:
            # Transformers Conv2d: [B, C, H, W] -> [B, N, C]
            return out.flatten(2).transpose(1, 2)
        else:
            # OpenCLIP PatchEmbed: 已经是 [B, N, C]
            return out


class _SiglipPosEmbed(nn.Module):
    def __init__(self, embeddings: nn.Module, naflex: bool = True):
        super().__init__()
        self.embeddings = embeddings
        self.naflex = naflex

    def forward(self, x: torch.Tensor, patch_embed: _SiglipPatchEmbed) -> torch.Tensor:
        height, width = patch_embed.last_hw or (
            patch_embed.grid_size[0] * patch_embed.patch_size[0],
            patch_embed.grid_size[1] * patch_embed.patch_size[1],
        )
        if self.naflex:
            pos = self.embeddings.interpolate_pos_encoding(x, height, width)
        else:
            # 修复：兼容 nn.Parameter (OpenCLIP) 和 nn.Embedding (Transformers)
            pos_emb = self.embeddings.position_embedding
            if callable(pos_emb):
                # HuggingFace: nn.Embedding 层
                pos = pos_emb(self.embeddings.position_ids)
            else:
                # OpenCLIP: nn.Parameter 张量
                pos = pos_emb
        return x + pos


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return x


class _OpenClipEmbeddings(nn.Module):
    def __init__(self, trunk: nn.Module) -> None:
        super().__init__()
        self.patch_embedding = trunk.patch_embed
        img_size = getattr(self.patch_embedding, "img_size", getattr(trunk, "img_size", 224))
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.image_size = img_size
        patch_size = getattr(self.patch_embedding, "patch_size", 16)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.grid_size = getattr(self.patch_embedding, "grid_size", (img_size[0] // patch_size[0], img_size[1] // patch_size[1]))
        self.num_patches = getattr(self.patch_embedding, "num_patches", self.grid_size[0] * self.grid_size[1])

        pos_embed = getattr(trunk, "pos_embed", None)
        if pos_embed is None:
            pos_embed = torch.zeros(1, self.num_patches, getattr(trunk, "embed_dim", 0))
        if pos_embed.shape[1] == self.num_patches + 1:
            pos_embed = pos_embed[:, 1:, :]
        self.position_embedding = nn.Parameter(pos_embed.detach().clone())
        self.register_buffer("position_ids", torch.arange(self.num_patches), persistent=False)
        self.last_hw: Optional[tuple[int, int]] = None

    def interpolate_pos_encoding(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        pos = self.position_embedding.to(dtype=x.dtype, device=x.device)
        patch_h = height // self.patch_size[0]
        patch_w = width // self.patch_size[1]
        if patch_h * patch_w == pos.shape[1]:
            return pos
        grid_h, grid_w = self.grid_size
        grid = pos.reshape(1, grid_h, grid_w, pos.shape[-1]).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(patch_h, patch_w), mode="bicubic", align_corners=False)
        return grid.permute(0, 2, 3, 1).reshape(1, -1, pos.shape[-1])


class _OpenClipAttentionAdapter(nn.Module):
    def __init__(self, attn: nn.Module) -> None:
        super().__init__()
        if not hasattr(attn, "qkv") or not hasattr(attn, "proj"):
            raise ValueError("OpenCLIP attention module is missing qkv/proj weights")

        embed_dim = attn.qkv.weight.shape[1]
        self.num_heads = getattr(attn, "num_heads", 1)
        self.head_dim = getattr(attn, "head_dim", embed_dim // self.num_heads)
        self.scale = getattr(attn, "scale", self.head_dim**-0.5)
        bias = attn.qkv.bias is not None

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=attn.proj.bias is not None)
        self.proj = self.out_proj

        q_weight, k_weight, v_weight = attn.qkv.weight.chunk(3, dim=0)
        with torch.no_grad():
            self.q_proj.weight.copy_(q_weight)
            self.k_proj.weight.copy_(k_weight)
            self.v_proj.weight.copy_(v_weight)
            if bias:
                q_bias, k_bias, v_bias = attn.qkv.bias.chunk(3)
                self.q_proj.bias.copy_(q_bias)
                self.k_proj.bias.copy_(k_bias)
                self.v_proj.bias.copy_(v_bias)
            self.out_proj.weight.copy_(attn.proj.weight)
            if attn.proj.bias is not None:
                self.out_proj.bias.copy_(attn.proj.bias)

        dropout = getattr(attn, "attn_drop", nn.Dropout(0.0))
        proj_dropout = getattr(attn, "proj_drop", nn.Dropout(0.0))
        self.attn_drop = nn.Dropout(getattr(dropout, "p", 0.0))
        self.proj_drop = nn.Dropout(getattr(proj_dropout, "p", 0.0))
        self.q_norm = getattr(attn, "q_norm", _Identity())
        self.k_norm = getattr(attn, "k_norm", _Identity())
        self.fused_attn = False
        self.dropout = getattr(attn, "dropout", getattr(dropout, "p", 0.0))

        # Original module no longer needed for forward pass
        self._original_attn = attn

        self.to(dtype=attn.qkv.weight.dtype)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, None]:
        B, N, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out, None


class _OpenClipBlock(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.layer_norm1 = block.norm1
        self.self_attn = _OpenClipAttentionAdapter(block.attn)
        self.layer_norm2 = block.norm2
        self.mlp = block.mlp


class _OpenClipEncoder(nn.Module):
    def __init__(self, blocks: List[_OpenClipBlock]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(blocks)


class _OpenClipVisionAdapter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        trunk = model
        if hasattr(trunk, "visual"):
            trunk = getattr(trunk, "visual")
        if hasattr(trunk, "trunk"):
            trunk = getattr(trunk, "trunk")

        self.embeddings = _OpenClipEmbeddings(trunk)
        blocks = [_OpenClipBlock(block) for block in getattr(trunk, "blocks", [])]
        self.encoder = _OpenClipEncoder(blocks)
        self.post_layernorm = getattr(trunk, "norm", nn.LayerNorm(blocks[0].mlp.fc2.out_features if blocks else 768))

        hidden_size = getattr(trunk, "embed_dim", self.post_layernorm.normalized_shape[0])
        num_layers = len(blocks)
        self.config = type("Cfg", (), {"hidden_size": hidden_size, "num_hidden_layers": num_layers})()

        self.use_head = False
        self.head = _Identity()
        # 修复：添加 backbone 属性避免循环引用
        self.backbone = None  # 或者可以指向 self.encoder

class _SiglipAttentionWrapper(nn.Module):
    def __init__(self, attention: nn.Module):
        super().__init__()
        self.attention = attention
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.out_proj = attention.out_proj
        self.proj = self.out_proj
        self.num_heads = attention.num_heads
        self.head_dim = attention.head_dim
        self.scale = attention.scale
        self.fused_attn = False
        dropout = getattr(attention, "dropout", 0.0)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.q_norm = _Identity()
        self.k_norm = _Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # pragma: no cover - unused direct call
        attn_output, _ = self.attention(hidden_states)
        return attn_output

    def qkv(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return torch.cat([q, k, v], dim=-1)


class _SiglipBlockAdapter(nn.Module):
    def __init__(self, block: nn.Module, drop_path: float):
        super().__init__()
        self.block = block
        self.norm1 = block.layer_norm1
        self.norm2 = block.layer_norm2
        self.attn = _SiglipAttentionWrapper(block.self_attn)
        self.mlp = block.mlp
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else _Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else _Identity()
        self.ls1 = _Identity()
        self.ls2 = _Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        attn_out = self.block.self_attn(self.norm1(x), attention_mask=attn_mask)[0]
        x = residual + self.drop_path1(attn_out)
        residual = x
        x = residual + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class SigLIP2ViTBackbone(nn.Module):
    """SigLIP2 ViT backbone emitting raw patch tokens.

    The SigLIP2 vision tower ships with a MAP pooling head. This class bypasses
    that head entirely and exposes the spatial patch tokens so they can be fed
    into the EoMT segmentation head. Positional embeddings are interpolated on
    the fly, enabling NaFlex-style variable input resolutions.
    """

    def __init__(
        self,
        model_id: Optional[str],
        out_indices: Iterable[int] = (-1,),
        drop_path: float = 0.0,
        naflex: bool = True,
        img_size: Optional[int] = None,
        fp16: bool = True,
        lora_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.config = Siglip2BackboneConfig(
            model_id=model_id,
            drop_path_rate=drop_path,
            naflex=naflex,
            img_size=img_size,
            fp16=fp16,
            out_indices=tuple(out_indices),
        )
        self.vision = self._load_vision_model()

        self.out_indices = list(out_indices)
        self.num_prefix_tokens = 0
        self.embed_dim = self.vision.config.hidden_size
        self.num_blocks = self.vision.config.num_hidden_layers

        self.patch_embed = _SiglipPatchEmbed(self.vision.embeddings)
        self.pos_embed = _SiglipPosEmbed(self.vision.embeddings, naflex=naflex)
        self.patch_drop = _Identity()
        self.norm_pre = _Identity()
        self.norm = self.vision.post_layernorm

        self.lora_stats: Optional[LoRAInjectionStats] = None
        if lora_cfg and lora_cfg.get("ENABLED", lora_cfg.get("enabled", False)):
            self.lora_stats = inject_lora(
                self.vision,
                last_n_layers=int(lora_cfg.get("LAST_N_LAYERS", lora_cfg.get("layers_last_n", 12))),
                r_attn=int(lora_cfg.get("RANK_ATTN", lora_cfg.get("rank_attn", lora_cfg.get("RANK", 16)))),
                r_ffn=int(lora_cfg.get("RANK_FFN", lora_cfg.get("rank_ffn", lora_cfg.get("RANK", 32)))),
                alpha_scale=float(lora_cfg.get("ALPHA_SCALE", lora_cfg.get("alpha_scale", 2.0))),
                dropout=float(lora_cfg.get("DROPOUT", lora_cfg.get("dropout", 0.0))),
                bias=str(lora_cfg.get("BIAS", lora_cfg.get("bias", "none"))),
                include_proj=bool(lora_cfg.get("INCLUDE_PROJ", lora_cfg.get("include_proj", False))),
            )

        drop_path_rates = torch.linspace(0, drop_path, self.vision.config.num_hidden_layers).tolist()
        self.blocks = nn.ModuleList(
            _SiglipBlockAdapter(layer, drop)
            for layer, drop in zip(self.vision.encoder.layers, drop_path_rates)
        )

        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean)
        self.register_buffer("pixel_std", std)
        # 修复：添加 backbone 属性避免循环引用
        # 当作为 teacher 使用时，common.py 会检查这个属性
        self.backbone = self.vision  # 指向 vision model，而不是 self
    # ---------------------------------------------------------------------
    # model loading utilities
    # ---------------------------------------------------------------------
    def _load_vision_model(self) -> nn.Module:
        if SiglipVisionModel is None:
            return self._build_dummy_vision_model()

        if self.config.model_id:
            oc_path = self._detect_open_clip_path(Path(self.config.model_id))
            if oc_path is not None:
                return self._load_open_clip_vision_model(oc_path)
            if AutoModel is None:
                raise ImportError("transformers AutoModel is unavailable")
            try:
                model = AutoModel.from_pretrained(
                    self.config.model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                )
            except OSError:
                cfg = AutoConfig.from_pretrained(self.config.model_id, trust_remote_code=True)
                model = AutoModel.from_config(cfg, trust_remote_code=True)
        else:
            cfg_kwargs = {}
            if self.config.img_size is not None:
                cfg_kwargs["image_size"] = self.config.img_size
            cfg = SiglipVisionConfig(**cfg_kwargs)
            model = SiglipVisionModel(cfg)

        vision_model = self._extract_vision_transformer(model)
        if hasattr(vision_model, "use_head"):
            vision_model.use_head = False
            vision_model.head = _Identity()
        return vision_model

    @staticmethod
    def _detect_open_clip_path(path: Path) -> Optional[Path]:
        if path.is_file():
            if "open_clip" in path.name:
                return path.parent
            return None
        if not path.exists():
            return None
        for candidate in (
            path / "open_clip_model.safetensors",
            path / "open_clip_pytorch_model.bin",
            path / "model.open_clip.pt",
        ):
            if candidate.exists():
                return path
        return None

    def _load_open_clip_vision_model(self, path: Path) -> nn.Module:
        if open_clip is None:
            raise ImportError("open_clip_torch is required to load OpenCLIP checkpoints")

        model_root = path
        precision = "fp16" if self.config.fp16 else "fp32"
        try:
            clip_model = open_clip.create_model_from_pretrained(
                f"local-dir:{model_root}",
                device="cpu",
                precision=precision,
                return_transform=False,
                cache_dir=None,
            )
        except Exception:
            # Fallback: attempt to read architecture metadata
            config_path = model_root / "open_clip_config.json"
            model_name: Optional[str] = None
            if config_path.exists():
                try:
                    with config_path.open("r", encoding="utf-8") as handle:
                        cfg = json.load(handle)
                    model_name = cfg.get("model_name") or cfg.get("model")
                    if model_name is None and isinstance(cfg.get("architectures"), list):
                        model_name = cfg["architectures"][0]
                except Exception:
                    model_name = None
            if model_name is None:
                model_name = "local-dir:" + str(model_root)
            clip_model = open_clip.create_model_from_pretrained(
                model_name,
                pretrained=str(model_root / "open_clip_model.safetensors")
                if (model_root / "open_clip_model.safetensors").exists()
                else None,
                device="cpu",
                precision=precision,
                return_transform=False,
                cache_dir=None,
            )

        dtype = torch.float16 if self.config.fp16 else torch.float32
        clip_model = clip_model.to(dtype=dtype)
        vision_model = _OpenClipVisionAdapter(clip_model).to(dtype=dtype)
        return vision_model

    def _build_dummy_vision_model(self) -> nn.Module:
        embed_dim = 64
        num_layers = 4
        num_heads = 4
        patch_size = 16

        class DummyEmbeddings(nn.Module):
            def __init__(self, embed_dim: int, patch_size: int) -> None:
                super().__init__()
                self.patch_embedding = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
                self.image_size = 224
                self.patch_size = patch_size
                self.num_patches = (self.image_size // patch_size) ** 2
                self.position_ids = nn.Parameter(torch.arange(self.num_patches), requires_grad=False)
                self.position_embedding = nn.Embedding(self.num_patches, embed_dim)

            def interpolate_pos_encoding(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
                num_positions = self.position_embedding.weight.shape[0]
                dim = x.shape[-1]
                pos = self.position_embedding.weight.view(1, int(num_positions**0.5), int(num_positions**0.5), dim)
                pos = pos.permute(0, 3, 1, 2)
                pos = F.interpolate(pos, size=(height // self.patch_size, width // self.patch_size), mode="bicubic")
                return pos.permute(0, 2, 3, 1).reshape(1, -1, dim)

        class DummyAttention(nn.Module):
            def __init__(self, embed_dim: int, num_heads: int) -> None:
                super().__init__()
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.head_dim = embed_dim // num_heads
                self.scale = self.head_dim**-0.5
                self.q_proj = nn.Linear(embed_dim, embed_dim)
                self.k_proj = nn.Linear(embed_dim, embed_dim)
                self.v_proj = nn.Linear(embed_dim, embed_dim)
                self.out_proj = nn.Linear(embed_dim, embed_dim)
                self.dropout = 0.0
                self.attn_drop = nn.Dropout(0.0)
                self.proj_drop = nn.Dropout(0.0)
                self.fused_attn = False
                self.q_norm = _Identity()
                self.k_norm = _Identity()

            def forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **_: Any,
            ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
                B, N, C = hidden_states.shape
                q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                k = k.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                out = torch.matmul(attn, v)
                out = out.transpose(1, 2).reshape(B, N, C)
                return self.out_proj(out), None

        class DummyLayer(nn.Module):
            def __init__(self, embed_dim: int, num_heads: int) -> None:
                super().__init__()
                self.layer_norm1 = nn.LayerNorm(embed_dim)
                self.self_attn = DummyAttention(embed_dim, num_heads)
                self.layer_norm2 = nn.LayerNorm(embed_dim)
                self.mlp = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim),
                )

        class DummyVision(nn.Module):
            def __init__(self, embed_dim: int, num_heads: int, num_layers: int, patch_size: int) -> None:
                super().__init__()
                self.config = type("Cfg", (), {"hidden_size": embed_dim, "num_hidden_layers": num_layers})()
                self.embeddings = DummyEmbeddings(embed_dim, patch_size)
                self.encoder = type("Encoder", (), {})()
                self.encoder.layers = nn.ModuleList([DummyLayer(embed_dim, num_heads) for _ in range(num_layers)])
                self.post_layernorm = nn.LayerNorm(embed_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                patches = self.embeddings.patch_embedding(x)
                patches = patches.flatten(2).transpose(1, 2)
                pos = self.embeddings.interpolate_pos_encoding(patches, x.shape[-2], x.shape[-1])
                tokens = patches + pos
                for layer in self.encoder.layers:
                    attn_in = layer.layer_norm1(tokens)
                    attn_out, _ = layer.self_attn(attn_in)
                    tokens = tokens + attn_out
                    tokens = tokens + layer.mlp(layer.layer_norm2(tokens))
                return self.post_layernorm(tokens)

        dummy = DummyVision(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, patch_size=patch_size)
        dummy.use_head = False
        dummy.head = _Identity()
        return dummy

    @staticmethod
    def _extract_vision_transformer(model: nn.Module) -> nn.Module:
        if hasattr(model, "vision_model"):
            vision_model = getattr(model, "vision_model")
            if hasattr(vision_model, "vision_model"):
                return vision_model.vision_model
            return vision_model
        if hasattr(model, "vision_tower"):
            tower = getattr(model, "vision_tower")
            if hasattr(tower, "vision_model"):
                return tower.vision_model
            return tower
        return model

    # ------------------------------------------------------------------
    # forward utilities
    # ------------------------------------------------------------------
    def _apply_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos_embed(x, self.patch_embed)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - timm compatibility
        return self._apply_positional_encoding(x)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        dtype = self.patch_embed.weight_dtype
        tokens = self.patch_embed(x.to(dtype=dtype))
        tokens = self._apply_positional_encoding(tokens)
        tokens = self.patch_drop(tokens)
        tokens = self.norm_pre(tokens)

        outputs: List[torch.Tensor] = []
        for idx, block in enumerate(self.blocks):
            tokens = block(tokens)
            if idx in self.out_indices:
                outputs.append(self.norm(tokens))
        if -1 in self.out_indices or (len(outputs) == 0):
            outputs.append(self.norm(tokens))
        return outputs

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # pragma: no cover - compatibility helper
        return self.forward_features(x)

    def get_lora_summary(self) -> Optional[LoRAInjectionStats]:
        """Return cached LoRA statistics or recompute them on demand."""

        if self.lora_stats is not None:
            return self.lora_stats
        return summarize_lora(self.vision)
