from __future__ import annotations

import argparse
import json
import math
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import torch
import yaml
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.loggers.logger import Logger as LightningLoggerBase

from eomt.data.coco_ov_vocab import COCO_STUFF_53, COCO_THINGS_80, SYNONYMS, build_templates
from eomt.datasets.coco_panoptic_directory import COCOPanopticDirectory
from eomt.models.backbone_registry import build_backbone
from eomt.models.eomt import EoMT
from eomt.models.open_vocab_head import OpenVocabHead
from training.mask_classification_panoptic import MaskClassificationPanoptic

LOGGER = logging.getLogger(__name__)


class EncoderWrapper(torch.nn.Module):
    """Minimal wrapper exposing ``backbone``/pixel stats for EoMT."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    @property
    def pixel_mean(self) -> torch.Tensor:
        return self.backbone.pixel_mean

    @property
    def pixel_std(self) -> torch.Tensor:
        return self.backbone.pixel_std


# ---------------------------------------------------------------------------
# configuration helpers
# ---------------------------------------------------------------------------

def _decode_value(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        if value.startswith("0x"):
            return int(value, 16)
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",")]
        return [_decode_value(p) for p in parts]
    if value.startswith("{") and value.endswith("}"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _set_by_path(cfg: MutableMapping[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    node: MutableMapping[str, Any] = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], MutableMapping):
            node[key] = {}
        node = node[key]  # type: ignore[assignment]
    node[keys[-1]] = value


def load_config(path: str, overrides: Sequence[str]) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = deepcopy(cfg) if isinstance(cfg, dict) else {}

    if len(overrides) % 2 != 0:
        raise ValueError("Overrides must be KEY VALUE pairs")

    for key, raw in zip(overrides[0::2], overrides[1::2]):
        value = _decode_value(raw)
        _set_by_path(cfg, key, value)

    return cfg


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _resolve_resolution(cfg: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    res_cfg = cfg.get("RESOLUTION", {})
    preset = str(res_cfg.get("PRESET", "Safe"))
    preset_key = preset.upper()
    preset_cfg = res_cfg.get(preset_key, {})
    return preset, preset_cfg


def build_datamodule(cfg: Dict[str, Any]) -> COCOPanopticDirectory:
    data_cfg = cfg.get("DATA", {})
    preset, preset_cfg = _resolve_resolution(cfg)
    batch_size = int(data_cfg.get("BATCH_SIZE", 2))
    num_workers = int(data_cfg.get("NUM_WORKERS", 4))
    crop_sizes = preset_cfg.get("CROP_SIZES") or [data_cfg.get("TRAIN_RES_MAX", 640)]
    img_max = int(max(crop_sizes))
    short_edge = preset_cfg.get("SHORT_EDGE")
    if short_edge is not None:
        short_edge = (int(short_edge[0]), int(short_edge[1]))
    train_transform_cfg = {
        "short_edge": short_edge,
        "long_edge_max": int(preset_cfg.get("LONG_EDGE_MAX", img_max)),
        "crop_sizes": [int(v) for v in crop_sizes],
        "color_jitter": bool(preset_cfg.get("COLOR_JITTER", True)),
        "flip_prob": float(preset_cfg.get("FLIP_PROB", 0.5)),
        "keep_full_instances": True,
    }
    datamodule = COCOPanopticDirectory(
        path=data_cfg.get("ROOT", ""),
        stuff_classes=list(range(len(COCO_THINGS_80), len(COCO_THINGS_80) + len(COCO_STUFF_53))),
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=(img_max, img_max),
        num_classes=len(COCO_THINGS_80) + len(COCO_STUFF_53),
        check_empty_targets=data_cfg.get("CHECK_EMPTY_TARGETS", True),
        pin_memory=bool(data_cfg.get("PIN_MEMORY", True)),
        persistent_workers=bool(data_cfg.get("PERSISTENT_WORKERS", True)),
        train_transform_cfg=train_transform_cfg,
        resolution_preset=preset,
    )
    datamodule.setup("fit")
    return datamodule


def build_open_vocab_head(cfg: Dict[str, Any]) -> Optional[OpenVocabHead]:
    ov_cfg = cfg.get("MODEL", {}).get("OPEN_VOCAB", {})  # 修改这里
    if not ov_cfg.get("ENABLED", False):
        return None

    text_model_id = ov_cfg.get("TEXT_MODEL_ID")
    if text_model_id in (None, ""):
        raise ValueError("OPEN_VOCAB.TEXT_MODEL_ID must be specified when OPEN_VOCAB.ENABLED is true")

    multilingual = bool(ov_cfg.get("MULTILINGUAL", False))
    templates = build_templates(multilingual)
    synonyms = dict(SYNONYMS)
    custom_synonyms = ov_cfg.get("SYNONYMS_JSON")
    if custom_synonyms:
        with Path(custom_synonyms).open("r", encoding="utf-8") as handle:
            synonyms.update(json.load(handle))

    total_classes = len(COCO_THINGS_80) + len(COCO_STUFF_53)
    per_class_bias_cfg = ov_cfg.get("PER_CLASS_BIAS", None)
    per_class_bias: Optional[torch.Tensor | float]
    if isinstance(per_class_bias_cfg, str) and per_class_bias_cfg.lower() == "auto":
        per_class_bias = torch.zeros(total_classes)
    else:
        per_class_bias = per_class_bias_cfg

        # ✅ 新增：尝试从 OpenCLIP 加载 text model
    text_model = None
    tokenizer = None
    model_path = Path(text_model_id)

    # 检查是否是 OpenCLIP 格式
    if model_path.exists() and any((model_path / f).exists() for f in [
        "open_clip_model.safetensors", 
        "open_clip_pytorch_model.bin", 
        "model.open_clip.pt"
    ]):
        try:
            import open_clip
            LOGGER.info(f"Loading OpenCLIP model from {text_model_id} for text encoder")
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                f"local-dir:{model_path}",
                device="cpu",
                precision="fp32",  # 使用 fp32 避免精度问题
            )
            # 提取 text model
            if hasattr(clip_model, 'text'):
                text_model = clip_model.text
            elif hasattr(clip_model, 'transformer'):
                text_model = clip_model.transformer
            else:
                LOGGER.warning("OpenCLIP model has no text encoder, falling back to transformers")
                text_model = None
            
            # 创建tokenizer（OpenCLIP 使用自己的 tokenizer）
            if text_model is not None:
                tokenizer = open_clip.get_tokenizer(f"local-dir:{model_path}")
                LOGGER.info("✅ Successfully loaded OpenCLIP text encoder")
        except Exception as e:
            LOGGER.warning(f"Failed to load OpenCLIP text model: {e}, falling back to transformers")
            text_model = None
            tokenizer = None

    head = OpenVocabHead(
        text_model_id=text_model_id,
        templates_things=templates["things"],
        templates_stuff=templates["stuff"],
        temperature=float(ov_cfg.get("TEMP", 0.02)),
        gamma=float(ov_cfg.get("POOL_TEMPERATURE", 1.0)),
        calibration_bias=float(ov_cfg.get("CALIBRATION_BIAS", 0.0)),
        energy_reject_thr=float(ov_cfg.get("ENERGY_REJECT_THR", float("-inf"))),
        energy_type=str(ov_cfg.get("ENERGY_TYPE", "max")),
        class_names_things=COCO_THINGS_80,
        class_names_stuff=COCO_STUFF_53,
        synonyms=synonyms,
        multilingual=multilingual,
        multilingual_templates=templates.get("multilingual"),
        per_class_bias=per_class_bias,
        text_model=text_model,  # ✅ 传入已加载的 text_model
        tokenizer=tokenizer,    # ✅ 传入 tokenizer
    )
    if hasattr(head, "text_model") and head.text_model is not None:
        head.text_model.eval()
        for param in head.text_model.parameters():
            param.requires_grad_(False)
    return head


def build_backbone_from_cfg(cfg: Dict[str, Any]) -> Tuple[torch.nn.Module, bool]:
    backbone_cfg = deepcopy(cfg.get("MODEL", {}).get("BACKBONE", {}))
    name = backbone_cfg.pop("NAME", None)
    if name is None:
        raise ValueError("MODEL.BACKBONE.NAME must be provided")
    freeze = bool(backbone_cfg.pop("FREEZE", False))
    backbone = build_backbone(name, **backbone_cfg)
    if not hasattr(backbone, "backbone"):
        backbone.backbone = backbone  # type: ignore[attr-defined]
    if freeze:
        for param in backbone.parameters():
            param.requires_grad_(False)
    return backbone, freeze


def compute_mask_schedule(
    cfg: Dict[str, Any],
    steps_per_epoch: int,
) -> Tuple[List[int], List[int]]:
    anneal_cfg = cfg.get("ANNEAL", {})
    if not anneal_cfg.get("ENABLED", False):
        return [], []
    epochs = int(cfg.get("SOLVER", {}).get("EPOCHS", 1))
    total_steps = max(1, steps_per_epoch * epochs)
    num_blocks = max(1, int(anneal_cfg.get("L2_BLOCKS", 1)))
    start_fracs = torch.linspace(0.0, 0.6, steps=num_blocks)
    end_fracs = torch.linspace(0.4, 1.0, steps=num_blocks)
    start_steps = [int(total_steps * frac) for frac in start_fracs]
    end_steps = [max(start + 1, int(total_steps * frac)) for start, frac in zip(start_steps, end_fracs)]
    return start_steps, end_steps


def build_network(cfg: Dict[str, Any], backbone: torch.nn.Module, ov_head: Optional[OpenVocabHead]) -> EoMT:
    head_cfg = cfg.get("MODEL", {}).get("HEAD", {})
    anneal_cfg = cfg.get("ANNEAL", {})
    num_classes = len(COCO_THINGS_80) + len(COCO_STUFF_53)
    num_queries = int(head_cfg.get("NUM_QUERIES", 150))
    num_blocks = int(head_cfg.get("NUM_BLOCKS", anneal_cfg.get("L2_BLOCKS", 4)))
    query_init = str(head_cfg.get("QUERY_INIT", "learnable"))
    masked_attn_enabled = bool(anneal_cfg.get("ENABLED", True))

    encoder = EncoderWrapper(backbone)
    network = EoMT(
        encoder=encoder,
        num_classes=num_classes,
        num_q=num_queries,
        num_blocks=num_blocks,
        masked_attn_enabled=masked_attn_enabled,
        open_vocab_head=ov_head,
        fuse_closed_head=bool(cfg.get("MODEL", {}).get("OPEN_VOCAB", {}).get("FUSE_CLOSED_HEAD", False)),
        query_init=query_init,
    )
    return network


def build_teacher(cfg: Dict[str, Any]) -> Optional[torch.nn.Module]:
    loss_cfg = cfg.get("LOSS", {})
    distill_cfg = loss_cfg.get("DISTILL", {})
    need_teacher = distill_cfg.get("FEAT_ALIGN", 0.0) > 0.0 or distill_cfg.get("ITC_WEIGHT", 0.0) > 0.0
    if not need_teacher:
        return None

    backbone_cfg = deepcopy(cfg.get("MODEL", {}).get("BACKBONE", {}))
    name = backbone_cfg.pop("NAME", None)
    if name is None:
        raise ValueError("MODEL.BACKBONE.NAME must be provided for teacher construction")
    backbone_cfg.setdefault("LORA", {})
    backbone_cfg["LORA"]["ENABLED"] = False
    teacher = build_backbone(name, **backbone_cfg)
    if not hasattr(teacher, "backbone"):
        teacher.backbone = teacher  # type: ignore[attr-defined]
    teacher.eval()
    
    # 修复：强制 Teacher 使用 FP32，避免混合精度梯度问题
    teacher = teacher.float()  # ✅ 关键修复！
    for param in teacher.parameters():
        param.requires_grad_(False)

    # 额外保险：确保所有 buffer 也是 FP32
    for name, buffer in teacher.named_buffers():
        if buffer.dtype == torch.float16:
            buffer.data = buffer.float()

    return teacher


def build_module(
    cfg: Dict[str, Any],
    network: EoMT,
    datamodule: COCOPanopticDirectory,
    teacher_backbone: Optional[torch.nn.Module],
    start_steps: Sequence[int],
    end_steps: Sequence[int],
    steps_per_epoch: int,
    backbone_frozen: bool,
) -> MaskClassificationPanoptic:
    solver_cfg = cfg.get("SOLVER", {})
    loss_cfg = cfg.get("LOSS", {})
    anneal_cfg = cfg.get("ANNEAL", {})
    data_cfg = cfg.get("DATA", {})
    seg_cfg = loss_cfg.get("SEG", {})
    ov_cfg = loss_cfg.get("OV", {})
    aux_cfg = loss_cfg.get("AUX", {})
    match_cfg = loss_cfg.get("MATCH", {})
    res_cfg = cfg.get("RESOLUTION", {})
    perfplus_cfg = res_cfg.get("PERFPLUS", {})
    lora_cfg = cfg.get("MODEL", {}).get("BACKBONE", {}).get("LORA", {})
    stage_cfg = {
        "stage": cfg.get("STAGE", "A"),
        "global_batch_size": int(cfg.get("GLOBAL_BATCH_SIZE", 0)),
        "backbone_freeze": cfg.get("BACKBONE_FREEZE", backbone_frozen),
        "use_lora": cfg.get("USE_LORA", cfg.get("MODEL", {}).get("BACKBONE", {}).get("LORA", {}).get("ENABLED", False)),
        "use_lora_attn": bool(lora_cfg.get("ENABLED", False)),
        "use_lora_ffn": bool(lora_cfg.get("ENABLED", False)),
        "resolution_preset": getattr(datamodule, "resolution_preset", "Safe"),
        "eval_interval": int(cfg.get("EVAL", {}).get("INTERVAL_EPOCHS", 1)),
        "seed": cfg.get("SEED", 0),
        "epochs": int(cfg.get("EPOCHS", solver_cfg.get("EPOCHS", 1))),
        "output_dir": cfg.get("OUTPUT_DIR", "outputs"),
        "best_ckpt_name": cfg.get("STAGE_BEST_CKPT", f"{str(cfg.get('STAGE', 'A')).lower()}_best.ckpt"),
        "perfplus_memory_limit_gb": float(perfplus_cfg.get("MEMORY_LIMIT_GB", 23.5)),
        "perfplus_monitor_iters": int(perfplus_cfg.get("MONITOR_ITERS", 50)),
        "fixed_res": int(data_cfg.get("TRAIN_RES_MAX", 0)),
    }

    img_max = int(data_cfg.get("TRAIN_RES_MAX", 1024))
    total_layers = getattr(network.encoder.backbone, "num_blocks", network.num_blocks)    
    start_block = max(0, total_layers - network.num_blocks)
    stage_cfg["query_blocks"] = f"{start_block}..{total_layers - 1}"
    module = MaskClassificationPanoptic(
        network=network,
        img_size=(img_max, img_max),
        num_classes=len(COCO_THINGS_80) + len(COCO_STUFF_53),
        stuff_classes=list(range(len(COCO_THINGS_80), len(COCO_THINGS_80) + len(COCO_STUFF_53))),
        attn_mask_annealing_enabled=bool(anneal_cfg.get("ENABLED", True)),
        attn_mask_annealing_start_steps=list(start_steps) if start_steps else None,
        attn_mask_annealing_end_steps=list(end_steps) if end_steps else None,
        lr=float(solver_cfg.get("LR_HEAD", 5e-4)),
        lr_lora=float(solver_cfg.get("LR_LORA", solver_cfg.get("LR_HEAD", 5e-4))),
        llrd=float(solver_cfg.get("LLRD", 0.8)),
        weight_decay=float(solver_cfg.get("WD", 0.05)),
        poly_power=float(anneal_cfg.get("FACTOR", 0.9)),
        warmup_steps=list(solver_cfg.get("WARMUP_STEPS", [500, 1000])),
        open_vocab_kl_weight=float(ov_cfg.get("KL_TEXT_WEIGHT", 0.0)),
        open_vocab_kl_temperature=float(ov_cfg.get("KL_TEMPERATURE", 2.0)),
        mask_coefficient=float(seg_cfg.get("LAMBDA_BCE", 2.0)),
        dice_coefficient=float(seg_cfg.get("LAMBDA_DICE", 1.0)),
        class_coefficient=float(ov_cfg.get("CE_WEIGHT", 1.0)),
        open_vocab_kl_seen_only=bool(cfg.get("OPEN_VOCAB", {}).get("KL_SEEN_ONLY", False)),
        open_vocab_kl_top_p=ov_cfg.get("KL_TOP_P"),
        open_vocab_kl_top_k=ov_cfg.get("KL_TOP_K"),
        matcher_weights=match_cfg,
        aux_weight=float(aux_cfg.get("WEIGHT", 0.3)),
        stage_config=stage_cfg,
        steps_per_epoch=steps_per_epoch,
        stage_head_lr=float(solver_cfg.get("LR_HEAD", 5e-4)),
        stage_other_lr=float(solver_cfg.get("LR_OTHER", solver_cfg.get("LR_HEAD", 5e-4))),
        stage_wd_head=float(solver_cfg.get("WD_HEAD", 0.0)),
        stage_wd_other=float(solver_cfg.get("WD_OTHER", solver_cfg.get("WD", 0.05))),
        stage_lora_lr=float(solver_cfg.get("LR_LORA", solver_cfg.get("LR_HEAD", 5e-4))),
        stage_wd_lora=float(solver_cfg.get("WD_LORA", solver_cfg.get("WD_OTHER", solver_cfg.get("WD", 0.05)))),
        stage_betas=tuple(solver_cfg.get("BETAS", (0.9, 0.98))),
        stage_warmup_ratio=float(solver_cfg.get("WARMUP_RATIO", 0.05)),
        distill_feat_weight=float(loss_cfg.get("DISTILL", {}).get("FEAT_ALIGN", 0.0)),
        distill_itc_weight=float(loss_cfg.get("DISTILL", {}).get("ITC_WEIGHT", 0.0)),
        distill_temperature=float(loss_cfg.get("DISTILL", {}).get("TEMPERATURE", 0.07)),
        teacher_backbone=teacher_backbone,
    )
    return module


def _build_wandb_logger(
    cfg: Dict[str, Any],
    output_dir: Path,
) -> Optional[WandbLogger]:
    logging_cfg = cfg.get("LOGGING", {})
    wandb_cfg = logging_cfg.get("WANDB", {})
    if not bool(wandb_cfg.get("ENABLED", False)):
        return None

    data_cfg = cfg.get("DATA", {})
    tags = list(wandb_cfg.get("TAGS", []) or [])
    stage_tag = output_dir.name
    if stage_tag and stage_tag not in tags:
        tags.append(stage_tag)
    dataset_tag = data_cfg.get("DATASET")
    if dataset_tag and dataset_tag not in tags:
        tags.append(str(dataset_tag))

    project = wandb_cfg.get("PROJECT") or stage_tag or "eomt"
    run_name = wandb_cfg.get("NAME") or stage_tag
    wandb_logger = WandbLogger(
        project=project,
        entity=wandb_cfg.get("ENTITY"),
        name=run_name,
        save_dir=str(output_dir),
        group=wandb_cfg.get("GROUP"),
        job_type=wandb_cfg.get("JOB_TYPE"),
        tags=tags,
        log_model=bool(wandb_cfg.get("LOG_MODEL", False)),
        mode=wandb_cfg.get("MODE", "online"),
        resume=wandb_cfg.get("RESUME", "never"),
        notes=wandb_cfg.get("NOTES"),
        id=wandb_cfg.get("ID"),
    )
    return wandb_logger


def _maybe_find_wandb_logger(
    logger: Optional[LightningLoggerBase | Iterable[LightningLoggerBase]],
) -> Optional[WandbLogger]:
    if logger is None:
        return None
    if isinstance(logger, WandbLogger):
        return logger
    if isinstance(logger, Iterable):
        for entry in logger:
            found = _maybe_find_wandb_logger(entry)
            if found is not None:
                return found
    return None

def _rank0_only(func):
    """装饰器：只在主进程执行函数"""
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = int(os.environ.get("RANK", 0))
        
        # 检查是否是主进程
        if local_rank == 0 and global_rank == 0:
            return func(*args, **kwargs)
        return None
    
    return wrapper

@_rank0_only
def _log_wandb_metadata(
    cfg: Dict[str, Any],
    module: MaskClassificationPanoptic,
    datamodule: COCOPanopticDirectory,
    steps_per_epoch: int,
    output_dir: Path,
    logger: Optional[LightningLoggerBase | Iterable[LightningLoggerBase]],
) -> None:
    wandb_logger = _maybe_find_wandb_logger(logger)
    if wandb_logger is None:
        return

    wandb_cfg = cfg.get("LOGGING", {}).get("WANDB", {})

    # 安全获取 experiment
    try:
        experiment = wandb_logger.experiment
        if not hasattr(experiment, "config"):
            LOGGER.warning("⚠️ WandB experiment not properly initialized")
            return
    except Exception as e:
        LOGGER.warning(f"⚠️ Failed to get WandB experiment: {e}")
        return

    if wandb_cfg.get("LOG_CONFIG", True):
        experiment.config.update(cfg, allow_val_change=True)

    train_dataset = getattr(datamodule, "train_dataset", None)
    val_dataset = getattr(datamodule, "val_dataset", None)
    data_cfg = cfg.get("DATA", {})
    dataset_meta = {
        "root": data_cfg.get("ROOT"),
        "train_split": data_cfg.get("TRAIN_SPLIT"),
        "val_split": data_cfg.get("VAL_SPLIT"),
        "train_samples": len(train_dataset) if train_dataset is not None else None,
        "val_samples": len(val_dataset) if val_dataset is not None else None,
        "img_max": data_cfg.get("TRAIN_RES_MAX"),
        "batch_size": getattr(datamodule, "batch_size", data_cfg.get("BATCH_SIZE")),
        "num_workers": getattr(datamodule, "num_workers", data_cfg.get("NUM_WORKERS")),
    }

    solver_cfg = cfg.get("SOLVER", {})
    max_epochs = int(solver_cfg.get("EPOCHS", 1))
    warmup_steps = list(module.warmup_steps) if module.warmup_steps is not None else []
    training_meta = {
        "steps_per_epoch": steps_per_epoch,
        "total_epochs": max_epochs,
        "total_optimization_steps": steps_per_epoch * max_epochs,
        "warmup_steps": warmup_steps,
        "poly_power": module.poly_power,
        "lr_head": solver_cfg.get("LR_HEAD"),
        "lr_lora": solver_cfg.get("LR_LORA", solver_cfg.get("LR_HEAD")),
        "llrd": module.llrd,
        "weight_decay": module.weight_decay,
        "output_dir": str(output_dir),
        "resume": bool(wandb_cfg.get("RESUME", False)),
    }

    experiment.config.update(
        {
            "dataset_meta": dataset_meta,
            "training_meta": training_meta,
            "stage": output_dir.name,
        },
        allow_val_change=True,
    )

    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    summary_payload = {
        "parameters/total": total_params,
        "parameters/trainable": trainable_params,
        "parameters/frozen": total_params - trainable_params,
        "classes/num_classes": module.num_classes,
        "classes/stuff": len(getattr(module, "stuff_classes", [])),
    }
    experiment.summary.update(summary_payload)

    watch_cfg = wandb_cfg.get("WATCH", {})
    if watch_cfg.get("ENABLED", True):
        wandb_logger.watch(
            module,
            log=watch_cfg.get("LOG", "gradients"),
            log_freq=int(watch_cfg.get("LOG_FREQ", 250)),
        )


def trainer_kwargs(
    cfg: Dict[str, Any], output_dir: Path, resume: bool = False
) -> Tuple[Dict[str, Any], Optional[str]]:
    solver_cfg = cfg.get("SOLVER", {})
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(save_last=True, save_top_k=-1, every_n_epochs=5),
    ]

    # ========== ⭐ 新增：Stage A 专用的 best checkpoint callback ========== 
    stage_value = cfg.get("STAGE", "")
    # ⭐ 兼容两种配置格式：
    # 1. STAGE: "A"（字符串）
    # 2. STAGE: { STAGE: "A", ... }（字典）
    if isinstance(stage_value, dict):
        stage_value = stage_value.get("STAGE", "")

    stage_upper = str(stage_value).upper()
    if stage_upper in ["A", "B"]:  # ⭐ 修改：包含 Stage B
        # 创建专门监控 PQ 的 checkpoint callback
        best_pq_callback = ModelCheckpoint(
            dirpath=str(output_dir),           # 保存目录
            filename="stage_a_best",           # 文件名（不需要 .ckpt 后缀）
            monitor="metrics/val_pq_all",      # 监控的指标
            mode="max",                         # 最大化 PQ
            save_top_k=1,                       # 只保存最佳的1个
            save_last=False,                    # 不额外保存 last
            save_on_train_epoch_end=False,     # 只在 validation 后保存
            auto_insert_metric_name=False,      # 不在文件名中插入指标名
        )
        callbacks.append(best_pq_callback)
        LOGGER.info("Stage A: Added best checkpoint callback monitoring 'metrics/val_pq_all'")
    # ====================================================================

    loggers: List[LightningLoggerBase] = []
    csv_logger = CSVLogger(save_dir=str(output_dir), name="logs")
    loggers.append(csv_logger)
    wandb_logger = _build_wandb_logger(cfg, output_dir)
    if wandb_logger is not None:
        loggers.append(wandb_logger)
    kwargs: Dict[str, Any] = {
        "max_epochs": int(solver_cfg.get("EPOCHS", 12)),
        "gradient_clip_val": float(solver_cfg.get("CLIP_GRAD", 1.0)),
        "default_root_dir": str(output_dir),
        "callbacks": callbacks,
        "logger": loggers if len(loggers) > 1 else loggers[0],
        "precision": cfg.get("TRAINER", {}).get("PRECISION", "16-mixed"),
        
        # ==================== 多卡训练参数 ====================
        "accelerator": cfg.get("TRAINER", {}).get("ACCELERATOR", "gpu"),
        "devices": cfg.get("TRAINER", {}).get("DEVICES", 1),
        "strategy": cfg.get("TRAINER", {}).get("STRATEGY", "auto"),
        "num_nodes": cfg.get("TRAINER", {}).get("NUM_NODES", 1),
        "sync_batchnorm": cfg.get("TRAINER", {}).get("SYNC_BATCHNORM", False),
        # ====================================================

        # ⭐⭐⭐ 关键修复：禁用 sanity check ⭐⭐⭐
        "num_sanity_val_steps": 0,
        # 其他验证参数
        "limit_val_batches": cfg.get("TRAINER", {}).get("LIMIT_VAL_BATCHES", None),
        "val_check_interval": cfg.get("TRAINER", {}).get("VAL_CHECK_INTERVAL", 1.0),
        # ====================================================

    }
    resume_path: Optional[str] = None
    if resume:
        last_ckpt = output_dir / "last.ckpt"
        if last_ckpt.exists():
            resume_path = str(last_ckpt)
            LOGGER.info("Resuming from %s", last_ckpt)
    return kwargs, resume_path

def compute_steps_per_epoch(
    cfg: Dict[str, Any],
    datamodule: LightningDataModule,
) -> int:
    """
    计算每个 epoch 的训练步数。
    
    在 DDP 模式下，必须手动计算，因为在 Trainer 创建前 dataloader 
    还没有被 Lightning 转换为使用 DistributedSampler。
    
    Args:
        cfg: 配置字典
        datamodule: Lightning DataModule 实例
        
    Returns:
        每个 epoch 的步数（全局视角，所有设备同步一次算一步）
    """
    import math
    
    # 获取数据集大小
    dataset_size = len(datamodule.train_dataset) if hasattr(datamodule, 'train_dataset') else None
    
    # 获取配置参数
    data_cfg = cfg.get("DATA", {})
    per_device_batch_size = data_cfg.get("BATCH_SIZE", 1)
    
    trainer_cfg = cfg.get("TRAINER", {})
    num_devices = trainer_cfg.get("DEVICES", 1)
    if isinstance(num_devices, list):
        num_devices = len(num_devices)
    
    # 计算或获取 global batch size
    global_batch_size = cfg.get("GLOBAL_BATCH_SIZE")
    if global_batch_size is None:
        global_batch_size = per_device_batch_size * num_devices
        LOGGER.info(
            "GLOBAL_BATCH_SIZE not specified, calculated as %d × %d = %d",
            per_device_batch_size, num_devices, global_batch_size
        )
    
    # 手动计算 steps_per_epoch
    if dataset_size and global_batch_size > 0:
        # 考虑 drop_last (通常为 True)
        drop_last = data_cfg.get("DROP_LAST", True)
        if drop_last:
            steps_per_epoch = dataset_size // global_batch_size
        else:
            steps_per_epoch = math.ceil(dataset_size / global_batch_size)
        
        LOGGER.info(
            "✅ Computed steps_per_epoch=%d (dataset_size=%d, per_device_bs=%d, "
            "num_devices=%d, global_bs=%d, drop_last=%s)",
            steps_per_epoch, dataset_size, per_device_batch_size,
            num_devices, global_batch_size, drop_last
        )
        
        # 验证：与 dataloader 的 len() 进行对比
        try:
            train_loader = datamodule.train_dataloader()
            loader_len = len(train_loader)
            
            # 在 DDP setup 前，loader 没有 DistributedSampler
            # len(loader) 应该约等于 dataset_size / per_device_batch_size
            if drop_last:
                expected_loader_len = dataset_size // per_device_batch_size
            else:
                expected_loader_len = math.ceil(dataset_size / per_device_batch_size)
            
            # 允许 ±1 的误差（边界情况）
            if abs(loader_len - expected_loader_len) <= 1:
                LOGGER.debug(
                    "📊 Dataloader verification OK: len=%d "
                    "(expected ~%d = %d / %d, pre-DDP mode)",
                    loader_len, expected_loader_len, dataset_size, per_device_batch_size
                )
            else:
                LOGGER.warning(
                    "⚠️ Dataloader len=%d differs from expected ~%d "
                    "(dataset_size=%d / per_device_bs=%d). This may indicate a configuration issue.",
                    loader_len, expected_loader_len, dataset_size, per_device_batch_size
                )
        except Exception as e:
            LOGGER.debug("Dataloader length verification skipped: %s", e)
        
        return steps_per_epoch
    else:
        # 降级方案：使用 len(dataloader)
        LOGGER.warning(
            "⚠️ Cannot compute steps_per_epoch (dataset_size=%s, global_bs=%s). "
            "Using len(train_dataloader()) as fallback.",
            dataset_size, global_batch_size
        )
        train_loader = datamodule.train_dataloader()
        return len(train_loader)

def build_training_components(
    cfg: Dict[str, Any],
    resume: bool = False,
) -> Tuple[MaskClassificationPanoptic, COCOPanopticDirectory, Dict[str, Any], Optional[str]]:
    output_dir = Path(cfg.get("OUTPUT_DIR", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("SEED", 0))
    seed_everything(seed, workers=True)
    LOGGER.info("Global seed set to %d", seed)

    datamodule = build_datamodule(cfg)
    # ⭐ 修改：正确计算 DDP 环境下的 steps_per_epoch
    train_loader = datamodule.train_dataloader()
    # ⭐ 简洁调用：计算每个 epoch 的步数
    steps_per_epoch = compute_steps_per_epoch(cfg, datamodule)
    
    start_steps, end_steps = compute_mask_schedule(cfg, steps_per_epoch)

    backbone, backbone_frozen = build_backbone_from_cfg(cfg)
    ov_head = build_open_vocab_head(cfg)
    network = build_network(cfg, backbone, ov_head)
    teacher = build_teacher(cfg)
    module = build_module(
            cfg, 
            network, 
            datamodule, 
            teacher, 
            start_steps, 
            end_steps,
            steps_per_epoch,      # ⭐ 添加缺失参数
            backbone_frozen,      # ⭐ 添加缺失参数
        )

    # ⭐ 新增：处理 Stage B 从 Stage A checkpoint 加载
    stage_a_ckpt = cfg.get("RESUME_FROM_STAGE_A")
    if stage_a_ckpt:
        stage_a_path = Path(stage_a_ckpt)
        if stage_a_path.exists():
            LOGGER.info(f"Loading Stage A checkpoint from: {stage_a_ckpt}")
            try:
                checkpoint = torch.load(stage_a_ckpt, map_location="cpu")
                state_dict = checkpoint.get("state_dict", checkpoint)
                
                # 加载权重，strict=False 允许 LoRA 参数不匹配
                missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)
                
                # 过滤掉预期的 LoRA 相关的 missing keys（因为 Stage A 没有 LoRA）
                lora_missing = [k for k in missing_keys if "lora" in k.lower()]
                other_missing = [k for k in missing_keys if "lora" not in k.lower()]
                
                LOGGER.info(f"✅ Loaded Stage A checkpoint successfully")
                LOGGER.info(f"  - LoRA params (expected missing): {len(lora_missing)}")
                if other_missing:
                    LOGGER.warning(f"  - Other missing keys: {len(other_missing)}")
                    if len(other_missing) < 10:
                        for key in other_missing:
                            LOGGER.warning(f"    - {key}")
                if unexpected_keys:
                    LOGGER.warning(f"  - Unexpected keys: {len(unexpected_keys)}")
                    if len(unexpected_keys) < 10:
                        for key in unexpected_keys:
                            LOGGER.warning(f"    - {key}")
            except Exception as e:
                LOGGER.error(f"❌ Failed to load Stage A checkpoint: {e}")
                raise
        else:
            LOGGER.error(f"❌ Stage A checkpoint not found: {stage_a_ckpt}")
            raise FileNotFoundError(f"Stage A checkpoint not found: {stage_a_ckpt}")

    if hasattr(backbone, "get_lora_summary"):
        summary = backbone.get_lora_summary()
        if summary is not None:
            LOGGER.info("LoRA trainable params: %d", summary.total_trainable)
            for idx, info in enumerate(summary.per_layer):
                attn = [name.upper() for name in ("q", "k", "v", "proj") if info.get(name, False)]
                ffn = [name.upper() for name in ("fc1", "fc2") if info.get(name, False)]
                LOGGER.info(
                    "LoRA layer %02d | attention=%s | ffn=%s",
                    idx,
                    ",".join(attn) if attn else "-",
                    ",".join(ffn) if ffn else "-",
                )

    kwargs, resume_path = trainer_kwargs(cfg, output_dir, resume=resume)
    _log_wandb_metadata(cfg, module, datamodule, steps_per_epoch, output_dir, kwargs.get("logger"))
    return module, datamodule, kwargs, resume_path


def build_eval_components(
    cfg: Dict[str, Any],
) -> Tuple[MaskClassificationPanoptic, COCOPanopticDirectory, Dict[str, Any], Optional[str]]:
    module, datamodule, kwargs, _ = build_training_components(cfg, resume=False)
    return module, datamodule, kwargs, None


def default_argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config-file", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--resume", action="store_true", help="Resume from OUTPUT_DIR/last.ckpt if available")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Configuration overrides")
    return parser
