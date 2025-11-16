from __future__ import annotations

import argparse
import json
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import torch
import yaml
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

def build_datamodule(cfg: Dict[str, Any]) -> COCOPanopticDirectory:
    data_cfg = cfg.get("DATA", {})
    batch_size = int(data_cfg.get("BATCH_SIZE", 2))
    num_workers = int(data_cfg.get("NUM_WORKERS", 4))
    img_max = int(data_cfg.get("TRAIN_RES_MAX", 1024))
    img_size = (img_max, img_max)
    datamodule = COCOPanopticDirectory(
        path=data_cfg.get("ROOT", ""),
        stuff_classes=list(range(len(COCO_THINGS_80), len(COCO_THINGS_80) + len(COCO_STUFF_53))),
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        num_classes=len(COCO_THINGS_80) + len(COCO_STUFF_53),
        check_empty_targets=data_cfg.get("CHECK_EMPTY_TARGETS", True),
    )
    datamodule.setup("fit")
    return datamodule


def build_open_vocab_head(cfg: Dict[str, Any]) -> Optional[OpenVocabHead]:
    ov_cfg = cfg.get("OPEN_VOCAB", {})
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

    per_class_bias = ov_cfg.get("PER_CLASS_BIAS", None)
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
    )
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
        fuse_closed_head=bool(cfg.get("OPEN_VOCAB", {}).get("FUSE_CLOSED_HEAD", False)),
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
) -> MaskClassificationPanoptic:
    solver_cfg = cfg.get("SOLVER", {})
    loss_cfg = cfg.get("LOSS", {})
    anneal_cfg = cfg.get("ANNEAL", {})
    data_cfg = cfg.get("DATA", {})

    img_max = int(data_cfg.get("TRAIN_RES_MAX", 1024))
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
        open_vocab_kl_weight=float(loss_cfg.get("OV", {}).get("KL_TEXT_WEIGHT", 0.0)),
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


def build_training_components(
    cfg: Dict[str, Any],
    resume: bool = False,
) -> Tuple[MaskClassificationPanoptic, COCOPanopticDirectory, Dict[str, Any], Optional[str]]:
    output_dir = Path(cfg.get("OUTPUT_DIR", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(cfg)
    steps_per_epoch = len(datamodule.train_dataloader())
    start_steps, end_steps = compute_mask_schedule(cfg, steps_per_epoch)

    backbone, _ = build_backbone_from_cfg(cfg)
    ov_head = build_open_vocab_head(cfg)
    network = build_network(cfg, backbone, ov_head)
    teacher = build_teacher(cfg)
    module = build_module(cfg, network, datamodule, teacher, start_steps, end_steps)

    if hasattr(backbone, "get_lora_summary"):
        summary = backbone.get_lora_summary()
        if summary is not None:
            LOGGER.info("LoRA trainable params: %d", summary.total_trainable)

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
