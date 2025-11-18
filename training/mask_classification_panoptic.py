# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import Any, Dict, List, Optional
from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric.utilities import rank_zero_info
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule

# T ≈ 6159 × 100 = 615 900 total steps
# start steps ≈ [0.2T, 0.5T, 0.8T] ≈ [123 180, 307 950, 492 720]
# end steps   ≈ [0.4T, 0.7T, 1.0T] ≈ [246 360, 431 130, 615 900]

class MaskClassificationPanoptic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        stuff_classes: list[int],
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        lr_lora: Optional[float] = None,
        llrd: float = 0.8,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.2,
        overlap_thresh: float = 0.8,
        open_vocab_kl_weight: float = 0.0,
        open_vocab_kl_temperature: float = 2.0,
        matcher_weights: Optional[Dict[str, Any]] = None,
        aux_weight: float = 0.3,
        stage_config: Optional[Dict[str, Any]] = None,
        steps_per_epoch: Optional[int] = None,
        stage_head_lr: Optional[float] = None,
        stage_other_lr: Optional[float] = None,
        stage_wd_head: float = 0.0,
        stage_wd_other: float = 0.05,
        stage_lora_lr: Optional[float] = None,
        stage_wd_lora: float = 0.05,
        stage_betas: Optional[tuple[float, float]] = None,
        stage_warmup_ratio: float = 0.05,
        distill_feat_weight: float = 0.0,
        distill_itc_weight: float = 0.0,
        distill_temperature: float = 0.07,
        ckpt_path: Optional[str] = None,
        load_ckpt_class_head: bool = True,
        teacher_backbone: Optional[nn.Module] = None,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            lr_lora=lr_lora,
            llrd=llrd,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            load_ckpt_class_head=load_ckpt_class_head,
            distill_feat_weight=distill_feat_weight,
            distill_itc_weight=distill_itc_weight,
            distill_temperature=distill_temperature,
            teacher_backbone=teacher_backbone,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = stuff_classes
        self.open_vocab_enabled = getattr(network, "open_vocab_head", None) is not None
        self.aux_weight = aux_weight
        self.stage_config = stage_config or {}
        self.steps_per_epoch = steps_per_epoch or 0
        self.stage_total_epochs = int(self.stage_config.get("epochs", 1))
        self.stage_head_lr = stage_head_lr or lr
        self.stage_other_lr = stage_other_lr or lr
        self.stage_lora_lr = stage_lora_lr or self.lr_lora
        self.stage_wd_head = stage_wd_head
        self.stage_wd_other = stage_wd_other
        self.stage_wd_lora = stage_wd_lora
        self.stage_betas = stage_betas or (0.9, 0.98)
        self.stage_warmup_ratio = stage_warmup_ratio
        self.stage_total_steps = None
        self.stage_is_a = str(self.stage_config.get("stage", "")).upper() == "A"
        self.stage_is_b = str(self.stage_config.get("stage", "")).upper() == "B"
        self._latest_attn_probs: List[float] = []
        self._stage_best_ckpt_path: Optional[Path] = None
        self._stage_best_metric: Optional[float] = None
        self._stage_a_calibration_done = False
        self._perfplus_fallback_done = False
        self._perfplus_monitor_iters = int(self.stage_config.get("perfplus_monitor_iters", 50))
        self._perfplus_memory_limit_gb = float(self.stage_config.get("perfplus_memory_limit_gb", 23.5))
        self._calibration_mode = False
        self._saved_eval_attn_probs: Optional[torch.Tensor] = None

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            # num_labels= 2,
            no_object_coefficient=no_object_coefficient,
            open_vocab_enabled=self.open_vocab_enabled,
            open_vocab_kl_weight=open_vocab_kl_weight,
            open_vocab_kl_teacher_temp=open_vocab_kl_temperature,
            matcher_weights=matcher_weights,
        )

        if self.teacher_backbone is None and hasattr(network, "teacher_backbone"):
            self.teacher_backbone = getattr(network, "teacher_backbone")

        thing_classes = [i for i in range(num_classes) if i not in stuff_classes]
        self.init_metrics_panoptic(
            thing_classes, stuff_classes, 1 # self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1
        )

    # ------------------------------------------------------------------
    # optimizers / schedulers
    # ------------------------------------------------------------------
    def _compute_stage_total_steps(self) -> int:
        if self.stage_total_steps:
            return self.stage_total_steps
        total_epochs = self.stage_total_epochs or (self.trainer.max_epochs if self.trainer else 1)
        if self.steps_per_epoch:
            total = self.steps_per_epoch * total_epochs
        elif self.trainer is not None and self.trainer.estimated_stepping_batches is not None:
            total = int(self.trainer.estimated_stepping_batches)
        else:
            total = 1
        self.stage_total_steps = max(1, total)
        return self.stage_total_steps

    def _stage_a_param_split(self):
        heads, others, lora = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            lowered = name.lower()
            if "lora" in lowered:
                lora.append((name, param))
                continue
            if name.startswith("network.mask_head") or name.startswith("network.class_head") or name.startswith("network.open_vocab_head"):
                heads.append((name, param))
            else:
                others.append((name, param))
        return heads, others, lora

    def _log_stage_param_stats(self, heads, others, lora):
        head_params = sum(p.numel() for _, p in heads)
        other_params = sum(p.numel() for _, p in others)
        lora_params = sum(p.numel() for _, p in lora)
        total = head_params + other_params + lora_params
        if total == 0:
            return
        rank_zero_info(
            "StageA param split | heads: %d (%.2f%%) other: %d (%.2f%%) lora: %d",
            head_params,
            100.0 * head_params / total,
            other_params,
            100.0 * other_params / total,
            lora_params,
        )

    @staticmethod
    def _is_head_param(name: str) -> bool:
        return name.startswith("network.mask_head") or name.startswith("network.class_head") or name.startswith("network.open_vocab_head")

    def _stage_b_param_split(self):
        heads, norm_bias, others, lora = [], [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if getattr(param, "_is_lora_param", False):
                lora.append((name, param))
                continue
            if self._is_head_param(name):
                heads.append((name, param))
                continue
            if name.endswith(".bias") or name.split(".")[-1] == "bias" or "norm" in name.lower():
                norm_bias.append((name, param))
                continue
            others.append((name, param))
        return heads, norm_bias, others, lora

    def _log_stage_b_param_stats(self, heads, norm_bias, others, lora):
        counts = [
            ("heads", sum(p.numel() for _, p in heads)),
            ("norm_bias", sum(p.numel() for _, p in norm_bias)),
            ("others", sum(p.numel() for _, p in others)),
            ("lora", sum(p.numel() for _, p in lora)),
        ]
        total = sum(value for _, value in counts)
        if total == 0:
            return
        rank_zero_info(
            "StageB param split | heads=%d (%.2f%%) norm_bias=%d (%.2f%%) others=%d (%.2f%%) lora=%d (%.2f%%)",
            counts[0][1],
            100.0 * counts[0][1] / total,
            counts[1][1],
            100.0 * counts[1][1] / total,
            counts[2][1],
            100.0 * counts[2][1] / total,
            counts[3][1],
            100.0 * counts[3][1] / total,
        )

    def _configure_stage_a_optim(self):
        heads, others, lora = self._stage_a_param_split()
        if lora:
            raise RuntimeError("LoRA parameters must remain frozen during Stage A")
        self._log_stage_param_stats(heads, others, lora)
        optim_groups = []
        if heads:
            optim_groups.append(
                {"params": [p for _, p in heads], "lr": self.stage_head_lr, "weight_decay": self.stage_wd_head}
            )
        if others:
            optim_groups.append(
                {"params": [p for _, p in others], "lr": self.stage_other_lr, "weight_decay": self.stage_wd_other}
            )
        rank_zero_info(
            "StageA optim groups | heads lr=%.2e wd=%.3f | others lr=%.2e wd=%.3f",
            self.stage_head_lr,
            self.stage_wd_head,
            self.stage_other_lr,
            self.stage_wd_other,
        )
        optimizer = AdamW(optim_groups, betas=self.stage_betas)
        total_steps = self._compute_stage_total_steps()
        warmup_steps = max(1, int(total_steps * self.stage_warmup_ratio))
        rank_zero_info("StageA scheduler: warmup_steps=%d total_steps=%d", warmup_steps, total_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def _assert_stage_b_backbone_freeze(self) -> None:
        trainable_backbone = [
            name
            for name, param in self.named_parameters()
            if param.requires_grad
            and name.startswith("network.encoder.backbone")
            and not getattr(param, "_is_lora_param", False)
        ]
        if trainable_backbone:
            raise RuntimeError(
                "Stage B requires frozen backbone weights aside from LoRA adapters. Found: %s"
                % trainable_backbone[:5]
            )

    def _configure_stage_b_optim(self):
        heads, norm_bias, others, lora = self._stage_b_param_split()
        self._assert_stage_b_backbone_freeze()
        if not lora:
            raise RuntimeError("Stage B expects trainable LoRA parameters")
        self._log_stage_b_param_stats(heads, norm_bias, others, lora)

        optim_groups = []
        combined_no_wd = heads + norm_bias
        if combined_no_wd:
            optim_groups.append(
                {
                    "params": [p for _, p in combined_no_wd],
                    "lr": self.stage_head_lr,
                    "weight_decay": self.stage_wd_head,
                }
            )
        if others:
            optim_groups.append(
                {
                    "params": [p for _, p in others],
                    "lr": self.stage_other_lr,
                    "weight_decay": self.stage_wd_other,
                }
            )
        optim_groups.append(
            {
                "params": [p for _, p in lora],
                "lr": self.stage_lora_lr,
                "weight_decay": self.stage_wd_lora,
            }
        )

        rank_zero_info(
            "StageB optim groups | heads+norm lr=%.2e wd=%.3f | others lr=%.2e wd=%.3f | lora lr=%.2e wd=%.3f",
            self.stage_head_lr,
            self.stage_wd_head,
            self.stage_other_lr,
            self.stage_wd_other,
            self.stage_lora_lr,
            self.stage_wd_lora,
        )

        optimizer = AdamW(optim_groups, betas=self.stage_betas)
        total_steps = self._compute_stage_total_steps()
        warmup_steps = max(1, int(total_steps * self.stage_warmup_ratio))
        rank_zero_info("StageB scheduler: warmup_steps=%d total_steps=%d", warmup_steps, total_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def configure_optimizers(self):
        if self.stage_is_a:
            return self._configure_stage_a_optim()
        if self.stage_is_b:
            return self._configure_stage_b_optim()
        return super().configure_optimizers()

    def _assert_stage_a_freeze(self) -> None:
        trainable_backbone = [
            name
            for name, param in self.named_parameters()
            if param.requires_grad and name.startswith("network.encoder.backbone")
        ]
        if trainable_backbone:
            raise RuntimeError(
                f"Stage A requires a frozen backbone, but found trainable parameters: {trainable_backbone[:5]}"
            )
        lora = [
            name
            for name, param in self.named_parameters()
            if param.requires_grad and "lora" in name.lower()
        ]
        if lora:
            raise RuntimeError(f"LoRA parameters must be disabled during Stage A: {lora[:5]}")

    def on_fit_start(self) -> None:
        super().on_fit_start()
        stage_label = str(self.stage_config.get("stage", "")).upper()
        rank_zero_info(
            "Stage=%s; backbone_freeze=%s; use_lora=%s; use_lora_attn=%s; use_lora_ffn=%s; query_blocks=%s; fixed_res=%s; epochs=%s; global_bs=%s",
            stage_label,
            self.stage_config.get("backbone_freeze"),
            self.stage_config.get("use_lora"),
            self.stage_config.get("use_lora_attn"),
            self.stage_config.get("use_lora_ffn"),
            self.stage_config.get("query_blocks"),
            self.stage_config.get("fixed_res"),
            self.stage_config.get("epochs"),
            self.stage_config.get("global_batch_size"),
        )
        rank_zero_info(
            "Preset=%s seed=%s",
            self.stage_config.get("resolution_preset"),
            self.stage_config.get("seed"),
        )
        if self.stage_is_a:
            self.stage_total_epochs = int(self.stage_config.get("epochs", self.trainer.max_epochs))
            self.stage_total_steps = None
            self._assert_stage_a_freeze()
            if (
                torch.cuda.is_available()
                and str(self.stage_config.get("resolution_preset", "")).lower() == "perfplus"
            ):
                torch.cuda.reset_peak_memory_stats()

    # def _update_stage_a_attn_probs(self) -> None:
    #     if not self.network.masked_attn_enabled:
    #         return
    #     total_steps = self._compute_stage_total_steps()
    #     progress = 0.0
    #     if total_steps > 0:
    #         progress = min(1.0, max(0.0, self.global_step / float(total_steps)))
    #     value = (1.0 - progress) ** 0.9
    #     value = max(0.5, float(value))
    #     self.network.attn_mask_probs.fill_(value)
    #     self._latest_attn_probs = [float(value)] * len(self.network.attn_mask_probs)

    # def _update_stage_b_attn_probs(self) -> None:
    #     if not self.network.masked_attn_enabled:
    #         return
    #     total_steps = self._compute_stage_total_steps()
    #     if total_steps <= 0:
    #         return
    #     progress = min(1.0, max(0.0, self.global_step / float(total_steps)))
    #     value = max(0.0, float((1.0 - progress) ** 0.9))
    #     self.network.attn_mask_probs.fill_(value)
    #     self._latest_attn_probs = [value] * len(self.network.attn_mask_probs)

    def _monitor_perfplus_memory(self) -> None:
        if self._perfplus_fallback_done:
            return
        if str(self.stage_config.get("resolution_preset", "")).lower() != "perfplus":
            self._perfplus_fallback_done = True
            return
        if not torch.cuda.is_available() or self.device.type != "cuda":
            self._perfplus_fallback_done = True
            return
        if self.current_epoch > 0 or self.global_step >= self._perfplus_monitor_iters:
            self._perfplus_fallback_done = True
            return
        mem_bytes = torch.cuda.max_memory_allocated(self.device)
        mem_gb = mem_bytes / (1024 ** 3)
        if mem_gb > self._perfplus_memory_limit_gb:
            self.stage_config["perfplus_fallback_triggered"] = True
            self.stage_config["resolution_preset"] = "Safe(fallback)"
            self._perfplus_fallback_done = True
            rank_zero_info(
                "PerfPlus memory exceeded (%.2f GB > %.2f GB). Please fallback to Safe preset for stability.",
                mem_gb,
                self._perfplus_memory_limit_gb,
            )

    def _resolve_stage_best_ckpt_path(self) -> Optional[Path]:
        if self._stage_best_ckpt_path is not None:
            return self._stage_best_ckpt_path
        base_dir = self.stage_config.get("output_dir") or (self.trainer.default_root_dir if self.trainer else None)
        if not base_dir:
            return None
        ckpt_name = self.stage_config.get("best_ckpt_name") or f"{str(self.stage_config.get('stage', 'stage')).lower()}_best.ckpt"
        path = Path(base_dir) / ckpt_name
        self._stage_best_ckpt_path = path
        return path

    def _maybe_save_stage_best_ckpt(self) -> None:
        if self.trainer is None:
            return
        pq_all = self.trainer.callback_metrics.get("metrics/val_pq_all")
        if pq_all is None:
            return
        pq_value = float(pq_all)
        if self._stage_best_metric is not None and pq_value <= self._stage_best_metric:
            return
        ckpt_path = self._resolve_stage_best_ckpt_path()
        if ckpt_path is None:
            return
        self._stage_best_metric = pq_value
        if self.trainer.is_global_zero:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            self.trainer.save_checkpoint(str(ckpt_path))
            rank_zero_info(
                "Stage%s best PQ_all=%.3f saved to %s",
                str(self.stage_config.get("stage", "")).upper(),
                pq_value,
                ckpt_path,
            )
        if self.trainer.strategy is not None:
            self.trainer.strategy.barrier("stage_best_ckpt")

    def on_train_batch_end(self, outputs, batch, batch_idx=None, dataloader_idx=None):
        """
        训练 batch 结束时的处理。
        
        对于 Stage A 和 Stage B：
        1. 调用父类的 on_train_batch_end 执行标准的退火逻辑
        2. 执行各自特定的额外处理（内存监控、日志等）
        """
        if self.stage_is_a or self.stage_is_b:
            # ⭐ 关键：调用父类的标准退火实现
            # 这会执行 lightning_module.py 中的分层退火逻辑
            super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)
            
            # Stage A 特定的额外处理
            if self.stage_is_a:
                self._monitor_perfplus_memory()
            
            # 通用的清理操作（Stage A 和 B 都需要）
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        else:
            # 非 Stage A/B 的情况，完全使用父类逻辑
            super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if self._latest_attn_probs:
            for i, prob in enumerate(self._latest_attn_probs):
                self.log(
                    f"attn_mask_prob_epoch_{i}",
                    prob,
                    prog_bar=False,
                    sync_dist=True,
                )
            rank_zero_info(
                "Epoch %d stage=%s attn_mask_probs=%s",
                self.current_epoch,
                str(self.stage_config.get("stage", "")).upper(),
                [f"{p:.3f}" for p in self._latest_attn_probs],
            )

            # ⭐ 添加：详细的进度调试信息
            if self.attn_mask_annealing_enabled and self.network.masked_attn_enabled:
                total_steps = self._compute_stage_total_steps()
                progress = self.global_step / total_steps if total_steps > 0 else 0
                rank_zero_info(
                    "Stage %s Annealing Debug | global_step=%d total_steps=%d progress=%.2f%% "
                    "steps_per_epoch=%d stage_total_epochs=%d",
                    stage_label,
                    self.global_step,
                    total_steps,
                    progress * 100,
                    self.steps_per_epoch,
                    self.stage_total_epochs,
                )

    # ------------------------------------------------------------------

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        if self.stage_is_b and hasattr(self.network, "attn_mask_probs"):
            self._saved_eval_attn_probs = self.network.attn_mask_probs.detach().clone()
            self.network.attn_mask_probs.zero_()

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        outputs = self(transformed_imgs)
        mask_logits_per_layer = outputs["mask_logits"]
        class_logits_per_layer = outputs["class_logits"]
        open_vocab_logits_per_layer = outputs.get("open_vocab_logits")

        is_crowds = [target["is_crowd"] for target in targets]
        original_targets = targets
        targets = self.to_per_pixel_targets_panoptic(targets)
        

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            # ⭐ 只评估最后一个block（训练时中间的blocks是辅助损失，验证时不需要）
            if i != len(mask_logits_per_layer) - 1:
                continue

            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits, img_sizes
            )
            ov_logits = None
            if open_vocab_logits_per_layer:
                ov_logits = open_vocab_logits_per_layer[min(i, len(open_vocab_logits_per_layer) - 1)]
            preds = self.to_per_pixel_preds_panoptic(
                mask_logits,
                class_logits,
                self.stuff_classes,
                self.mask_thresh,
                self.overlap_thresh,
                open_vocab_logits=ov_logits if self.open_vocab_enabled else None,
            )
            metric_idx = 0
            self.update_metrics_panoptic(preds, targets, is_crowds, metric_idx)
            
        if (
            False #batch_idx < 1 #batch_idx % 5 == 0
            and self.trainer.is_global_zero
            and not getattr(self, "_calibration_mode", False)
        ):  # 每5个batch可视化一次，或者完全禁用
            self.plot_panoptic(
                img=imgs[0],  # 第一张图像
                targets=original_targets[0],  # 对应的target
                mask_logits=mask_logits_per_layer[-1][0],  # 最后一个block的第一张图像的mask logits
                class_logits=class_logits_per_layer[-1][0],  # 最后一个block的第一张图像的class logits
                log_prefix="val",
                block_idx=len(mask_logits_per_layer) - 1,  # 最后一个block
                batch_idx=batch_idx,
            )
    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_panoptic("val")
        # self._maybe_save_stage_best_ckpt()
        if self.stage_is_b and self._saved_eval_attn_probs is not None:
            self.network.attn_mask_probs.data.copy_(self._saved_eval_attn_probs)
            self._saved_eval_attn_probs = None

    def on_validation_end(self):
        self._on_eval_end_panoptic("val")

    def on_fit_end(self) -> None:
        super().on_fit_end()
        if self._stage_best_metric is not None:
            rank_zero_info(
                "Stage%s final best PQ_all=%.3f @ %s",
                str(self.stage_config.get("stage", "")).upper(),
                self._stage_best_metric,
                self._resolve_stage_best_ckpt_path(),
            )
        if self.stage_is_a:
            self._run_stage_a_calibration()

    # ------------------------------------------------------------------
    # Stage A calibration utilities
    # ------------------------------------------------------------------

    def _stage_a_get_val_loader(self):
        if self.trainer is None:
            return None
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return None
        if getattr(datamodule, "val_dataset", None) is None:
            try:
                datamodule.setup(None)
            except TypeError:
                datamodule.setup()
        loader = datamodule.val_dataloader()
        if isinstance(loader, (list, tuple)):
            return loader[0]
        return loader

    def _move_to_device(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, list):
            return [self._move_to_device(item, device) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self._move_to_device(item, device) for item in obj)
        if isinstance(obj, dict):
            return {key: self._move_to_device(value, device) for key, value in obj.items()}
        return obj

    def _collect_panoptic_scores(self):
        if not self.metrics:
            return None
        target_idx = len(self.metrics) - 1
        pq_all = pq_things = pq_stuff = None
        for idx, metric in enumerate(self.metrics):
            try:
                result = metric.compute()[:-1]
            except Exception:
                metric.reset()
                continue
            metric.reset()
            if idx != target_idx:
                continue
            if result.numel() == 0:
                continue
            pq = result[:, 0]
            num_things = len(metric.things)
            pq_all = float(pq.mean())
            pq_things = float(pq[:num_things].mean()) if num_things > 0 else None
            pq_stuff = float(pq[num_things:].mean()) if pq.shape[0] > num_things else None
        if pq_all is None:
            return None
        return {
            "pq_all": pq_all,
            "pq_things": pq_things,
            "pq_stuff": pq_stuff,
        }

    def _evaluate_calibration_combo(self, loader, tau_value: float, bias_scale: float, base_bias: Optional[torch.Tensor]):
        head = getattr(self.network, "open_vocab_head", None)
        if head is None:
            return None
        device = self.device
        with torch.inference_mode():
            head.temperature.data.copy_(torch.tensor(float(tau_value), device=head.temperature.device))
            if base_bias is not None and head.per_class_bias is not None:
                scaled = base_bias * bias_scale
                head.per_class_bias.data.copy_(scaled.to(head.per_class_bias.device))
            for metric in self.metrics:
                metric.reset()
            for batch_idx, batch in enumerate(loader):
                imgs, targets = batch
                imgs = tuple(self._move_to_device(img, device) for img in imgs)
                targets = tuple(self._move_to_device(target, device) for target in targets)
                self.eval_step((imgs, targets), batch_idx=batch_idx, log_prefix="calib")
            scores = self._collect_panoptic_scores()
        return scores

    def _run_stage_a_calibration(self) -> None:
        if (
            not self.stage_is_a
            or not self.open_vocab_enabled
            or self.trainer is None
            or self._stage_a_calibration_done
        ):
            return
        strategy = getattr(self.trainer, "strategy", None)
        if strategy is not None:
            strategy.barrier("stage_a_calibration_sync_start")
        if not self.trainer.is_global_zero:
            self._stage_a_calibration_done = True
            if strategy is not None:
                strategy.barrier("stage_a_calibration_sync_done")
            return
        loader = self._stage_a_get_val_loader()
        if loader is None:
            rank_zero_info("StageA calibration skipped: missing validation dataloader")
            self._stage_a_calibration_done = True
            if strategy is not None:
                strategy.barrier("stage_a_calibration_sync_done")
            return
        head = getattr(self.network, "open_vocab_head", None)
        if head is None:
            self._stage_a_calibration_done = True
            if strategy is not None:
                strategy.barrier("stage_a_calibration_sync_done")
            return
        base_tau = float(head.temperature.detach().clone().cpu())
        base_bias = head.per_class_bias.detach().clone() if head.per_class_bias is not None else None
        tau_grid = sorted({max(1e-6, base_tau * scale) for scale in (0.9, 1.0, 1.1)})
        bias_scales = [1.0]
        if base_bias is not None and base_bias.numel() > 0:
            bias_scales = [0.9, 1.0, 1.1]
        results = []
        was_training = self.training
        self.eval()
        self._calibration_mode = True
        try:
            for tau_value in tau_grid:
                for bias_scale in bias_scales:
                    loader_instance = self._stage_a_get_val_loader()
                    if loader_instance is None:
                        continue
                    scores = self._evaluate_calibration_combo(
                        loader_instance,
                        tau_value,
                        bias_scale,
                        base_bias,
                    )
                    if scores is None:
                        continue
                    scores["tau"] = tau_value
                    scores["bias_scale"] = bias_scale
                    results.append(scores)
        finally:
            head.temperature.data.copy_(torch.tensor(base_tau, device=head.temperature.device))
            if base_bias is not None and head.per_class_bias is not None:
                head.per_class_bias.data.copy_(base_bias.to(head.per_class_bias.device))
            self._calibration_mode = False
            self.train(was_training)
        self._stage_a_calibration_done = True
        if not results:
            rank_zero_info("StageA calibration produced no results")
            if strategy is not None:
                strategy.barrier("stage_a_calibration_sync_done")
            return
        best = max(results, key=lambda item: item.get("pq_all", float("-inf")))
        rank_zero_info(
            "StageA calibration sweep completed over %d combos. Best tau=%.4f bias_scale=%.2f | PQ_all=%.3f PQ_things=%s PQ_stuff=%s",
            len(results),
            best.get("tau", base_tau),
            best.get("bias_scale", 1.0),
            best.get("pq_all", float("nan")),
            "{:.3f}".format(best.get("pq_things")) if best.get("pq_things") is not None else "n/a",
            "{:.3f}".format(best.get("pq_stuff")) if best.get("pq_stuff") is not None else "n/a",
        )
        if strategy is not None:
            strategy.barrier("stage_a_calibration_sync_done")
