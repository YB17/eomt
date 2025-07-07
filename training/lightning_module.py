# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from:
# - the torchmetrics library by the PyTorch Lightning team
# - the Mask2Former repository by Facebook, Inc. and its affiliates
# All used under the Apache 2.0 License.
# ---------------------------------------------------------------

import math
from typing import Optional, cast
import lightning
from lightning.fabric.utilities import rank_zero_info
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.detection import PanopticQuality, MeanAveragePrecision
from torchmetrics.functional.detection._panoptic_quality_common import (
    _prepocess_inputs,
    _Color,
    _get_color_areas,
    _calculate_iou,
)
import wandb
from PIL import Image
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import io
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import interpolate
from torchvision.transforms.v2.functional import pad
import cv2

from training.two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule

bold_green = "\033[1;32m"
reset = "\033[0m"


class LightningModule(lightning.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]],
        attn_mask_annealing_end_steps: Optional[list[int]],
        lr: float,
        llrd: float,
        weight_decay: float,
        poly_power: float,
        warmup_steps: tuple[int, int],
        ckpt_path=None,
        load_ckpt_class_head=True,
    ):
        super().__init__()

        self.network = network
        self.img_size = img_size
        self.num_classes = num_classes
        self.attn_mask_annealing_enabled = attn_mask_annealing_enabled
        self.attn_mask_annealing_start_steps = attn_mask_annealing_start_steps
        self.attn_mask_annealing_end_steps = attn_mask_annealing_end_steps
        self.lr = lr
        self.llrd = llrd
        self.weight_decay = weight_decay
        self.poly_power = poly_power
        self.warmup_steps = warmup_steps

        self.strict_loading = False

        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)

            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            ckpt = {k: v for k, v in ckpt.items() if "criterion.empty_weight" not in k}

            if not load_ckpt_class_head:
                ckpt = {
                    k: v
                    for k, v in ckpt.items()
                    if "class_head" not in k and "class_predictor" not in k
                }

            incompatible_keys = self.load_state_dict(ckpt, strict=False)

            if incompatible_keys.missing_keys:
                if not load_ckpt_class_head:
                    missing_keys = [
                        key
                        for key in incompatible_keys.missing_keys
                        if "class_head" not in key and "class_predictor" not in key
                    ]
                else:
                    missing_keys = incompatible_keys.missing_keys

                if missing_keys:
                    raise ValueError(f"Missing keys: {missing_keys}")

            if incompatible_keys.unexpected_keys:
                raise ValueError(
                    f"Unexpected keys: {incompatible_keys.unexpected_keys}"
                )

        self.log = torch.compiler.disable(self.log)  # type: ignore

    '''
    配置优化器和学习率调度器:
    整体效果：
    - 对 预训练 Backbone 应用更小的、分层衰减的学习率，避免过度破坏预训练特征；
    - 对 新加模块（解码器、分类头等）使用较大学习率，快速适配分割任务；
    - 结合 Warmup + PolyDecay 保证训练初期稳定，后期平滑收敛。
    这样的优化器配置能显著提升模型在分割任务上的收敛速度与最终性能，同时兼顾对预训练骨干的保护与新任务头的快速学习。
    '''
    def configure_optimizers(self):
        encoder_param_names = {
            n for n, _ in self.network.encoder.backbone.named_parameters()
        }
        backbone_param_groups = []
        other_param_groups = []
        backbone_blocks = len(self.network.encoder.backbone.blocks)
        block_i = backbone_blocks

        for name, param in reversed(list(self.named_parameters())):
            lr = self.lr
            if name.replace("network.encoder.backbone.", "") in encoder_param_names:
                name_list = name.split(".")
                is_block = False
                for i, key in enumerate(name_list):
                    if key == "blocks":
                        block_i = int(name_list[i + 1])
                        is_block = True
                if is_block or block_i == 0:
                    lr *= self.llrd ** (backbone_blocks - 1 - block_i)
                backbone_param_groups.append(
                    {"params": [param], "lr": lr, "name": name}
                )
            else:
                other_param_groups.append(
                    {"params": [param], "lr": self.lr, "name": name}
                )

        param_groups = backbone_param_groups + other_param_groups
        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)

        scheduler = TwoStageWarmupPolySchedule(
            optimizer,
            num_backbone_params=len(backbone_param_groups),
            warmup_steps=self.warmup_steps,
            total_steps=self.trainer.estimated_stepping_batches,
            poly_power=self.poly_power,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self, imgs):
        x = imgs / 255.0

        return self.network(x)

    def infer_semantic(img, target):
        with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
            imgs = [img.to(device)]
            img_sizes = [img.shape[-2:] for img in imgs]
            crops, origins = model.window_imgs_semantic(imgs)

            mask_logits_per_layer, class_logits_per_layer = model(crops)
            mask_logits = F.interpolate(mask_logits_per_layer[-1], data.img_size, mode="bilinear")

            crop_logits = model.to_per_pixel_logits_semantic(
                mask_logits, class_logits_per_layer[-1]
            )
            logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
            preds = logits[0].argmax(0).cpu()

        pred_array = preds.numpy()
        target_array = model.to_per_pixel_targets_semantic([target], IGNORE_INDEX)[0].numpy()
        return pred_array, target_array

    '''
    训练步骤:
    1. 前向传播:
    - 输入图像 x 归一化到 [0, 1]
    - 输入网络得到 mask_logits_per_block 和 class_logits_per_block
    - 遍历每个 block，计算损失
    - 累加损失到 losses_all_blocks
    - 返回 losses_all_blocks
    2. 反向传播:
    - 计算梯度
    - 更新模型参数
    - 返回 losses_all_blocks
    '''
    def training_step(self, batch, batch_idx):
        # 获取输入图像和目标
        imgs, targets = batch

        # 前向传播，获取 mask_logits_per_block 和 class_logits_per_block
        mask_logits_per_block, class_logits_per_block = self(imgs)

        # 初始化损失字典
        losses_all_blocks = {}

        # 遍历每个 block，计算损失
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block))
        ):
            # 计算损失
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )

            # 添加 block 后缀
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses
        
        # 在训练时每隔一定步数可视化一张图像（避免太频繁影响训练速度）
        # if batch_idx % 5 == 0:  # 改为每500个batch可视化一次，或者完全禁用
        #     self.plot_panoptic(
        #         img=imgs[0],  # 第一张图像
        #         targets=targets[0],  # 对应的target
        #         mask_logits=mask_logits_per_block[-1][0],  # 最后一个block的第一张图像的mask logits
        #         class_logits=class_logits_per_block[-1][0],  # 最后一个block的第一张图像的class logits
        #         log_prefix="train",
        #         block_idx=len(mask_logits_per_block) - 1,  # 最后一个block
        #         batch_idx=batch_idx,
        #     )
            # [0.2T, 0.5T, 0.8T] = [40, 100, 160]
            # [0.4T, 0.7T, 1.0T] = [80, 140, 200]
        
        return self.criterion.loss_total(losses_all_blocks, self.log)

    def validation_step(self, batch, batch_idx=0):
        return self.eval_step(batch, batch_idx, "val")

    '''
    验证步骤:
    1. 前向传播:
    - 输入图像 x 归一化到 [0, 1]
    - 输入网络得到 mask_logits_per_block 和 class_logits_per_block
    '''
    def mask_annealing(self, start_iter, current_iter, final_iter):
        device = self.device
        dtype = self.network.attn_mask_probs[0].dtype
        if current_iter < start_iter:
            return torch.ones(1, device=device, dtype=dtype)
        elif current_iter >= final_iter:
            return torch.zeros(1, device=device, dtype=dtype)
        else:
            progress = (current_iter - start_iter) / (final_iter - start_iter)
            progress = torch.tensor(progress, device=device, dtype=dtype)
            return (1.0 - progress).pow(self.poly_power)

    '''
    训练结束时:
    - 如果 attn_mask_annealing_enabled 为 True
    - 遍历每个 block，更新 attn_mask_probs
    - 记录 attn_mask_prob
    '''
    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx=None,
        dataloader_idx=None,
    ):
        if self.attn_mask_annealing_enabled:
            for i in range(self.network.num_blocks):
                self.network.attn_mask_probs[i] = self.mask_annealing(
                    self.attn_mask_annealing_start_steps[i],
                    self.global_step,
                    self.attn_mask_annealing_end_steps[i],
                )

            for i, attn_mask_prob in enumerate(self.network.attn_mask_probs):
                self.log(
                    f"attn_mask_prob_{i}",
                    attn_mask_prob,
                    on_step=True,
                )
        
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    '''
    初始化语义度量:
    - 创建一个 nn.ModuleList，包含 num_blocks 个 MulticlassJaccardIndex 度量
    - 每个度量用于计算特定 block 的语义分割性能
    - 度量使用指定的 ignore_idx 忽略背景类
    - 度量在每个 block 上独立计算
    '''
    def init_metrics_semantic(self, ignore_idx, num_blocks):
        self.metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,
                )
                for _ in range(num_blocks)
            ]
        )

    '''
    初始化实例度量:
    - 创建一个 nn.ModuleList，包含 num_blocks 个 MeanAveragePrecision 度量
    - 每个度量用于计算特定 block 的实例分割性能
    - 度量使用 segm 类型的 IoU 计算
    '''
    def init_metrics_instance(self, num_blocks):
        self.metrics = nn.ModuleList(
            [MeanAveragePrecision(iou_type="segm") for _ in range(num_blocks)]
        )

    '''
    初始化全景度量:
    - 创建一个 nn.ModuleList，包含 num_blocks 个 PanopticQuality 度量
    - 每个度量用于计算特定 block 的全景分割性能
    - 度量使用指定的 thing_classes 和 stuff_classes
    '''
    def init_metrics_panoptic(self, thing_classes, stuff_classes, num_blocks):
        self.metrics = nn.ModuleList(
            [
                PanopticQuality(
                    thing_classes,
                    stuff_classes + [self.num_classes],
                    return_sq_and_rq=True,
                    return_per_class=True,
                )
                for _ in range(num_blocks)
            ]
        )

    @torch.compiler.disable
    def update_metrics_semantic(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        block_idx,
    ):
        for i in range(len(preds)):
            self.metrics[block_idx].update(preds[i][None, ...], targets[i][None, ...])

    @torch.compiler.disable
    def update_metrics_instance(
        self,
        preds: list[dict],
        targets: list[dict],
        block_idx,
    ):
        self.metrics[block_idx].update(preds, targets)

    @torch.compiler.disable
    def update_metrics_panoptic(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        is_crowds: list[torch.Tensor],
        block_idx,
    ):
        for i in range(len(preds)):
            metric = self.metrics[block_idx]
            flatten_pred = _prepocess_inputs(
                metric.things,
                metric.stuffs,
                preds[i][None, ...],
                metric.void_color,
                metric.allow_unknown_preds_category,
            )[0]
            flatten_target = _prepocess_inputs(
                metric.things,
                metric.stuffs,
                targets[i][None, ...],
                metric.void_color,
                True,
            )[0]

            pred_areas = cast(
                dict[_Color, torch.Tensor], _get_color_areas(flatten_pred)
            )
            target_areas = cast(
                dict[_Color, torch.Tensor], _get_color_areas(flatten_target)
            )
            intersection_matrix = torch.transpose(
                torch.stack((flatten_pred, flatten_target), -1), -1, -2
            )
            intersection_areas = cast(
                dict[tuple[_Color, _Color], torch.Tensor],
                _get_color_areas(intersection_matrix),
            )

            pred_segment_matched = set()
            target_segment_matched = set()
            for pred_color, target_color in intersection_areas:
                if is_crowds[i][target_color[1]]:
                    continue
                if target_color == metric.void_color:
                    continue
                if pred_color[0] != target_color[0]:
                    continue
                iou = _calculate_iou(
                    pred_color,
                    target_color,
                    pred_areas,
                    target_areas,
                    intersection_areas,
                    metric.void_color,
                )
                continuous_id = metric.cat_id_to_continuous_id[target_color[0]]
                if iou > 0.5:
                    pred_segment_matched.add(pred_color)
                    target_segment_matched.add(target_color)
                    metric.iou_sum[continuous_id] += iou
                    metric.true_positives[continuous_id] += 1

            false_negative_colors = set(target_areas) - target_segment_matched
            false_positive_colors = set(pred_areas) - pred_segment_matched

            false_negative_colors.discard(metric.void_color)
            false_positive_colors.discard(metric.void_color)

            for target_color in list(false_negative_colors):
                void_target_area = intersection_areas.get(
                    (metric.void_color, target_color), 0
                )
                if void_target_area / target_areas[target_color] > 0.5:
                    false_negative_colors.discard(target_color)

            crowd_by_cat_id = {}
            for false_negative_color in false_negative_colors:
                if is_crowds[i][false_negative_color[1]]:
                    crowd_by_cat_id[false_negative_color[0]] = false_negative_color[1]
                    continue

                continuous_id = metric.cat_id_to_continuous_id[false_negative_color[0]]
                metric.false_negatives[continuous_id] += 1

            for pred_color in list(false_positive_colors):
                pred_void_crowd_area = intersection_areas.get(
                    (pred_color, metric.void_color), 0
                )

                if pred_color[0] in crowd_by_cat_id:
                    crowd_color = (pred_color[0], crowd_by_cat_id[pred_color[0]])
                    pred_void_crowd_area += intersection_areas.get(
                        (pred_color, crowd_color), 0
                    )

                if pred_void_crowd_area / pred_areas[pred_color] > 0.5:
                    false_positive_colors.discard(pred_color)

            for false_positive_color in false_positive_colors:
                continuous_id = metric.cat_id_to_continuous_id[false_positive_color[0]]
                metric.false_positives[continuous_id] += 1

    def block_postfix(self, block_idx):
        if not self.network.masked_attn_enabled:
            return ""
        return (
            f"_block_{-len(self.metrics) + block_idx + 1}"
            if block_idx != self.network.num_blocks
            else ""
        )

    def _on_eval_epoch_end_semantic(self, log_prefix, log_per_class=False):
        for i, metric in enumerate(self.metrics):  # type: ignore
            iou_per_class = metric.compute()
            metric.reset()

            block_postfix = self.block_postfix(i)
            if log_per_class:
                for class_idx, iou in enumerate(iou_per_class):
                    self.log(
                        f"metrics/{log_prefix}_iou_class_{class_idx}{block_postfix}",
                        iou,
                    )

            iou_all = float(iou_per_class.mean())
            self.log(
                f"metrics/{log_prefix}_iou_all{block_postfix}",
                iou_all,
            )

    def _on_eval_epoch_end_instance(self, log_prefix):
        for i, metric in enumerate(self.metrics):  # type: ignore
            results = metric.compute()
            metric.reset()

            block_postfix = self.block_postfix(i)
            self.log(
                f"metrics/{log_prefix}_ap_all{block_postfix}",
                results["map"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_small_all{block_postfix}",
                results["map_small"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_medium_all{block_postfix}",
                results["map_medium"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_large_all{block_postfix}",
                results["map_large"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_50_all{block_postfix}",
                results["map_50"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_75_all{block_postfix}",
                results["map_75"],
            )

    def _on_eval_epoch_end_panoptic(self, log_prefix, log_per_class=False):
        for i, metric in enumerate(self.metrics):  # type: ignore
            result = metric.compute()[:-1]
            metric.reset()

            pq, sq, rq = result[:, 0], result[:, 1], result[:, 2]

            block_postfix = self.block_postfix(i)
            if log_per_class:
                for class_idx in range(len(pq)):
                    self.log(
                        f"metrics/{log_prefix}_pq_class_{class_idx}{block_postfix}",
                        pq[class_idx],
                    )
                    self.log(
                        f"metrics/{log_prefix}_sq_class_{class_idx}{block_postfix}",
                        sq[class_idx],
                    )
                    self.log(
                        f"metrics/{log_prefix}_rq_class_{class_idx}{block_postfix}",
                        rq[class_idx],
                    )

            self.log(
                f"metrics/{log_prefix}_pq_all{block_postfix}",
                pq.mean(),
            )
            self.log(f"metrics/{log_prefix}_sq_all{block_postfix}", sq.mean())
            self.log(f"metrics/{log_prefix}_rq_all{block_postfix}", rq.mean())

            num_things = len(metric.things)
            pq_things, sq_things, rq_things = (
                result[:num_things, 0],
                result[:num_things, 1],
                result[:num_things, 2],
            )
            pq_stuff, sq_stuff, rq_stuff = (
                result[num_things:, 0],
                result[num_things:, 1],
                result[num_things:, 2],
            )

            self.log(
                f"metrics/{log_prefix}_pq_things{block_postfix}",
                pq_things.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_sq_things{block_postfix}",
                sq_things.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_rq_things{block_postfix}",
                rq_things.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_pq_stuff{block_postfix}",
                pq_stuff.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_sq_stuff{block_postfix}",
                sq_stuff.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_rq_stuff{block_postfix}",
                rq_stuff.mean(),
            )

    def _on_eval_end_semantic(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}mIoU: {self.trainer.callback_metrics[f'metrics/{log_prefix}_iou_all'] * 100:.1f}{reset}"
            )

    def _on_eval_end_instance(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}mAP All: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_all'] * 100:.1f} | "
                f"mAP Small: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_small_all'] * 100:.1f} | "
                f"mAP Medium: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_medium_all'] * 100:.1f} | "
                f"mAP Large: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_large_all'] * 100:.1f}{reset}"
            )

    def _on_eval_end_panoptic(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}PQ All: {self.trainer.callback_metrics[f'metrics/{log_prefix}_pq_all'] * 100:.1f} | "
                f"PQ Things: {self.trainer.callback_metrics[f'metrics/{log_prefix}_pq_things'] * 100:.1f} | "
                f"PQ Stuff: {self.trainer.callback_metrics[f'metrics/{log_prefix}_pq_stuff'] * 100:.1f}{reset}"
            )

    @torch.compiler.disable
    def plot_semantic(
        self,
        img,
        target,
        logits,
        log_prefix,
        block_idx,
        batch_idx,
        cmap="tab20",
    ):
        fig, axes = plt.subplots(1, 3, figsize=[15, 5], sharex=True, sharey=True)

        axes[0].imshow(img.cpu().numpy().transpose(1, 2, 0))
        axes[0].axis("off")

        target = target.cpu().numpy()
        unique_classes = np.unique(target)

        preds = torch.argmax(logits, dim=0).cpu().numpy()
        unique_classes = np.unique(np.concatenate((unique_classes, np.unique(preds))))

        num_classes = len(unique_classes)
        colors = plt.get_cmap(cmap, num_classes)(np.linspace(0, 1, num_classes))  # type: ignore

        if self.ignore_idx in unique_classes:
            colors[unique_classes == self.ignore_idx] = [0, 0, 0, 1]  # type: ignore

        custom_cmap = mcolors.ListedColormap(colors)  # type: ignore
        norm = mcolors.Normalize(0, num_classes - 1)

        axes[1].imshow(
            np.digitize(target, unique_classes) - 1,
            cmap=custom_cmap,
            norm=norm,
            interpolation="nearest",
        )
        axes[1].axis("off")

        if preds is not None:
            axes[2].imshow(
                np.digitize(preds, unique_classes, right=True),
                cmap=custom_cmap,
                norm=norm,
                interpolation="nearest",
            )
            axes[2].axis("off")

        patches = [
            Line2D([0], [0], color=colors[i], lw=4, label=str(unique_classes[i]))
            for i in range(num_classes)
        ]

        fig.legend(handles=patches, loc="upper left")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, facecolor="black")
        plt.close(fig)
        buf.seek(0)

        block_postfix = self.block_postfix(block_idx)
        name = f"{log_prefix}_pred_{batch_idx}{block_postfix}"
        self.trainer.logger.experiment.log({name: [wandb.Image(Image.open(buf))]})

    @torch.compiler.disable
    def scale_img_size_semantic(self, size: tuple[int, int]):
        factor = max(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    @torch.compiler.disable
    def window_imgs_semantic(self, imgs):
        crops, origins = [], []

        for i in range(len(imgs)):
            img = imgs[i]
            new_h, new_w = self.scale_img_size_semantic(img.shape[-2:])
            pil_img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
            resized_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized_img = (
                torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).to(img.device)
            )

            num_crops = math.ceil(max(resized_img.shape[-2:]) / min(self.img_size))
            overlap = num_crops * min(self.img_size) - max(resized_img.shape[-2:])
            overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0

            for j in range(num_crops):
                start = int(j * (min(self.img_size) - overlap_per_crop))
                end = start + min(self.img_size)
                if resized_img.shape[-2] > resized_img.shape[-1]:
                    crop = resized_img[:, start:end, :]
                else:
                    crop = resized_img[:, :, start:end]

                crops.append(crop)
                origins.append((i, start, end))

        return torch.stack(crops), origins

    def revert_window_logits_semantic(self, crop_logits, origins, img_sizes):
        logit_sums, logit_counts = [], []
        for size in img_sizes:
            h, w = self.scale_img_size_semantic(size)
            logit_sums.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )
            logit_counts.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )

        for crop_i, (img_i, start, end) in enumerate(origins):
            if img_sizes[img_i][0] > img_sizes[img_i][1]:
                logit_sums[img_i][:, start:end, :] += crop_logits[crop_i]
                logit_counts[img_i][:, start:end, :] += 1
            else:
                logit_sums[img_i][:, :, start:end] += crop_logits[crop_i]
                logit_counts[img_i][:, :, start:end] += 1

        return [
            interpolate(
                (sums / counts)[None, ...],
                img_sizes[i],
                mode="bilinear",
            )[0]
            for i, (sums, counts) in enumerate(zip(logit_sums, logit_counts))
        ]

    @staticmethod
    def to_per_pixel_logits_semantic(
        mask_logits: torch.Tensor, class_logits: torch.Tensor
    ):
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_semantic(
        targets: list[dict],
        ignore_idx,
    ):
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = torch.full(
                target["masks"].shape[-2:],
                ignore_idx,
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[mask] = target["labels"][i]

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    def scale_img_size_instance_panoptic(self, size: tuple[int, int]):
        factor = min(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    @torch.compiler.disable
    def resize_and_pad_imgs_instance_panoptic(self, imgs):
        transformed_imgs = []

        for img in imgs:
            new_h, new_w = self.scale_img_size_instance_panoptic(img.shape[-2:])

            pil_img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized_img = (
                torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).to(img.device)
            )

            pad_h = max(0, self.img_size[-2] - resized_img.shape[-2])
            pad_w = max(0, self.img_size[-1] - resized_img.shape[-1])
            padding = [0, 0, pad_w, pad_h]

            padded_img = pad(resized_img, padding)

            transformed_imgs.append(padded_img)

        return torch.stack(transformed_imgs)

    @torch.compiler.disable
    def revert_resize_and_pad_logits_instance_panoptic(
        self, transformed_logits, img_sizes
    ):
        logits = []
        for i in range(len(transformed_logits)):
            scaled_size = self.scale_img_size_instance_panoptic(img_sizes[i])
            logits_i = transformed_logits[i][:, : scaled_size[0], : scaled_size[1]]
            logits_i = interpolate(
                logits_i[None, ...],
                img_sizes[i],
                mode="bilinear",
            )[0]
            logits.append(logits_i)

        return logits

    def to_per_pixel_preds_panoptic(
        self, mask_logits_list, class_logits, stuff_classes, mask_thresh, overlap_thresh
    ):
        scores, classes = class_logits.softmax(dim=-1).max(-1)
        preds_list = []

        for i in range(len(mask_logits_list)):
            preds = -torch.ones(
                (*mask_logits_list[i].shape[-2:], 2),
                dtype=torch.long,
                device=class_logits.device,
            )
            preds[:, :, 0] = self.num_classes

            keep = classes[i].ne(class_logits.shape[-1] - 1) & (scores[i] > mask_thresh)
            if not keep.any():
                preds_list.append(preds)
                continue

            masks = mask_logits_list[i].sigmoid()
            segments = -torch.ones(
                *masks.shape[-2:],
                dtype=torch.long,
                device=class_logits.device,
            )

            mask_ids = (scores[i][keep][..., None, None] * masks[keep]).argmax(0)
            stuff_segment_ids, segment_id = {}, 0
            segment_and_class_ids = []

            for k, class_id in enumerate(classes[i][keep].tolist()):
                orig_mask = masks[keep][k] >= 0.5
                new_mask = mask_ids == k
                final_mask = orig_mask & new_mask

                orig_area = orig_mask.sum().item()
                new_area = new_mask.sum().item()
                final_area = final_mask.sum().item()
                if (
                    orig_area == 0
                    or new_area == 0
                    or final_area == 0
                    or new_area / orig_area < overlap_thresh
                ):
                    continue

                if class_id in stuff_classes:
                    if class_id in stuff_segment_ids:
                        segments[final_mask] = stuff_segment_ids[class_id]
                        continue
                    else:
                        stuff_segment_ids[class_id] = segment_id

                segments[final_mask] = segment_id
                segment_and_class_ids.append((segment_id, class_id))

                segment_id += 1

            for segment_id, class_id in segment_and_class_ids:
                segment_mask = segments == segment_id
                preds[:, :, 0] = torch.where(segment_mask, class_id, preds[:, :, 0])
                preds[:, :, 1] = torch.where(segment_mask, segment_id, preds[:, :, 1])

            preds_list.append(preds)

        return preds_list

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_panoptic(targets: list[dict]):
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = -torch.ones(
                (*target["masks"].shape[-2:], 2),
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[:, :, 0] = torch.where(
                    mask, target["labels"][i], per_pixel_target[:, :, 0]
                )

                per_pixel_target[:, :, 1] = torch.where(
                    mask,
                    torch.tensor(i, device=target["masks"].device),
                    per_pixel_target[:, :, 1],
                )

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            k.replace("._orig_mod", ""): v for k, v in checkpoint["state_dict"].items()
        }

    @torch.compiler.disable
    def plot_panoptic(
        self,
        img,
        targets,
        mask_logits,
        class_logits,
        log_prefix,
        block_idx,
        batch_idx,
    ):
        """
        为panoptic segmentation定制的可视化函数
        
        Args:
            img: 输入图像 (C, H, W)
            targets: ground truth targets (dict with masks and labels)
            mask_logits: 模型输出的mask logits (num_queries, H, W)
            class_logits: 模型输出的class logits (num_queries, num_classes+1)
            log_prefix: 日志前缀
            block_idx: block索引
            batch_idx: batch索引
        """
        fig, axes = plt.subplots(1, 3, figsize=[18, 6], sharex=True, sharey=True)
        
        # 1. 显示原始图像
        axes[0].imshow(img.cpu().numpy().transpose(1, 2, 0))
        axes[0].set_title("原始图像", fontsize=14)
        axes[0].axis("off")
        
        # 2. 生成GT panoptic分割图
        gt_panoptic = self._create_panoptic_visualization_gt(targets)
        axes[1].imshow(gt_panoptic)
        axes[1].set_title("Ground Truth", fontsize=14)
        axes[1].axis("off")
        
        # 3. 生成预测panoptic分割图
        pred_panoptic = self._create_panoptic_visualization_pred(
            mask_logits, class_logits, img.shape[-2:]
        )
        axes[2].imshow(pred_panoptic)
        axes[2].set_title("预测结果", fontsize=14)
        axes[2].axis("off")
        
        # 保存并上传到wandb
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, facecolor="white", dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        block_postfix = self.block_postfix(block_idx)
        name = f"{log_prefix}_panoptic_{batch_idx}{block_postfix}"
        self.trainer.logger.experiment.log({name: [wandb.Image(Image.open(buf))]})

    @torch.compiler.disable
    def _create_panoptic_visualization_gt(self, targets):
        """
        创建Ground Truth的panoptic可视化图像
        """
        H, W = targets["masks"].shape[-2:]
        panoptic_img = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 为每个mask分配不同颜色
        np.random.seed(42)  # 固定随机种子以确保颜色一致性
        
        for i, (mask, label) in enumerate(zip(targets["masks"], targets["labels"])):
            if mask.sum() == 0:  # 跳过空mask
                continue
            
            # 生成颜色：thing类用鲜艳颜色，stuff类用较淡颜色
            is_thing = label.item() not in self.stuff_classes
            if is_thing:
                # thing类用鲜艳的随机颜色
                color = (np.random.rand(3) * 255).astype(np.uint8)
            else:
                # stuff类用较淡的颜色
                base_color = np.random.rand(3)
                color = ((base_color * 0.7 + 0.3) * 255).astype(np.uint8)
            
            # 应用mask
            mask_np = mask.cpu().numpy().astype(bool)
            panoptic_img[mask_np] = color
            
            # 在mask上添加轮廓
            mask_uint8 = mask_np.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 绘制轮廓
            cv2.drawContours(panoptic_img, contours, -1, (255, 255, 255), 2)
            
            # 添加类别标签
            if len(contours) > 0:
                # 找到轮廓的中心点
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 确保坐标在图像范围内
                    cx = max(20, min(W-20, cx))
                    cy = max(20, min(H-20, cy))
                    
                    # 使用cv2绘制文本
                    label_text = str(label.item())
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    # 获取文本大小
                    text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
                    
                    # 绘制黑色背景矩形
                    cv2.rectangle(panoptic_img, 
                                (cx - text_size[0]//2 - 5, cy - text_size[1]//2 - 5),
                                (cx + text_size[0]//2 + 5, cy + text_size[1]//2 + 5),
                                (0, 0, 0), -1)
                    
                    # 绘制白色文本
                    cv2.putText(panoptic_img, label_text, 
                              (cx - text_size[0]//2, cy + text_size[1]//2), 
                              font, font_scale, (255, 255, 255), thickness)
        
        return panoptic_img.astype(np.float32) / 255.0

    @torch.compiler.disable  
    def _create_panoptic_visualization_pred(self, mask_logits, class_logits, img_size):
        """
        创建预测结果的panoptic可视化图像
        """
        H, W = img_size
        panoptic_img = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 获取预测结果
        scores, classes = class_logits.softmax(dim=-1).max(-1)
        
        # 首先将所有mask_logits插值到目标尺寸
        mask_logits_resized = torch.nn.functional.interpolate(
            mask_logits.unsqueeze(0),  # 添加batch维度
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )[0]  # 移除batch维度
        
        mask_probs = mask_logits_resized.sigmoid()
        
        # 降低阈值用于调试过拟合
        debug_thresh = min(0.01, self.mask_thresh)
        keep = classes.ne(class_logits.shape[-1] - 1) & (scores > debug_thresh)
        if not keep.any():
            return panoptic_img.astype(np.float32) / 255.0
        
        scores = scores[keep]
        classes = classes[keep]
        mask_probs = mask_probs[keep]
        
        # 根据置信度排序
        sorted_indices = torch.argsort(scores, descending=True)
        
        np.random.seed(42)
        
        for idx in sorted_indices:
            i = idx.item()
            mask = mask_probs[i] > 0.05
            if mask.sum() == 0:
                continue
            
            class_id = classes[i].item()
            
            # 生成颜色
            is_thing = class_id not in self.stuff_classes
            if is_thing:
                color = (np.random.rand(3) * 255).astype(np.uint8)
            else:
                base_color = np.random.rand(3)
                color = ((base_color * 0.7 + 0.3) * 255).astype(np.uint8)
            
            mask_np = mask.cpu().numpy().astype(bool)
            panoptic_img[mask_np] = color
        
        return panoptic_img.astype(np.float32) / 255.0

    @torch.compiler.disable
    def inference_and_visualize(self, dataloader=None, num_samples=4):
        """
        对validation set进行推理并可视化
        
        Args:
            dataloader: 验证数据加载器，如果为None则使用trainer的验证数据加载器
            num_samples: 要可视化的样本数量，默认为4
        """
        self.eval()
        
        # 获取数据加载器
        if dataloader is None:
            if self.trainer and hasattr(self.trainer, 'val_dataloaders'):
                dataloader = self.trainer.val_dataloaders
                if isinstance(dataloader, list):
                    dataloader = dataloader[0]  # 取第一个验证数据加载器
            else:
                raise ValueError("No dataloader provided and no trainer validation dataloader found")
        
        # 获取一个batch的数据
        with torch.no_grad():
            batch_iter = iter(dataloader)
            batch = next(batch_iter)
            imgs, targets = batch
            
            # 确保数据在正确的设备上
            if isinstance(imgs, torch.Tensor):
                imgs = imgs.to(self.device)
            elif isinstance(imgs, list):
                imgs = [img.to(self.device) for img in imgs]
            
            if isinstance(targets, list):
                for target in targets:
                    if isinstance(target, dict):
                        for key, value in target.items():
                            if isinstance(value, torch.Tensor):
                                target[key] = value.to(self.device)
            
            # 前向推理
            mask_logits_per_block, class_logits_per_block = self(imgs)
            
            # 使用最后一个block的输出进行可视化
            final_mask_logits = mask_logits_per_block[-1]
            final_class_logits = class_logits_per_block[-1]
            
            # 限制可视化的样本数量
            batch_size = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
            num_samples = min(num_samples, batch_size)
            
            print(f"开始可视化 {num_samples} 个样本...")
            
            # 对每个样本进行可视化
            for i in range(num_samples):
                if isinstance(imgs, torch.Tensor):
                    img = imgs[i]
                else:
                    img = imgs[i]
                
                target = targets[i] if isinstance(targets, list) else targets
                mask_logits = final_mask_logits[i]
                class_logits = final_class_logits[i]
                
                # 调用可视化函数
                self.plot_panoptic(
                    img=img,
                    targets=target,
                    mask_logits=mask_logits,
                    class_logits=class_logits,
                    log_prefix="inference",
                    block_idx=len(mask_logits_per_block) - 1,  # 最后一个block
                    batch_idx=i,
                )
            
            print(f"可视化完成！已上传 {num_samples} 个样本到 wandb")
            
            # 返回推理结果以便进一步分析
            return {
                'mask_logits_per_block': mask_logits_per_block,
                'class_logits_per_block': class_logits_per_block,
                'imgs': imgs,
                'targets': targets
            }

    @torch.compiler.disable 
    def run_inference_on_validation(self, max_batches=1, samples_per_batch=4):
        """
        在验证集上运行推理并可视化
        
        Args:
            max_batches: 最大处理的batch数量
            samples_per_batch: 每个batch中可视化的样本数量
        """
        if not self.trainer or not hasattr(self.trainer, 'val_dataloaders'):
            raise ValueError("Trainer or validation dataloader not found")
        
        self.eval()
        val_dataloader = self.trainer.val_dataloaders
        if isinstance(val_dataloader, list):
            val_dataloader = val_dataloader[0]
        
        print(f"开始在验证集上进行推理，将处理 {max_batches} 个batch，每个batch可视化 {samples_per_batch} 个样本")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx >= max_batches:
                    break
                
                print(f"处理第 {batch_idx + 1} 个batch...")
                
                imgs, targets = batch
                
                # 确保数据在正确的设备上
                if isinstance(imgs, torch.Tensor):
                    imgs = imgs.to(self.device)
                elif isinstance(imgs, list):
                    imgs = [img.to(self.device) for img in imgs]
                
                if isinstance(targets, list):
                    for target in targets:
                        if isinstance(target, dict):
                            for key, value in target.items():
                                if isinstance(value, torch.Tensor):
                                    target[key] = value.to(self.device)
                
                # 前向推理
                mask_logits_per_block, class_logits_per_block = self(imgs)
                
                # 使用最后一个block的输出进行可视化
                final_mask_logits = mask_logits_per_block[-1]
                final_class_logits = class_logits_per_block[-1]
                
                # 限制可视化的样本数量
                batch_size = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
                num_samples = min(samples_per_batch, batch_size)
                
                # 对当前batch的样本进行可视化
                for i in range(num_samples):
                    if isinstance(imgs, torch.Tensor):
                        img = imgs[i]
                    else:
                        img = imgs[i]
                    
                    target = targets[i] if isinstance(targets, list) else targets
                    mask_logits = final_mask_logits[i]
                    class_logits = final_class_logits[i]
                    
                    # 调用可视化函数
                    self.plot_panoptic(
                        img=img,
                        targets=target,
                        mask_logits=mask_logits,
                        class_logits=class_logits,
                        log_prefix=f"val_inference_batch_{batch_idx}",
                        block_idx=len(mask_logits_per_block) - 1,
                        batch_idx=i,
                    )
        
        print("验证集推理和可视化完成！")
