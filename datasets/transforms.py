# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from Detectron2 by Facebook, Inc. and its affiliates,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import wrap, TVTensor
from torch import nn, Tensor
from typing import Any, Union


class Transforms(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        color_jitter_enabled: bool,
        scale_range: tuple[float, float],
        max_brightness_delta: int = 32,
        max_contrast_factor: float = 0.5,
        saturation_factor: float = 0.5,
        max_hue_delta: int = 18,
    ):
        super().__init__()

        self.img_size = img_size
        self.color_jitter_enabled = color_jitter_enabled
        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0

        self.random_horizontal_flip = T.RandomHorizontalFlip()
        self.scale_jitter = T.ScaleJitter(target_size=img_size, scale_range=scale_range)
        self.random_crop = T.RandomCrop(img_size)

    def _random_factor(self, factor: float, center: float = 1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def _brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(
                img, self._random_factor(self.max_brightness_factor)
            )

        return img

    def _contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(img, self._random_factor(self.max_contrast_factor))

        return img

    def _saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img, self._random_factor(self.max_saturation_factor)
            )

        if torch.rand(()) < 0.5:
            img = F.adjust_hue(img, self._random_factor(self.max_hue_delta, center=0.0))

        return img

    def color_jitter(self, img):
        if not self.color_jitter_enabled:
            return img

        img = self._brightness(img)

        if torch.rand(()) < 0.5:
            img = self._contrast(img)
            img = self._saturation_and_hue(img)
        else:
            img = self._saturation_and_hue(img)
            img = self._contrast(img)

        return img

    def pad(
        self, img: Tensor, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding)
        target["masks"] = F.pad(target["masks"], padding)

        return img, target

    def _filter(self, target: dict[str, Union[Tensor, TVTensor]], keep: Tensor) -> dict:
        """过滤target字典，跳过非tensor字段"""
        filtered_target = {}
        
        for k, v in target.items():
            if isinstance(v, (Tensor, TVTensor)):
                # 只对tensor/TVTensor类型的值应用索引
                filtered_target[k] = wrap(v[keep], like=v)
            else:
                # 对于其他类型（如字符串image_id），直接保留
                filtered_target[k] = v
        
        return filtered_target

    def forward(
        self, img: Tensor, target: dict[str, Union[Tensor, TVTensor]]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        img_orig, target_orig = img, target

        target = self._filter(target, ~target["is_crowd"])

        img = self.color_jitter(img)
        img, target = self.random_horizontal_flip(img, target)
        img, target = self.scale_jitter(img, target)
        img, target = self.pad(img, target)
        img, target = self.random_crop(img, target)

        valid = target["masks"].flatten(1).any(1)
        if not valid.any():
            return self(img_orig, target_orig)

        target = self._filter(target, valid)

        return img, target

class MinimalTransforms(nn.Module):
    """
    最小数据变换类 - 用于过拟合小数据集
    只保留必要的尺寸调整和填充操作，关闭所有数据增强
    """
    def __init__(
        self,
        img_size: tuple[int, int],
    ):
        super().__init__()
        self.img_size = img_size

    def pad(
        self, img: Tensor, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        """只进行必要的填充操作以匹配目标尺寸"""
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding)
        target["masks"] = F.pad(target["masks"], padding)

        return img, target

    def resize_to_target_size(
        self, img: Tensor, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        """将图像和目标调整到目标尺寸"""
        if img.shape[-2:] != self.img_size:
            # 使用双线性插值调整图像尺寸
            img = F.resize(img, self.img_size, interpolation=F.InterpolationMode.BILINEAR)
            # 使用最近邻插值调整mask尺寸以保持标签的准确性
            target["masks"] = F.resize(
                target["masks"], 
                self.img_size, 
                interpolation=F.InterpolationMode.NEAREST
            )
        
        return img, target

    def _filter(self, target: dict[str, Union[Tensor, TVTensor]], keep: Tensor) -> dict:
        """过滤target字典，跳过非tensor字段"""
        filtered_target = {}
        
        for k, v in target.items():
            if isinstance(v, (Tensor, TVTensor)):
                # 只对tensor/TVTensor类型的值应用索引
                filtered_target[k] = wrap(v[keep], like=v)
            else:
                # 对于其他类型（如字符串image_id），直接保留
                filtered_target[k] = v
        
        return filtered_target

    def forward(
        self, img: Tensor, target: dict[str, Union[Tensor, TVTensor]]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        """
        前向传播 - 只进行必要的尺寸调整
        不进行任何随机数据增强操作
        """
        # 过滤掉crowd标注
        target = self._filter(target, ~target["is_crowd"])
        
        # 调整到目标尺寸
        img, target = self.resize_to_target_size(img, target)
        
        # 如果需要的话进行填充
        img, target = self.pad(img, target)
        
        # 检查是否有有效的mask
        valid = target["masks"].flatten(1).any(1)
        if not valid.any():
            # 如果没有有效mask，返回原始数据（这种情况在小数据集中应该很少见）
            return img, target

        target = self._filter(target, valid)

        return img, target
