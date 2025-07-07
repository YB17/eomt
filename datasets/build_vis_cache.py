#!/usr/bin/env python3
# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

"""
build_vis_cache.py - 构建视觉嵌入缓存（PyTorch Lightning版本）

为COCO Panoptic数据集中的每个图像生成SigLIP视觉嵌入，输出：
- cache/vis_embed/<imageID>.npz                  (每个图像的所有segment embeddings)
- cache/vis_embed/image_index.tsv                (图像索引文件)

每个NPZ文件包含：
- 'segment_ids': 该图像的所有segment ID数组
- 'embeddings': 对应的512维嵌入向量数组 (fp16)
- 'labels': 对应的类别标签数组

TSV索引文件格式:
    <image_id> <tab> <feat_file> <tab> <num_segments>
"""

import os
import sys
import json
import argparse
import tqdm
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, SequentialSampler

import torchvision.transforms.v2.functional as F
import lightning
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from eomt.novic.embedders import Embedder
from eomt.datasets.coco_panoptic_directory import COCOPanopticDirectory
from eomt.datasets.lightning_data_module import LightningDataModule


class VisualEmbeddingInferenceModule(lightning.LightningModule):
    """用于推理的Lightning模块"""
    
    def __init__(
        self,
        embedder_spec: str = "openclip:timm/ViT-B-16-SigLIP",
        bg_alpha: float = 0.3,
        pad_ratio: float = 0.1,
        cache_dir: str = "/tmp/vis_embed",
    ):
        super().__init__()
        self.embedder_spec = embedder_spec
        self.bg_alpha = bg_alpha
        self.pad_ratio = pad_ratio
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.total_images_processed = 0
        self.total_segments_processed = 0
        self.total_images_skipped = 0
        
    def setup(self, stage: str):
        """设置嵌入器"""
        if stage == "predict":
            print(f"[Rank {self.global_rank}] 加载SigLIP模型: {self.embedder_spec}")
            self.embedder = Embedder.create(
                spec=self.embedder_spec,
                amp=True,
                device=self.device,
                inference_batch_size=64,
                image_batch_size=32,
                load_model=True,
                compile_model=False,
            )
            self.image_transform = self.embedder.get_image_transform()
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Any]:
        """预测步骤：处理一个batch的数据"""
        batch_results = []
        
        # 提取批次信息
        batch_info = self.extract_image_and_segment_ids(batch)
        
        for image_id, image_tensor, masks, labels, target_ids in batch_info:
            if not masks or not target_ids:
                continue
            
            # 检查文件是否已存在
            feat_file = f"{image_id}.npz"
            out_path = self.cache_dir / feat_file
            
            if out_path.exists():
                self.total_images_skipped += 1
                # 读取已存在文件以获取segment数量用于索引
                try:
                    existing_data = np.load(out_path)
                    num_segments = len(existing_data['segment_ids'])
                    batch_results.append({
                        'image_id': image_id,
                        'feat_file': feat_file,
                        'num_segments': num_segments,
                        'status': 'skipped'
                    })
                except:
                    pass
                continue
            
            # 为所有mask准备图像
            processed_images, valid_indices, valid_segment_ids, valid_labels = self.prep_mask_image(
                image_tensor, masks, labels, target_ids
            )
            
            if not processed_images:
                continue
            
            # 批量处理图像
            image_tensors = []
            for pil_img in processed_images:
                tensor = self.image_transform(pil_img)
                image_tensors.append(tensor)
            
            if image_tensors:
                # 批量推理
                batch_tensor = torch.stack(image_tensors)
                
                with self.embedder.inference_mode():
                    embeddings = self.embedder.inference_image(batch_tensor)
                
                # 保存该图像的所有embeddings
                if self.save_image_embeddings(
                    image_id, valid_segment_ids, embeddings, valid_labels
                ):
                    self.total_images_processed += 1
                    self.total_segments_processed += len(embeddings)
                    
                    batch_results.append({
                        'image_id': image_id,
                        'feat_file': feat_file,
                        'num_segments': len(embeddings),
                        'status': 'processed'
                    })
        
        return {
            'batch_idx': batch_idx,
            'results': batch_results,
            'rank': self.global_rank
        }
    
    def extract_image_and_segment_ids(self, batch):
        """从数据加载器批次中提取图像ID和segment ID信息"""
        images, targets = batch
        batch_info = []
        
        for i in range(len(images)):
            image = images[i]
            target = targets[i]
            
            # 直接从target中获取图像ID
            image_id = target.get('image_id', f'image_{i}')
            
            # DirectoryDataset返回的target是一个字典
            if isinstance(target, dict):
                # 从字典中提取数据
                masks_tensor = target.get('masks', torch.empty(0, 0, 0))
                labels_tensor = target.get('labels', torch.empty(0))
                target_ids_tensor = target.get('target_ids', torch.empty(0))
                
                # 转换为列表形式
                masks = []
                if isinstance(masks_tensor, torch.Tensor) and masks_tensor.numel() > 0:
                    if masks_tensor.dim() >= 3:
                        # masks shape: [N, H, W]
                        masks = [masks_tensor[j] for j in range(masks_tensor.shape[0])]
                    else:
                        # 如果只有一个mask
                        masks = [masks_tensor]
                
                labels = []
                if isinstance(labels_tensor, torch.Tensor) and labels_tensor.numel() > 0:
                    labels = labels_tensor.tolist()
                
                target_ids = []
                if isinstance(target_ids_tensor, torch.Tensor) and target_ids_tensor.numel() > 0:
                    target_ids = target_ids_tensor.tolist()
                    # 确保target_ids是整数列表
                    target_ids = [int(tid) if isinstance(tid, (int, float)) else tid for tid in target_ids]
                
            else:
                print(f"[Rank {self.global_rank}] 无法解析target格式: {type(target)}")
                continue
            
            # 验证数据的有效性
            if not masks or not target_ids:
                continue
                
            if len(masks) != len(target_ids):
                # 取较小的长度
                min_len = min(len(masks), len(target_ids))
                masks = masks[:min_len]
                target_ids = target_ids[:min_len]
                if labels:
                    labels = labels[:min_len]
                
            batch_info.append((image_id, image, masks, labels, target_ids))
        
        return batch_info
    
    def prep_mask_image(self, image: torch.Tensor, masks: list, labels: list, target_ids: list):
        """为每个mask准备裁剪和混合后的图像"""
        if not masks:
            return [], [], [], []
        
        # 将图像转换为numpy数组 [H, W, C]，范围 [0, 1]
        if image.dim() == 3:
            rgb = image.permute(1, 2, 0).cpu().numpy()
        else:
            rgb = image.cpu().numpy()
        
        # 确保是RGB格式且在[0,1]范围内
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        processed_images = []
        valid_indices = []
        valid_segment_ids = []
        valid_labels = []
        
        for i, mask in enumerate(masks):
            try:
                # 转换mask为numpy数组
                mask_np = mask.cpu().numpy().astype(bool)
                
                # 检查mask是否有效
                if not mask_np.any():
                    continue
                    
                # 应用背景混合
                alpha = np.where(mask_np, 1.0, self.bg_alpha).astype(np.float32)
                blended = rgb * alpha[..., None] + (1 - alpha[..., None]) * 0.5

                # 获取mask的边界框
                ys, xs = np.where(mask_np)
                if len(ys) == 0 or len(xs) == 0:
                    continue
                    
                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()
                
                # 添加填充
                pad_y = int(self.pad_ratio * (y1 - y0 + 1))
                pad_x = int(self.pad_ratio * (x1 - x0 + 1))
                y0 = max(0, y0 - pad_y)
                y1 = min(mask_np.shape[0], y1 + pad_y + 1)
                x0 = max(0, x0 - pad_x)
                x1 = min(mask_np.shape[1], x1 + pad_x + 1)
                
                # 裁剪图像
                crop = blended[y0:y1, x0:x1]

                # 转换为PIL图像
                crop_pil = Image.fromarray((crop * 255).astype(np.uint8))
                processed_images.append(crop_pil)
                valid_indices.append(i)
                
                # 保存对应的segment_id和label
                if i < len(target_ids):
                    valid_segment_ids.append(target_ids[i])
                else:
                    valid_segment_ids.append(0)  # 默认值
                    
                if i < len(labels):
                    valid_labels.append(labels[i])
                else:
                    valid_labels.append(0)  # 默认值
                
            except Exception as e:
                print(f"[Rank {self.global_rank}] 处理mask {i}时出错: {e}")
                continue

        return processed_images, valid_indices, valid_segment_ids, valid_labels
    
    def save_image_embeddings(self, image_id: str, segment_ids: list, 
                             embeddings, labels: list) -> bool:
        """将一个图像的所有segment embeddings保存到NPZ文件"""
        # 检查输入有效性
        if not segment_ids:
            return False
        
        # 处理embeddings的不同类型
        if isinstance(embeddings, torch.Tensor):
            if embeddings.numel() == 0:  # 空张量
                return False
            embeddings_array = embeddings.cpu().numpy().astype(np.float16)
        elif hasattr(embeddings, '__len__'):  # 列表或类似序列
            if len(embeddings) == 0:
                return False
            embeddings_array = np.stack([emb.cpu().numpy().astype(np.float16) for emb in embeddings])
        else:
            return False
        
        # 转换为numpy数组
        segment_ids_array = np.array(segment_ids, dtype=np.int32)
        labels_array = np.array(labels, dtype=np.int32) if labels else np.zeros(len(segment_ids), dtype=np.int32)
        
        # 保存文件
        feat_file = f"{image_id}.npz"
        out_path = self.cache_dir / feat_file
        
        np.savez_compressed(
            out_path,
            segment_ids=segment_ids_array,
            embeddings=embeddings_array,
            labels=labels_array
        )
        
        return True


class EmbeddingResultWriter(BasePredictionWriter):
    """处理预测结果的写入器"""
    
    def __init__(self, cache_dir: str, write_interval: str = "batch"):
        super().__init__(write_interval)
        self.cache_dir = Path(cache_dir)
        self.index_file = self.cache_dir / "image_index.tsv"
        self.results_buffer = []
        
        # 在主进程初始化索引文件
        if write_interval == "batch":
            self._init_index_file()
    
    def _init_index_file(self):
        """初始化索引文件"""
        with open(self.index_file, "w", encoding='utf-8') as f:
            f.write("image_id\tfeat_file\tnum_segments\n")
    
    def write_on_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: lightning.LightningModule, 
        prediction: Any, 
        batch_indices: List[int], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int
    ):
        """在每个batch结束时写入结果"""
        if prediction and 'results' in prediction:
            self.results_buffer.extend(prediction['results'])
            
            # 每处理100个batch或在最后写入索引文件
            if len(self.results_buffer) >= 100 or batch_idx % 100 == 0:
                self._write_index_batch()
    
    def write_on_epoch_end(
        self, 
        trainer: Trainer, 
        pl_module: lightning.LightningModule, 
        predictions: List[Any], 
        batch_indices: List[List[int]]
    ):
        """在epoch结束时写入剩余结果"""
        self._write_index_batch()
    
    def _write_index_batch(self):
        """批量写入索引"""
        if not self.results_buffer:
            return
            
        with open(self.index_file, "a", encoding='utf-8') as f:
            for result in self.results_buffer:
                if result['status'] in ['processed', 'skipped']:
                    f.write(f"{result['image_id']}\t{result['feat_file']}\t{result['num_segments']}\n")
        
        self.results_buffer.clear()


class VisCacheDataModule(LightningDataModule):
    """用于构建视觉缓存的数据模块"""
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle: bool = False,
        **kwargs
    ):
        super().__init__(
            path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=(640, 640),
            num_classes=133,  # COCO Panoptic classes
            check_empty_targets=False,
            **kwargs
        )
        
        # COCO Panoptic的stuff类别 (80-132)
        self.stuff_classes = list(range(80, 133))
        self.shuffle = shuffle
    
    def setup(self, stage: str):
        """设置数据集"""
        if stage == "predict":
            self.predict_dataset = COCOPanopticDirectory(
                path=self.path,
                stuff_classes=self.stuff_classes,
                batch_size=self.dataloader_kwargs["batch_size"],
                num_workers=self.dataloader_kwargs["num_workers"],
                img_size=self.img_size,
                check_empty_targets=self.check_empty_targets,
                color_jitter_enabled=False,
                scale_range=(1.0, 1.0),  # 不进行缩放变换，保持原始尺寸
            )
            self.predict_dataset.setup("fit")
    
    def predict_dataloader(self):
        """返回预测数据加载器（关闭shuffle）"""
        return DataLoader(
            self.predict_dataset.train_dataset,
            batch_size=self.dataloader_kwargs["batch_size"],
            shuffle=self.shuffle,
            num_workers=self.dataloader_kwargs["num_workers"],
            pin_memory=self.dataloader_kwargs.get("pin_memory", True),
            persistent_workers=self.dataloader_kwargs.get("persistent_workers", True),
            collate_fn=self.train_collate,
            drop_last=False,
        )

'''
python -m eomt.datasets.build_vis_cache \
  --data_path /home/host_ssd/coconut_dataset/coco \
  --cache_dir /home/host_ssd/coconut_dataset/vis_embed \
  --batch_size 32 \
  --num_workers 32 \
  --devices 8 \
  --strategy ddp
'''

def parse_args():
    """解析命令行参数"""
    ap = argparse.ArgumentParser(description="构建COCO Panoptic视觉嵌入缓存（Lightning版本）")
    ap.add_argument("--data_path", 
                    default="/home/host_ssd/coconut_dataset/coco",
                    help="COCO数据集根目录路径")
    ap.add_argument("--cache_dir", 
                    default="/home/host_ssd/coconut_dataset/vis_embed",
                    help="缓存输出目录")
    ap.add_argument("--embedder_spec", 
                    default="openclip:timm/ViT-B-16-SigLIP",
                    help="SigLIP模型规格")
    ap.add_argument("--batch_size", 
                    type=int, 
                    default=8,
                    help="数据加载器批大小")
    ap.add_argument("--num_workers", 
                    type=int, 
                    default=8,
                    help="数据加载器工作进程数")
    ap.add_argument("--devices", 
                    type=int, 
                    default=1,
                    help="使用的GPU数量")
    ap.add_argument("--accelerator", 
                    default="gpu",
                    help="加速器类型")
    ap.add_argument("--strategy", 
                    default="auto",
                    help="分布式策略")
    ap.add_argument("--bg_alpha", 
                    type=float, 
                    default=0.3,
                    help="背景混合透明度")
    ap.add_argument("--pad_ratio", 
                    type=float, 
                    default=0.1,
                    help="裁剪时的填充比例")
    ap.add_argument("--shuffle", 
                    action="store_true",
                    help="是否对数据进行shuffle（默认关闭以保证确定性顺序）")
    return ap.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置路径
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"缓存目录: {cache_dir}")
    print(f"数据路径: {args.data_path}")
    print(f"使用设备: {args.devices} x {args.accelerator}")
    print(f"SigLIP模型: {args.embedder_spec}")
    
    # 创建Lightning模块
    model = VisualEmbeddingInferenceModule(
        embedder_spec=args.embedder_spec,
        bg_alpha=args.bg_alpha,
        pad_ratio=args.pad_ratio,
        cache_dir=str(cache_dir),
    )
    
    # 创建数据模块
    data_module = VisCacheDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
    )
    
    # 创建结果写入器
    writer = EmbeddingResultWriter(cache_dir=str(cache_dir))
    
    # 创建Trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision="16-mixed",
        callbacks=[writer],
        enable_model_summary=False,
        enable_progress_bar=True,
        logger=False,  # 禁用日志记录器
    )
    
    # 开始推理
    print("🚀 开始构建视觉嵌入缓存...")
    trainer.predict(model, datamodule=data_module)
    
    # 输出统计信息
    print(f"\n✅ 完成！")
    print(f"缓存目录: {cache_dir}")
    print(f"索引文件: {cache_dir / 'image_index.tsv'}")
    
    # 输出使用示例
    print(f"\n💡 使用示例:")
    print(f"```python")
    print(f"import numpy as np")
    print(f"")
    print(f"# 加载某个图像的embeddings")
    print(f"data = np.load('{cache_dir}/{{image_id}}.npz')")
    print(f"segment_ids = data['segment_ids']      # shape: (N,)")
    print(f"embeddings = data['embeddings']        # shape: (N, 512)")
    print(f"labels = data['labels']                # shape: (N,)")
    print(f"```")


if __name__ == "__main__":
    main()
