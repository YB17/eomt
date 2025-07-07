#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的脚本用于读取mask annotation的PNG图片并显示shape
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def read_mask_annotation(png_path):
    """
    读取mask annotation的PNG图片
    
    Args:
        png_path: PNG文件路径
        
    Returns:
        mask: 读取的mask数组
    """
    # 使用OpenCV读取PNG图片 (支持多通道)
    mask = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    
    if mask is None:
        raise ValueError(f"无法读取图片: {png_path}")
    
    return mask

def parse_panoptic_mask(mask):
    """
    解析panoptic mask，按照COCO panoptic格式
    RGB通道编码: R + G*256 + B*256^2
    
    Args:
        mask: 3通道的mask图片 (H, W, 3)
        
    Returns:
        segment_ids: 解析后的segment ID数组 (H, W)
    """
    if len(mask.shape) == 3 and mask.shape[2] >= 3:
        # BGR格式 (OpenCV默认)，转换为RGB格式的segment ID
        segment_ids = mask[:, :, 2] + mask[:, :, 1] * 256 + mask[:, :, 0] * 256**2
    else:
        # 单通道或其他格式
        segment_ids = mask
    
    return segment_ids

def display_mask_info(mask, png_path):
    """
    显示mask的详细信息
    
    Args:
        mask: mask数组
        png_path: 文件路径
    """
    print(f"\n=== Mask信息: {Path(png_path).name} ===")
    print(f"Shape: {mask.shape}")
    print(f"数据类型: {mask.dtype}")
    print(f"最小值: {mask.min()}")
    print(f"最大值: {mask.max()}")
    
    if len(mask.shape) == 3:
        print(f"通道数: {mask.shape[2]}")
        for i in range(mask.shape[2]):
            print(f"  通道{i}: min={mask[:,:,i].min()}, max={mask[:,:,i].max()}")
    
    # 如果是panoptic格式，解析segment IDs
    if len(mask.shape) == 3 and mask.shape[2] >= 3:
        segment_ids = parse_panoptic_mask(mask)
        unique_ids = np.unique(segment_ids)
        print(f"\nPanoptic解析:")
        print(f"  Segment IDs shape: {segment_ids.shape}")
        print(f"  唯一segment数量: {len(unique_ids)}")
        print(f"  Segment ID范围: {unique_ids.min()} - {unique_ids.max()}")
        print(f"  前10个segment IDs: {unique_ids[:10]}")

def visualize_mask(mask, save_path=None):
    """
    可视化mask
    
    Args:
        mask: mask数组
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(12, 8))
    
    if len(mask.shape) == 3:
        if mask.shape[2] >= 3:
            # 显示原始RGB mask
            plt.subplot(2, 2, 1)
            plt.imshow(mask[:,:,:3])
            plt.title("原始RGB Mask")
            plt.axis('off')
            
            # 显示解析后的segment IDs
            plt.subplot(2, 2, 2)
            segment_ids = parse_panoptic_mask(mask)
            plt.imshow(segment_ids, cmap='tab20')
            plt.title("Segment IDs")
            plt.colorbar()
            plt.axis('off')
            
            # 显示各个通道
            for i in range(min(3, mask.shape[2])):
                plt.subplot(2, 3, 4+i)
                plt.imshow(mask[:,:,i], cmap='gray')
                plt.title(f"通道 {i}")
                plt.axis('off')
        else:
            plt.imshow(mask, cmap='gray')
            plt.title("Mask")
            plt.colorbar()
            plt.axis('off')
    else:
        plt.imshow(mask, cmap='tab20')
        plt.title("Mask")
        plt.colorbar()
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果保存到: {save_path}")
    
    plt.show()

def main():
    """主函数示例"""
    # 示例用法
    png_path = "example_mask.png"  # 替换为您的PNG文件路径
    
    # 检查文件是否存在
    if not Path(png_path).exists():
        print(f"文件不存在: {png_path}")
        print("请将 'example_mask.png' 替换为您的实际PNG文件路径")
        return
    
    try:
        # 读取mask
        mask = read_mask_annotation(png_path)
        
        # 显示信息
        display_mask_info(mask, png_path)
        
        # 可视化 (可选)
        visualize_mask(mask, save_path="mask_visualization.png")
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main() 