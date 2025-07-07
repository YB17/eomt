#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单读取mask annotation PNG图片并显示shape
"""

import cv2
import numpy as np
from pathlib import Path

def read_and_show_mask_shape(png_path):
    """
    读取PNG mask文件并显示shape信息
    
    Args:
        png_path: PNG文件路径
    """
    print(f"正在读取: {png_path}")
    
    # 读取PNG文件 (保持原始格式)
    mask = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    
    if mask is None:
        print(f"❌ 无法读取文件: {png_path}")
        return None
    
    # 显示基本信息
    print(f"✅ 成功读取!")
    print(f"📐 Shape: {mask.shape}")
    print(f"🔢 数据类型: {mask.dtype}")
    print(f"📊 数值范围: {mask.min()} ~ {mask.max()}")
    
    # 如果是多通道图片
    if len(mask.shape) == 3:
        print(f"🎨 通道数: {mask.shape[2]}")
        print(f"🖼️  图片尺寸: {mask.shape[1]} x {mask.shape[0]} (宽x高)")
        
        # 显示每个通道的信息
        for i in range(mask.shape[2]):
            channel_min = mask[:, :, i].min()
            channel_max = mask[:, :, i].max()
            print(f"   通道{i}: {channel_min} ~ {channel_max}")
    
    elif len(mask.shape) == 2:
        print(f"🖼️  图片尺寸: {mask.shape[1]} x {mask.shape[0]} (宽x高)")
        print(f"⚫ 单通道灰度图")
    
    # 如果看起来像COCO panoptic格式 (3通道)
    if len(mask.shape) == 3 and mask.shape[2] >= 3:
        print(f"\n🔍 检测到可能的Panoptic格式，正在解析...")
        # 按照COCO panoptic格式解析: R + G*256 + B*256^2
        # 注意OpenCV读取是BGR格式
        segment_ids = mask[:, :, 2] + mask[:, :, 1] * 256 + mask[:, :, 0] * 256**2
        unique_segments = np.unique(segment_ids)
        
        print(f"🎯 解析结果:")
        print(f"   Segment IDs数量: {len(unique_segments)}")
        print(f"   ID范围: {unique_segments.min()} ~ {unique_segments.max()}")
        print(f"   前5个IDs: {unique_segments[:5]}")
    
    print("-" * 50)
    return mask

def main():
    """主函数 - 可以处理单个文件或批量文件"""
    
    # 方式1: 处理单个文件
    png_file = "/home/host_ssd/coconut_dataset/coco/panoptic_train2017/000000177680.png"  # 🔧 在这里修改您的文件路径
    
    if Path(png_file).exists():
        mask = read_and_show_mask_shape(png_file)
    else:
        print(f"⚠️  文件不存在: {png_file}")
        print("请修改上面的 png_file 变量为您的实际文件路径")
        
        # 方式2: 如果有多个文件，可以批量处理
        print("\n💡 如果您有多个文件要处理，可以取消下面代码的注释:")
        print("""
        # 批量处理示例
        mask_folder = Path("path/to/your/masks")  # 修改为您的mask文件夹路径
        for png_file in mask_folder.glob("*.png"):
            read_and_show_mask_shape(png_file)
        """)

if __name__ == "__main__":
    main() 