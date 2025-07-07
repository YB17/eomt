# Visual Embedding Cache Builder

## 概述

`build_vis_cache.py` 是一个用于为COCO Panoptic数据集构建视觉嵌入缓存的工具。它使用novic支持的SigLIP模型（`openclip:timm/ViT-B-16-SigLIP`）为每个mask实例生成高质量的视觉嵌入向量。

## 主要特性

- ✅ **框架集成**: 完全集成到EoMT框架中，使用`COCOPanopticDirectory`数据加载器
- ✅ **高效模型**: 使用novic的`openclip:timm/ViT-B-16-SigLIP`模型生成512维嵌入向量  
- ✅ **批量处理**: 支持批量推理，提高处理效率
- ✅ **内存优化**: 使用fp16格式存储，节省存储空间
- ✅ **断点续传**: 自动跳过已存在的嵌入文件
- ✅ **详细日志**: 提供详细的处理进度和错误信息

## 输出文件

### 1. 嵌入向量文件
- **路径**: `cache/vis_embed/<image_id>_<segment_id>.npy`
- **格式**: NumPy数组，fp16精度，512维向量
- **命名**: 图像ID + 下划线 + segment ID

### 2. 索引文件  
- **路径**: `cache/vis_embed/inst2feat.tsv`
- **格式**: 制表符分隔的文本文件
- **表头**: `image_id	segment_id	feat_file`
- **示例**:
  ```
  image_id	segment_id	feat_file
  000000397133	15466	000000397133_15466.npy
  000000037777	23456	000000037777_23456.npy
  ```

## 使用方法

### 基本用法

```bash
cd eomt
python datasets/build_vis_cache.py --data_path /path/to/coco/dataset
```

### 完整参数说明

```bash
python datasets/build_vis_cache.py \
    --data_path /home/host_ssd/coconut_dataset/coco \
    --cache_dir cache/vis_embed \
    --embedder_spec openclip:timm/ViT-B-16-SigLIP \
    --batch_size 8 \
    --num_workers 4 \
    --device cuda \
    --bg_alpha 0.3 \
    --pad_ratio 0.1
```

### 参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `/home/host_ssd/coconut_dataset/coco` | COCO数据集根目录路径 |
| `--cache_dir` | `cache/vis_embed` | 缓存输出目录 |
| `--embedder_spec` | `openclip:timm/ViT-B-16-SigLIP` | SigLIP模型规格 |
| `--batch_size` | `8` | 数据加载器批大小 |
| `--num_workers` | `4` | 数据加载器工作进程数 |
| `--device` | `cuda` | 推理设备 |
| `--bg_alpha` | `0.3` | 背景混合透明度 |
| `--pad_ratio` | `0.1` | 裁剪时的填充比例 |

## 环境要求

### 必要依赖

```bash
# 基础依赖
pip install torch torchvision tqdm numpy pillow

# novic依赖  
pip install open_clip_torch transformers

# EoMT框架依赖
pip install lightning
```

### 目录结构

确保您的COCO数据集具有以下结构：
```
coco_dataset/
├── train2017/           # 训练图像
├── val2017/             # 验证图像  
├── panoptic_train2017/  # 训练全景分割标注
├── panoptic_val2017/    # 验证全景分割标注
└── annotations/         # JSON标注文件
    ├── panoptic_train2017.json
    └── panoptic_val2017.json
```

## 处理流程

### 1. 数据加载
- 使用`COCOPanopticDirectory`创建数据加载器
- 支持COCO Panoptic格式的RGB编码: `segment_id = R + G*256 + B*256^2`
- 自动过滤无效的mask和target

### 2. 图像预处理
- **背景混合**: 使用`bg_alpha`参数控制背景透明度
- **边界框裁剪**: 基于mask计算边界框并添加填充
- **尺寸标准化**: 使用SigLIP的图像预处理管道

### 3. 嵌入生成
- **批量推理**: 同时处理多个mask以提高效率
- **混合精度**: 使用AMP加速推理过程
- **内存管理**: 自动管理GPU内存，避免OOM

### 4. 结果保存
- **fp16存储**: 减少存储空间占用
- **文件命名**: 使用图像ID和segment ID确保唯一性
- **索引维护**: 同步更新TSV索引文件

## 性能优化

### GPU内存优化
- 调整`--batch_size`参数以适应您的GPU内存
- 使用`--device cpu`在CPU上运行（较慢但内存需求低）

### 处理速度优化  
- 增加`--num_workers`以提高数据加载速度
- 使用SSD存储以减少I/O瓶颈
- 确保CUDA版本兼容以启用GPU加速

### 存储优化
- 嵌入向量使用fp16格式，每个512维向量约1KB
- 预计存储需求：COCO训练集约 ~100GB

## 故障排除

### 常见错误

#### 1. 导入错误
```bash
ImportError: No module named 'embedders'
```
**解决方案**: 确保novic目录在Python路径中：
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/novic
```

#### 2. CUDA内存不足
```bash
RuntimeError: CUDA out of memory
```
**解决方案**: 
- 减小`--batch_size`参数
- 使用`--device cpu`
- 关闭其他GPU程序

#### 3. 数据路径错误
```bash
FileNotFoundError: [Errno 2] No such file or directory
```
**解决方案**: 
- 检查`--data_path`是否正确
- 确保COCO数据集目录结构完整

### 调试建议

1. **运行测试脚本**:
   ```bash
   python test_build_vis_cache.py
   ```

2. **使用小批量测试**:
   ```bash
   python datasets/build_vis_cache.py --batch_size 1 --num_workers 0
   ```

3. **检查日志输出**:
   观察进度条和错误信息，确认处理状态

## 与novic的兼容性

### 模型规格
- 使用标准的novic embedder规格：`openclip:timm/ViT-B-16-SigLIP`
- 支持novic的所有配置选项（AMP、设备选择等）
- 输出格式与novic嵌入缓存兼容

### 数据格式
- 512维向量，L2归一化
- fp16精度存储
- 与novic训练流程无缝集成

## 后续使用

生成的嵌入缓存可以用于：
- EoMT模型训练的特征初始化
- 相似性搜索和检索任务  
- 零样本分类和检测
- 视觉特征分析和可视化

## 技术细节

### 嵌入提取过程
1. **Mask应用**: 将mask区域设为不透明，背景设为半透明
2. **边界框计算**: 基于mask的最小外接矩形
3. **填充添加**: 增加10%的边界填充以保持上下文
4. **图像裁剪**: 提取包含目标的最小矩形区域
5. **标准化**: 应用SigLIP的图像预处理管道
6. **特征提取**: 通过SigLIP模型获得512维特征向量

### 文件格式说明
```python
# 加载嵌入向量示例
import numpy as np
embedding = np.load('cache/vis_embed/000000397133_15466.npy')
print(f"Shape: {embedding.shape}")  # (512,)
print(f"Dtype: {embedding.dtype}")  # float16
print(f"Norm: {np.linalg.norm(embedding)}")  # ~1.0 (L2归一化)
``` 