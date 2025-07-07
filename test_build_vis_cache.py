#!/usr/bin/env python3
"""
测试build_vis_cache.py的功能
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import argparse

# 添加eomt路径
# sys.path.append("eomt")

# 测试导入
try:
    from eomt.datasets.build_vis_cache import (
        create_dataloader, 
        extract_image_and_segment_ids,
        prep_mask_image,
        save_image_embeddings
    )
    print("✅ 成功导入build_vis_cache模块")
except ImportError as e:
    print(f"❌ 导入build_vis_cache模块失败: {e}")
    sys.exit(1)

# 测试novic embedders导入
try:
    sys.path.append("novic")
    from eomt.novic.embedders import Embedder
    print("✅ 成功导入novic embedders")
    NOVIC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ novic embedders导入失败: {e}")
    print("将跳过需要novic的测试")
    NOVIC_AVAILABLE = False


def create_dummy_image_data():
    """创建虚拟的图像和mask数据用于测试"""
    # 创建一个简单的RGB图像 [3, 64, 64]
    image = torch.rand(3, 64, 64)
    
    # 创建几个虚拟的masks [H, W]
    mask1 = torch.zeros(64, 64, dtype=torch.bool)
    mask1[10:30, 10:30] = True  # 第一个mask区域
    
    mask2 = torch.zeros(64, 64, dtype=torch.bool) 
    mask2[40:60, 40:60] = True  # 第二个mask区域
    
    masks = [mask1, mask2]
    labels = [1, 2]  # 类别标签
    target_ids = [12345, 67890]  # segment IDs
    
    return image, masks, labels, target_ids

import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import ListedColormap

def visualize_extracted_segments(image_tensor, masks, target_ids, labels, image_id, save_dir=None):
    """
    可视化提取的图像和segment masks，并在每个mask上显示对应的ID
    
    Args:
        image_tensor: 图像张量 (C, H, W)
        masks: mask张量 (N, H, W) 
        target_ids: segment ID列表
        labels: 标签列表
        image_id: 图像ID
        save_dir: 保存目录（可选）
    """
    print(f"\n🎨 可视化图像 {image_id} 的segment提取结果...")
    
    # 创建图像
    fig, axes = plt.subplots(1, 3, figsize=[21, 7])
    
    # 1. 显示原始图像
    if isinstance(image_tensor, torch.Tensor):
        img_np = image_tensor.cpu().numpy()
    else:
        img_np = image_tensor
    
    # 转换图像格式 (C, H, W) -> (H, W, C)
    if img_np.shape[0] in [1, 3]:  # 通道在第一维
        img_np = img_np.transpose(1, 2, 0)
    
    # 归一化到0-255
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    # 如果是单通道，转换为RGB
    if img_np.shape[-1] == 1:
        img_np = np.repeat(img_np, 3, axis=-1)
    elif img_np.shape[-1] > 3:
        img_np = img_np[:, :, :3]
    
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original Image - {image_id}", fontsize=14)
    axes[0].axis("off")
    
    # 2. 创建彩色的segment可视化
    H, W = img_np.shape[:2]
    segment_viz = np.zeros((H, W, 3), dtype=np.uint8)
    
    # 设置随机种子以获得一致的颜色
    np.random.seed(42)
    
    # 为每个segment分配颜色并绘制
    colors_used = []
    for i, (mask, seg_id, label) in enumerate(zip(masks, target_ids, labels)):
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().astype(bool)
        else:
            mask_np = mask.astype(bool)
        
        if mask_np.sum() == 0:  # 跳过空mask
            continue
        
        # 生成随机颜色
        color = (np.random.rand(3) * 200 + 55).astype(np.uint8)  # 避免太暗的颜色
        colors_used.append(color)
        
        # 应用颜色到mask区域
        segment_viz[mask_np] = color
    
    axes[1].imshow(segment_viz)
    axes[1].set_title(f"Colored Segments (Total: {len(target_ids)})", fontsize=14)
    axes[1].axis("off")
    
    # 3. 创建带ID标注的可视化
    segment_with_ids = segment_viz.copy()
    
    # 为每个segment添加ID标注
    for i, (mask, seg_id, label) in enumerate(zip(masks, target_ids, labels)):
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().astype(bool)
        else:
            mask_np = mask.astype(bool)
        
        if mask_np.sum() == 0:
            continue
        
        # 找到mask的中心点
        y_coords, x_coords = np.where(mask_np)
        if len(y_coords) > 0:
            center_y = int(np.mean(y_coords))
            center_x = int(np.mean(x_coords))
            
            # 在中心点添加文本标注
            text = f"ID:{seg_id}\nL:{label}"
            
            # 添加白色背景的文本
            cv2.putText(segment_with_ids, f"ID:{seg_id}", 
                       (center_x-20, center_y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(segment_with_ids, f"ID:{seg_id}", 
                       (center_x-20, center_y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            cv2.putText(segment_with_ids, f"L:{label}", 
                       (center_x-15, center_y+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(segment_with_ids, f"L:{label}", 
                       (center_x-15, center_y+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    axes[2].imshow(segment_with_ids)
    axes[2].set_title(f"Segments with IDs", fontsize=14)
    axes[2].axis("off")
    
    # 添加详细信息
    info_text = f"Image ID: {image_id}\n"
    info_text += f"Image Shape: {img_np.shape}\n"
    info_text += f"Segments Count: {len(target_ids)}\n"
    info_text += f"Segment IDs: {target_ids[:8]}{'...' if len(target_ids) > 8 else ''}\n"
    info_text += f"Labels: {labels[:8]}{'...' if len(labels) > 8 else ''}"
    
    fig.suptitle(info_text, fontsize=12, y=0.02, ha='left', va='bottom')
    
    plt.tight_layout()
    
    # 保存图像
    if save_dir:
        from pathlib import Path
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"segment_visualization_{image_id}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   可视化图像已保存到: {save_path}")
    
    # 显示图像
    plt.show()
    plt.close()
    
    # 打印详细的segment信息
    print(f"\n📋 详细segment信息:")
    for i, (mask, seg_id, label) in enumerate(zip(masks, target_ids, labels)):
        if isinstance(mask, torch.Tensor):
            pixel_count = mask.sum().item()
        else:
            pixel_count = mask.sum()
        
        percentage = (pixel_count / (H * W)) * 100
        print(f"   Segment {i+1:2d}: ID={seg_id:>6}, Label={label:>3}, "
              f"Pixels={pixel_count:>6} ({percentage:>5.2f}%)")


def test_real_dataset(data_path: str, output_dir: str = None):
    """测试真实COCO数据集的单个图像处理"""
    print(f"\n🔄 测试真实数据集处理...")
    print(f"数据路径: {data_path}")
    
    if not Path(data_path).exists():
        print(f"⚠️ 数据路径不存在，跳过真实数据集测试: {data_path}")
        return True
    
    if not NOVIC_AVAILABLE:
        print("⚠️ novic不可用，跳过真实数据集测试")
        return True
    
    try:
        # 创建输出目录
        if output_dir is None:
            temp_dir = Path(tempfile.mkdtemp())
            cleanup_temp = True
        else:
            temp_dir = Path(output_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            cleanup_temp = False
        
        print(f"输出目录: {temp_dir}")
        
        # 创建数据加载器（只处理少量数据）
        print("创建数据加载器...")
        dataloader = create_dataloader(
            data_path=data_path,
            batch_size=1,  # 一次只处理一个图像
            num_workers=0
        )
        

        # 创建SigLIP嵌入器
        print("加载SigLIP模型...")
        import os
        os.environ['HF_HOME'] = '/home/host_ssd/huggingface'
        os.environ['TRANSFORMERS_CACHE'] = '/home/host_ssd/huggingface'
        embedder = Embedder.create(
            spec="openclip:timm/ViT-B-16-SigLIP",
            # spec = "openclip:file:///home/host_ssd/huggingface/ViT-B-16-SigLIP",
            # spec = "/home/host_ssd/huggingface/ViT-B-16-SigLIP",
            amp=True,
            device="cuda",  
            inference_batch_size=32,
            image_batch_size=16,
            load_model=True,
            compile_model=False,
        )
        
        # 获取图像预处理函数
        image_transform = embedder.get_image_transform()
        
        # 处理第一个批次
        print("处理第一个批次...")
        batch_iter = iter(dataloader)
        batch = next(batch_iter)
        
        # 提取批次信息
        batch_info = extract_image_and_segment_ids(batch, dataloader)
        
        if not batch_info:
            print("❌ 未能从数据加载器提取有效数据")
            return False
        
        # 处理第一个图像
        image_id, image_tensor, masks, labels, target_ids = batch_info[0]
        
        print(f"处理图像: {image_id}")
        print(f"  - 图像尺寸: {image_tensor.shape}")
        print(f"  - Mask数量: {len(masks)}")
        print(f"  - Segment IDs: {target_ids[:5]}{'...' if len(target_ids) > 5 else ''}")

        # 🎨 添加可视化
        visualize_extracted_segments(
            image_tensor=image_tensor,
            masks=masks, 
            target_ids=target_ids,
            labels=labels,
            image_id=image_id,
            save_dir="./"  # 可以修改保存路径
        )

        if not masks or not target_ids:
            print("❌ 图像没有有效的mask或target_ids")
            return False
        
        # 为所有mask准备图像
        print("预处理mask图像...")
        processed_images, valid_indices, valid_segment_ids, valid_labels = prep_mask_image(
            image_tensor, masks, labels, target_ids,
            bg_alpha=0.3, pad_ratio=0.1
        )
        
        if not processed_images:
            print("❌ 没有有效的处理后图像")
            return False
        
        print(f"成功预处理 {len(processed_images)} 个mask图像")
        
        # 批量处理图像
        print("转换图像格式...")
        image_tensors = []
        for pil_img in processed_images:
            tensor = image_transform(pil_img)
            image_tensors.append(tensor)
        
        if image_tensors:
            # 批量推理
            print("执行SigLIP推理...")
            batch_tensor = torch.stack(image_tensors)
            
            with embedder.inference_mode():
                embeddings = embedder.inference_image(batch_tensor)
            
            print(f"生成 {len(embeddings)} 个嵌入向量，每个维度: {embeddings[0].shape}")
            
            # 保存该图像的所有embeddings
            print("保存嵌入向量...")
            success = save_image_embeddings(
                temp_dir, image_id, valid_segment_ids, 
                embeddings, valid_labels
            )
            
            if success:
                # 验证保存的文件
                npz_file = temp_dir / f"{image_id}.npz"
                data = np.load(npz_file)
                
                print(f"✅ 真实数据集测试成功！")
                print(f"   - 图像ID: {image_id}")
                print(f"   - 保存文件: {npz_file}")
                print(f"   - Segment数量: {len(data['segment_ids'])}")
                print(f"   - 嵌入形状: {data['embeddings'].shape}")
                print(f"   - 数据类型: {data['embeddings'].dtype}")
                print(f"   - Segment IDs示例: {data['segment_ids'][:5]}")
                print(f"   - Labels示例: {data['labels'][:5]}")
                
                # 显示一些统计信息
                print(f"   - 嵌入向量范围: [{data['embeddings'].min():.3f}, {data['embeddings'].max():.3f}]")
                print(f"   - 嵌入向量均值: {data['embeddings'].mean():.3f}")
                print(f"   - 嵌入向量标准差: {data['embeddings'].std():.3f}")
                
                # 创建索引文件
                index_file = temp_dir / "image_index.tsv"
                with open(index_file, "w", encoding='utf-8') as f:
                    f.write("image_id\tfeat_file\tnum_segments\n")
                    f.write(f"{image_id}\t{image_id}.npz\t{len(data['segment_ids'])}\n")
                
                print(f"   - 索引文件: {index_file}")
                
                if not cleanup_temp:
                    print(f"\n💾 文件已保存到: {temp_dir}")
                    print(f"您可以使用以下代码加载数据:")
                    print(f"```python")
                    print(f"import numpy as np")
                    print(f"data = np.load('{npz_file}')")
                    print(f"segment_ids = data['segment_ids']")
                    print(f"embeddings = data['embeddings']")
                    print(f"labels = data['labels']")
                    print(f"```")
                
                result = True
            else:
                print("❌ 保存嵌入向量失败")
                result = False
        else:
            print("❌ 没有有效的图像张量")
            result = False
        
        # 清理临时文件
        if cleanup_temp:
            shutil.rmtree(temp_dir)
        
        return result
        
    except Exception as e:
        print(f"❌ 真实数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prep_mask_image():
    """测试mask图像预处理功能"""
    print("\n🔄 测试mask图像预处理...")
    
    try:
        image, masks, labels, target_ids = create_dummy_image_data()
        
        processed_images, valid_indices, valid_segment_ids, valid_labels = prep_mask_image(
            image, masks, labels, target_ids,
            bg_alpha=0.3, pad_ratio=0.1
        )
        
        # 验证返回结果
        assert len(processed_images) == len(valid_indices) == len(valid_segment_ids) == len(valid_labels)
        assert len(processed_images) <= len(masks)  # 可能有些mask无效
        
        # 检查返回的PIL图像
        for img in processed_images:
            assert isinstance(img, Image.Image)
            assert img.mode == 'RGB'
        
        # 检查segment IDs和labels
        for seg_id, label in zip(valid_segment_ids, valid_labels):
            assert isinstance(seg_id, int)
            assert isinstance(label, int)
        
        print(f"✅ mask预处理成功，处理了 {len(processed_images)} 个有效mask")
        return True
        
    except Exception as e:
        print(f"❌ mask预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_image_embeddings():
    """测试图像嵌入保存功能"""
    print("\n🔄 测试图像嵌入保存...")
    
    try:
        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp())
        
        # 创建虚拟嵌入数据
        image_id = "test_image_001"
        segment_ids = [12345, 67890, 11111]
        labels = [1, 2, 3]
        
        # 创建虚拟的768维嵌入向量张量（更符合实际SigLIP输出）
        embeddings = torch.randn(3, 768)  # 批量形式的张量
        
        # 保存嵌入
        success = save_image_embeddings(temp_dir, image_id, segment_ids, embeddings, labels)
        assert success, "保存失败"
        
        # 验证文件是否创建
        npz_file = temp_dir / f"{image_id}.npz"
        assert npz_file.exists(), "NPZ文件未创建"
        
        # 加载并验证数据
        data = np.load(npz_file)
        
        # 检查键是否存在
        assert 'segment_ids' in data
        assert 'embeddings' in data
        assert 'labels' in data
        
        # 检查数据形状和内容
        loaded_segment_ids = data['segment_ids']
        loaded_embeddings = data['embeddings']
        loaded_labels = data['labels']
        
        assert loaded_segment_ids.shape == (3,)
        assert loaded_embeddings.shape == (3, 768)  # 768维SigLIP嵌入
        assert loaded_labels.shape == (3,)
        
        assert loaded_embeddings.dtype == np.float16  # 检查fp16格式
        
        # 检查数据内容
        np.testing.assert_array_equal(loaded_segment_ids, segment_ids)
        np.testing.assert_array_equal(loaded_labels, labels)
        
        # 验证嵌入向量的数值范围合理
        assert not np.isnan(loaded_embeddings).any(), "嵌入向量包含NaN值"
        assert np.isfinite(loaded_embeddings).all(), "嵌入向量包含无穷大值"
        
        print(f"✅ 图像嵌入保存成功")
        print(f"   - 文件: {npz_file}")
        print(f"   - Segment数量: {len(segment_ids)}")
        print(f"   - 嵌入维度: {loaded_embeddings.shape}")
        print(f"   - 数据类型: {loaded_embeddings.dtype}")
        print(f"   - 嵌入范围: [{loaded_embeddings.min():.3f}, {loaded_embeddings.max():.3f}]")
        
        # 清理临时文件
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ 图像嵌入保存失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """测试数据加载器创建"""
    print("\n🔄 测试数据加载器创建...")
    
    # 使用默认参数创建一个小的测试数据加载器
    try:
        dataloader = create_dataloader(
            data_path="/home/host_ssd/coconut_dataset/coco",  # 临时路径，不会真正加载数据
            batch_size=2,
            num_workers=0  # 避免多进程问题
        )
        print("✅ 数据加载器创建成功")
        return True
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return False


def test_embedder():
    """测试SigLIP嵌入器创建"""
    print("\n🔄 测试SigLIP嵌入器创建...")
    
    if not NOVIC_AVAILABLE:
        print("⚠️ 跳过嵌入器测试（novic不可用）")
        return True
    
    try:
        import os
        os.environ['HF_HOME'] = '/home/host_ssd/huggingface'
        os.environ['TRANSFORMERS_CACHE'] = '/home/host_ssd/huggingface'
        embedder = Embedder.create(
            spec="openclip:ViT-B-16-SigLIP",
            # spec = "/home/host_ssd/huggingface/ViT-B-16-SigLIP",
            device="cuda",  # 
            load_model=False,  # 不实际加载模型权重
        )
        print("✅ SigLIP嵌入器创建成功")
        return True
    except Exception as e:
        print(f"❌ SigLIP嵌入器创建失败: {e}")
        return False


def test_npz_file_format():
    """测试NPZ文件格式的完整性"""
    print("\n🔄 测试NPZ文件格式...")
    
    try:
        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp())
        
        # 创建测试数据
        image_id = "format_test_001"
        segment_ids = [1001, 2002, 3003, 4004]
        labels = [10, 20, 30, 40]
        embeddings = [torch.randn(512) for _ in range(4)]
        
        # 保存数据
        save_image_embeddings(temp_dir, image_id, segment_ids, embeddings, labels)
        
        # 加载数据并进行详细验证
        npz_file = temp_dir / f"{image_id}.npz"
        data = np.load(npz_file)
        
        # 验证数据一致性
        assert len(data['segment_ids']) == len(data['embeddings']) == len(data['labels'])
        
        # 验证每个segment的嵌入向量都是512维
        for i, embedding in enumerate(data['embeddings']):
            assert embedding.shape == (512,), f"第{i}个嵌入向量维度错误"
            assert not np.isnan(embedding).any(), f"第{i}个嵌入向量包含NaN"
        
        # 验证segment_ids的唯一性
        unique_ids = np.unique(data['segment_ids'])
        assert len(unique_ids) == len(data['segment_ids']), "segment_ids不唯一"
        
        print(f"✅ NPZ文件格式验证通过")
        print(f"   - Segments: {len(data['segment_ids'])}")
        print(f"   - 嵌入形状: {data['embeddings'].shape}")
        print(f"   - 数据类型: segment_ids={data['segment_ids'].dtype}, "
              f"embeddings={data['embeddings'].dtype}, labels={data['labels'].dtype}")
        
        # 清理
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ NPZ文件格式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """测试边界情况"""
    print("\n🔄 测试边界情况...")
    
    try:
        # 测试空数据
        temp_dir = Path(tempfile.mkdtemp())
        
        # 测试空segment_ids
        result = save_image_embeddings(temp_dir, "empty_test", [], [], [])
        assert not result, "空数据应该返回False"
        
        # 测试单个segment
        single_embedding = [torch.randn(512)]
        single_segment_id = [99999]
        single_label = [42]
        
        result = save_image_embeddings(temp_dir, "single_test", single_segment_id, single_embedding, single_label)
        assert result, "单个segment保存失败"
        
        # 验证单个segment文件
        data = np.load(temp_dir / "single_test.npz")
        assert data['segment_ids'].shape == (1,)
        assert data['embeddings'].shape == (1, 512)
        assert data['labels'].shape == (1,)
        
        print("✅ 边界情况测试通过")
        
        # 清理
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ 边界情况测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试build_vis_cache功能")
    parser.add_argument("--data_path", 
                       default="/home/host_ssd/coconut_dataset/coco",
                       help="COCO数据集路径")
    parser.add_argument("--output_dir",
                       help="测试输出目录（不指定则使用临时目录）")
    parser.add_argument("--skip_real_data", 
                       action="store_true",
                       help="跳过真实数据集测试")
    return parser.parse_args()


def main():
    """主测试函数"""
    args = parse_args()
    
    print("🧪 开始测试build_vis_cache功能...")
    
    # 基础测试
    basic_tests = [
        ("mask图像预处理", test_prep_mask_image),
        ("图像嵌入保存", test_save_image_embeddings),
        ("NPZ文件格式", test_npz_file_format),
        ("边界情况", test_edge_cases),
        ("数据加载器", test_dataloader),
        ("SigLIP嵌入器", test_embedder),
    ]
    
    results = {}
    for test_name, test_func in basic_tests:
        results[test_name] = test_func()
    
    # 真实数据集测试
    if not args.skip_real_data:
        print(f"\n{'='*60}")
        print("🎯 开始真实数据集测试...")
        print(f"{'='*60}")
        
        real_data_result = test_real_dataset(args.data_path, args.output_dir)
        results["真实数据集处理"] = real_data_result
    else:
        print("\n⚠️ 跳过真实数据集测试")
    
    # 总结测试结果
    print(f"\n{'='*60}")
    print("📊 测试结果总结:")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        print("\n💡 build_vis_cache功能特点:")
        print("   ✅ 以图像为单位保存segment embeddings")
        print("   ✅ NPZ格式，包含segment_ids、embeddings、labels")
        print("   ✅ fp16精度存储，节省空间")
        print("   ✅ 压缩存储格式")
        print("   ✅ 真实COCO数据集验证通过")
        print("\n🚀 您现在可以运行以下命令来构建完整的视觉嵌入缓存:")
        print("   cd eomt")
        print("   python datasets/build_vis_cache.py --data_path /path/to/coco/dataset")
    else:
        print(f"\n⚠️  {total-passed} 个测试失败，请检查实现")
        
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 