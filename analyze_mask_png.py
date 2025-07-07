import numpy as np
from PIL import Image
import json
from pathlib import Path

def analyze_panoptic_png(png_path, json_path=None, image_id=None):
    """
    分析COCO panoptic PNG文件中的segment信息
    
    Args:
        png_path: PNG文件路径
        json_path: 对应的JSON文件路径（可选）
        image_id: 图像ID（可选，用于在JSON中查找对应信息）
    """
    print(f"🔍 分析PNG文件: {png_path}")
    
    # 读取PNG文件
    png_image = Image.open(png_path)
    print(f"   图像尺寸: {png_image.size}")
    print(f"   图像模式: {png_image.mode}")
    
    # 转换为RGB模式（如果不是的话）
    if png_image.mode != 'RGB':
        png_image = png_image.convert('RGB')
    
    # 转换为numpy数组
    rgb_array = np.array(png_image)
    print(f"   数组形状: {rgb_array.shape}")
    
    # 将RGB值转换为segment ID
    # COCO panoptic格式: segment_id = R + G * 256 + B * 256^2
    r, g, b = rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2]
    segment_ids = r.astype(np.uint32) + g.astype(np.uint32) * 256 + b.astype(np.uint32) * 256 * 256
    
    # 获取所有唯一的segment ID
    unique_segment_ids = np.unique(segment_ids)
    print(f"   发现的segment ID数量: {len(unique_segment_ids)}")
    print(f"   所有segment IDs: {sorted(unique_segment_ids)}")
    
    # 分析每个segment的像素数量
    print("\n📊 Segment统计:")
    for sid in sorted(unique_segment_ids):
        pixel_count = np.sum(segment_ids == sid)
        percentage = (pixel_count / segment_ids.size) * 100
        if sid == 0:
            print(f"   ID {sid:>8}: {pixel_count:>8} 像素 ({percentage:>6.2f}%) - 背景")
        else:
            print(f"   ID {sid:>8}: {pixel_count:>8} 像素 ({percentage:>6.2f}%)")
    
    # 如果提供了JSON文件，进行对比
    if json_path and Path(json_path).exists():
        print(f"\n🔗 对比JSON文件: {json_path}")
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # 查找对应的图像信息
        target_annotation = None
        if image_id is not None:
            for annotation in json_data['annotations']:
                if annotation['image_id'] == image_id:
                    target_annotation = annotation
                    break
        
        if target_annotation:
            json_segment_ids = {seg['id'] for seg in target_annotation['segments_info']}
            png_segment_ids = set(unique_segment_ids) - {0}  # 排除背景
            
            print(f"   JSON中的segment数量: {len(json_segment_ids)}")
            print(f"   JSON中的segment IDs: {sorted(json_segment_ids)}")
            print(f"   PNG中的segment数量: {len(png_segment_ids)} (排除背景)")
            print(f"   PNG中的segment IDs: {sorted(png_segment_ids)}")
            
            # 检查一致性
            if json_segment_ids == png_segment_ids:
                print("   ✅ JSON和PNG中的segment ID完全一致")
            else:
                print("   ⚠️  JSON和PNG中的segment ID不一致:")
                only_in_json = json_segment_ids - png_segment_ids
                only_in_png = png_segment_ids - json_segment_ids
                
                if only_in_json:
                    print(f"     只在JSON中: {sorted(only_in_json)}")
                if only_in_png:
                    print(f"     只在PNG中: {sorted(only_in_png)}")
            
            # 显示详细的segment信息
            print(f"\n📋 JSON中的segment详细信息:")
            for seg in target_annotation['segments_info']:
                print(f"   ID {seg['id']:>3}: category_id={seg['category_id']:>3}, "
                      f"isthing={seg.get('isthing', 'N/A')}, iscrowd={seg.get('iscrowd', 'N/A')}")
        else:
            print(f"   ⚠️  在JSON中未找到image_id={image_id}的信息")
    
    return {
        'png_path': png_path,
        'unique_segment_ids': unique_segment_ids,
        'segment_pixel_counts': {sid: np.sum(segment_ids == sid) for sid in unique_segment_ids},
        'image_shape': rgb_array.shape
    }

def batch_analyze_panoptic_files(png_dir, json_path, image_ids=None, max_files=5):
    """
    批量分析多个PNG文件
    
    Args:
        png_dir: PNG文件目录
        json_path: JSON文件路径
        image_ids: 要分析的图像ID列表（如果为None则分析前max_files个）
        max_files: 最大分析文件数
    """
    png_dir = Path(png_dir)
    
    # 读取JSON文件
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # 如果没有指定image_ids，则取前几个
    if image_ids is None:
        image_ids = [ann['image_id'] for ann in json_data['annotations'][:max_files]]
    
    results = []
    
    for image_id in image_ids:
        # 构建PNG文件名（通常是12位数字，左侧补零）
        png_filename = f"{image_id:012d}.png"
        png_path = png_dir / png_filename
        
        if png_path.exists():
            print(f"\n{'='*60}")
            result = analyze_panoptic_png(png_path, json_path, image_id)
            results.append(result)
        else:
            print(f"⚠️  文件不存在: {png_path}")
    
    return results

# 使用示例
if __name__ == "__main__":
    # 分析单个文件
    png_path = "/home/host_ssd/coconut_dataset/coco/panoptic_train2017/000000000008.png"  # 替换为你的实际路径
    json_path = "/home/host_ssd/coconut_dataset/coco/annotations/panoptic_train2017.json"  # 替换为你的实际路径
    
    # 分析特定图像
    result = analyze_panoptic_png(png_path, json_path, image_id=8)
    
    # 或者批量分析
    # png_dir = "/path/to/your/panoptic_train2017/"
    # results = batch_analyze_panoptic_files(png_dir, json_path, image_ids=[8, 139, 285], max_files=3)