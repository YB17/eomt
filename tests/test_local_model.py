import os
from pathlib import Path
import torch

# 测试 OpenCLIP 格式的 SigLIP2 模型
local_path = "/home/host/siglip2/"

print(f"🔍 测试路径: {local_path}")
print(f"📁 目录存在: {os.path.exists(local_path)}")

# 检查文件
files = os.listdir(local_path)
print(f"📄 文件列表:\n   " + "\n   ".join(files))

# 检查 OpenCLIP 特征文件
openclip_files = [
    "open_clip_model.safetensors",
    "open_clip_pytorch_model.bin",
    "open_clip_config.json",
]
found_openclip = [f for f in openclip_files if f in files]
print(f"\n✅ OpenCLIP 文件: {found_openclip}")

# 尝试加载为 EoMT backbone
try:
    from eomt.models.backbones.siglip2_vit import SigLIP2ViTBackbone
    
    print("\n" + "="*60)
    print("🚀 测试 1: 加载 SigLIP2ViTBackbone (OpenCLIP)")
    print("="*60)
    
    backbone = SigLIP2ViTBackbone(
        model_id=local_path,
        out_indices=(-1,),
        drop_path=0.0,
        naflex=True,
        fp16=True,
    )
    
    print(f"✅ Backbone 加载成功!")
    print(f"   嵌入维度: {backbone.embed_dim}")
    print(f"   层数: {backbone.num_blocks}")
    print(f"   参数量: {sum(p.numel() for p in backbone.parameters()) / 1e6:.1f}M")
    
    # 测试前向传播
    print("\n" + "="*60)
    print("🧪 测试 2: 前向传播")
    print("="*60)
    
    dummy_input = torch.randn(1, 3, 512, 512)
    if torch.cuda.is_available():
        backbone = backbone.cuda()
        dummy_input = dummy_input.cuda()
    
    with torch.no_grad():
        outputs = backbone(dummy_input)
    
    print(f"✅ 前向传播成功!")
    print(f"   输出数量: {len(outputs)}")
    print(f"   输出形状: {outputs[0].shape}")
    print(f"   预期形状: [1, {(512//16)**2}, {backbone.embed_dim}]")
    
    print("\n" + "="*60)
    print("🎉 所有测试通过!")
    print("="*60)
    
except ImportError as e:
    print(f"\n❌ 导入失败: {e}")
    print("   请确保已安装: pip install open_clip_torch")
    
except Exception as e:
    print(f"\n❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()