from transformers import AutoModel, AutoConfig

local_path = "/home/host/siglip2/"

try:
    config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)
    print(f"✅ 配置加载成功: {config.model_type}")
    
    model = AutoModel.from_pretrained(local_path, trust_remote_code=True)
    print(f"✅ 模型加载成功: {model.__class__.__name__}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
except Exception as e:
    print(f"❌ 加载失败: {e}")