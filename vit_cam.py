import math
from pytorch_grad_cam import EigenCAM

# 核心难点 1：ViT 序列特征到 2D 空间特征的转换 (Reshape Transform)
def vit_reshape_transform(tensor):
    # 输入 tensor 维度通常是 [B, N, D] 或者 [B, N+1, D] (包含 CLS token)
    b, n, d = tensor.shape
    
    # 判断是否包含 CLS token，并计算特征图边长 H
    h = int(math.sqrt(n))
    if h * h != n:
        h = int(math.sqrt(n - 1))
        tokens = tensor[:, 1:, :]  # 丢弃第一个 CLS token
    else:
        tokens = tensor
        
    # 将 [B, H*H, D] 重组为 [B, H, H, D]，再 Permute 为 CAM 需要的 [B, D, H, W]
    return tokens.reshape(b, h, h, d).permute(0, 3, 1, 2)

def generate_cam_visualization(model, image_tensor):
    # 核心难点 2：精准定位提取特征的 Hook 点 (通常是 ViT 最后一层的 LayerNorm)
    # 比如在 SigLIP/PaliGemma 中：
    target_layer = model.paligemma.vision_tower.encoder.layers[-1].layer_norm
    
    # 核心难点 3：使用 EigenCAM 而不是常规的 GradCAM
    cam = EigenCAM(
        model=model.vision_tower, 
        target_layers=[target_layer],
        reshape_transform=vit_reshape_transform # 传入维度转换逻辑
    )
    
    # 提取第一张图的激活热力图：输出维度 [H, W]
    grayscale_cam = cam(input_tensor=image_tensor)[0]
    return grayscale_cam