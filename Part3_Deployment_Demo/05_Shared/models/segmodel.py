"""
Part2分割模型

简单CNN分割模型，包含CBAM注意力模块。
来自第二部分的训练实现。
"""

import torch
import torch.nn as nn


class CBAM(nn.Module):
    """
    CBAM注意力模块（通道注意力 + 空间注意力）

    Reference: "CBAM: Convolutional Block Attention Module"
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_att


class SegModel(nn.Module):
    """
    简单语义分割模型

    结构: Encoder(CNN + CBAM) -> Decoder(1x1 Conv)
    输入: [B, 3, H, W] RGB图像
    输出: [B, num_classes, H, W] 分割logits
    """
    def __init__(self, num_classes=8):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            # 第1层: 3 -> 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(64),

            # 第2层: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第3层: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 解码器
        self.decoder = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            分割logits [B, num_classes, H, W]
        """
        feat = self.encoder(x)
        return self.decoder(feat)


def create_segmodel(num_classes=8, pretrained_path=None, device='cuda'):
    """
    创建分割模型实例

    Args:
        num_classes: 类别数
        pretrained_path: 预训练权重路径
        device: 设备

    Returns:
        model: SegModel实例
    """
    model = SegModel(num_classes=num_classes)
    model = model.to(device)

    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint)
        # 修复Windows编码问题
        import sys
        if sys.platform == 'win32':
            print(f"[OK] Loaded pretrained weights: {pretrained_path}")
        else:
            print(f"✅ 加载预训练权重: {pretrained_path}")

    return model


if __name__ == '__main__':
    # 测试模型
    model = create_segmodel(num_classes=8, device='cpu')
    dummy_input = torch.randn(1, 3, 256, 256)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")

    # 统计参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {params / 1e6:.2f}M")
