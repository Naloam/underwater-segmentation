"""
CLIP Semantic Branch Module

使用HuggingFace预训练CLIP模型提取语义特征
"""

import torch
import torch.nn as nn
from typing import Optional


class CLIPSemanticBranch(nn.Module):
    """
    CLIP语义特征提取分支

    使用CLIP预训练模型提取图像语义特征
    CLIP权重保持冻结，仅作为特征提取器
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        embed_dim: int = 512,
        out_dim: int = 256,
        freeze: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.freeze = freeze

        # 动态导入CLIP（只在需要时加载）
        self.clip_model = None
        self.proj = nn.Linear(embed_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        self._is_loaded = False

    def _load_clip(self):
        """延迟加载CLIP模型"""
        if self._is_loaded:
            return

        try:
            from transformers import CLIPModel, CLIPProcessor
            self.clip_model = CLIPModel.from_pretrained(self.model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(self.model_name)

            if self.freeze:
                for param in self.clip_model.parameters():
                    param.requires_grad = False

            self._is_loaded = True
            print(f"[CLIP] Loaded {self.model_name} (frozen={self.freeze})")

        except Exception as e:
            print(f"[CLIP] Failed to load: {e}")
            print("[CLIP] Using fallback: random projection")

            # Fallback: 使用简单的卷积特征替代
            self.clip_model = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=4, padding=3),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, self.embed_dim)
            )
            if self.freeze:
                for param in self.clip_model.parameters():
                    param.requires_grad = False

            self._is_loaded = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] RGB图像
        Returns:
            features: [B, out_dim] 语义特征
        """
        if not self._is_loaded:
            self._load_clip()

        # 确保 fallback 模型在正确的设备上
        if not hasattr(self.clip_model, 'vision_model'):
            device = x.device
            self.clip_model = self.clip_model.to(device)
            self.proj = self.proj.to(device)
            self.norm = self.norm.to(device)

        B, C, H, W = x.shape

        # CLIP模型的输入
        if hasattr(self.clip_model, 'vision_model'):
            # HuggingFace CLIP - 全局语义特征
            with torch.set_grad_enabled(not self.freeze):
                outputs = self.clip_model.vision_model(pixel_values=x)
                features = outputs.pooler_output  # [B, 512]
                features = self.clip_model.visual_projection(features)
                # 投影到目标维度
                features = self.proj(features)  # [B, out_dim]
        else:
            # Fallback模型 - 全局特征
            with torch.set_grad_enabled(not self.freeze):
                conv_out = self.clip_model[0](x)  # [B, 64, H//4, W//4]
                conv_out = conv_out.mean(dim=[2, 3])  # [B, 64]
                features = self.clip_model[4](conv_out)  # [B, 512]
                # 应用投影到out_dim
                features = self.proj(features)  # [B, out_dim]

        # Layer norm
        features = self.norm(features)

        return features

    def load_from_pretrained(self, checkpoint_path: Optional[str] = None):
        """加载预训练权重（如果有的话）"""
        # CLIP权重已通过HuggingFace加载
        pass


class CLIPFeatureExtractor(nn.Module):
    """
    CLIP多尺度特征提取器

    提取CLIP中间层特征，用于融合
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        out_dim: int = 256,
        freeze: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.out_dim = out_dim
        self.freeze = freeze
        self._is_loaded = False

        self.clip_model = None
        self.proj = nn.Linear(512, out_dim)  # CLIP base输出512维

    def _load_clip(self):
        """延迟加载CLIP"""
        if self._is_loaded:
            return

        try:
            from transformers import CLIPVisionModel
            self.clip_model = CLIPVisionModel.from_pretrained(self.model_name)

            if self.freeze:
                for param in self.clip_model.parameters():
                    param.requires_grad = False

            self._is_loaded = True
            print(f"[CLIP] Loaded vision model: {self.model_name}")

        except Exception as e:
            print(f"[CLIP] Load failed: {e}")
            # 创建占位模型
            self.clip_model = nn.Identity()
            self._is_loaded = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: [B, out_dim]
        """
        if not self._is_loaded:
            self._load_clip()

        if hasattr(self.clip_model, 'encoder'):
            with torch.set_grad_enabled(not self.freeze):
                outputs = self.clip_model(pixel_values=x, output_hidden_states=True)
                # 使用最后一层隐藏状态
                hidden_states = outputs.hidden_states[-1]  # [B, L, D]
                # 取CLS token或池化
                features = hidden_states.mean(dim=1)  # [B, D]
        else:
            features = torch.zeros(x.shape[0], 512, device=x.device)

        features = self.proj(features)
        return features


if __name__ == "__main__":
    # 测试CLIP分支
    print("Testing CLIP Semantic Branch...")

    model = CLIPSemanticBranch(
        model_name="openai/clip-vit-base-patch32",
        out_dim=256,
        freeze=True
    )

    # 模拟输入
    x = torch.randn(2, 3, 224, 224)

    # 首次运行会加载CLIP模型
    print("\nFirst forward pass (loading CLIP)...")
    features = model(x)
    print(f"Output shape: {features.shape}")

    # 再次运行
    print("\nSecond forward pass...")
    features = model(x)
    print(f"Output shape: {features.shape}")

    # 参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\nTrainable params: {total_params:,}")
    print(f"Frozen params: {frozen_params:,}")
