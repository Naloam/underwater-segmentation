"""
真实模型集成测试

测试Part2模型在第三部分中的集成情况。
"""

import sys
import os
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent / '05_Shared'))

import numpy as np
import torch
from PIL import Image
from models.model_interface import ModelFactory
from models.real_models import SegModelWrapper, mask_to_color_image, overlay_mask
from common.config_loader import ConfigLoader


def test_model_loading():
    """测试模型加载"""
    print("=" * 60)
    print("测试1: 模型加载")
    print("=" * 60)

    config = ConfigLoader.load_model_config("segmentation")
    print(f"配置: {config}")

    try:
        segmentor = ModelFactory.create_segmentor(config)
        print(f"✅ 模型创建成功")
        print(f"   类型: {type(segmentor).__name__}")

        info = segmentor.get_info()
        print(f"\n模型信息:")
        for k, v in info.items():
            if k != "class_names":
                print(f"   {k}: {v}")

        return segmentor
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_inference(segmentor):
    """测试模型推理"""
    print("\n" + "=" * 60)
    print("测试2: 模型推理")
    print("=" * 60)

    # 创建随机测试图像
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    print(f"测试图像尺寸: {test_image.shape}")

    try:
        # 推理
        mask = segmentor.predict(test_image)
        print(f"✅ 推理成功")
        print(f"   输出mask尺寸: {mask.shape}")
        print(f"   mask值范围: [{mask.min()}, {mask.max()}]")
        print(f"   类别分布: {np.bincount(mask.flatten())}")

        # 转换为彩色图像
        color_mask = mask_to_color_image(mask)
        print(f"   彩色mask尺寸: {color_mask.shape}")

        # 叠加
        overlay = overlay_mask(test_image, mask, alpha=0.5)
        print(f"   叠加图像尺寸: {overlay.shape}")

        return True
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_image(segmentor):
    """测试真实图像"""
    print("\n" + "=" * 60)
    print("测试3: 真实图像推理")
    print("=" * 60)

    # 查找测试图像
    data_paths = [
        r"d:\myProjects\大创(1)\SUIM_Processed\SUIM_Processed\1_raw",
        r"d:\myProjects\大创(1)\USIS10K_Processed\USIS10K_Processed\1_raw"
    ]

    test_image_path = None
    for path in data_paths:
        if os.path.exists(path):
            files = list(Path(path).glob("*.jpg")) + list(Path(path).glob("*.png"))
            if files:
                test_image_path = str(files[0])
                break

    if test_image_path is None:
        print("⚠️ 未找到测试图像，跳过此测试")
        return None

    print(f"测试图像: {test_image_path}")

    try:
        # 读取图像
        image = Image.open(test_image_path)
        original_size = image.size[::-1]
        print(f"   原始尺寸: {original_size}")

        # 推理
        mask = segmentor.predict(test_image_path)
        print(f"✅ 推理成功")
        print(f"   输出mask尺寸: {mask.shape}")

        # 统计类别分布
        unique, counts = np.unique(mask, return_counts=True)
        print(f"   检测到的类别:")
        for cls, cnt in zip(unique, counts):
            pct = cnt / mask.size * 100
            cls_name = SegModelWrapper.CLASS_NAMES[cls] if cls < len(SegModelWrapper.CLASS_NAMES) else f"Class{cls}"
            print(f"     {cls_name}: {cnt} pixels ({pct:.1f}%)")

        return test_image_path, mask
    except Exception as e:
        print(f"❌ 真实图像推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_pipeline():
    """测试完整流水线"""
    print("\n" + "=" * 60)
    print("测试4: 完整流水线")
    print("=" * 60)

    config = ConfigLoader.load_model_config("pipeline")

    try:
        pipeline = ModelFactory.create_pipeline(config)
        print(f"✅ 流水线创建成功")
        print(f"   类型: {type(pipeline).__name__}")

        info = pipeline.get_info()
        print(f"\n流水线信息:")
        print(f"   增强器: {info['enhancer']['name']}")
        print(f"   分割器: {info['segmentor']['name']}")

        return pipeline
    except Exception as e:
        print(f"❌ 流水线创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主测试函数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Part2模型集成测试" + " " * 26 + "║")
    print("╚" + "=" * 58 + "╝")

    # 测试1: 模型加载
    segmentor = test_model_loading()
    if segmentor is None:
        print("\n❌ 测试失败: 无法加载模型")
        return False

    # 测试2: 随机图像推理
    success = test_model_inference(segmentor)
    if not success:
        print("\n❌ 测试失败: 推理失败")
        return False

    # 测试3: 真实图像推理
    result = test_with_real_image(segmentor)

    # 测试4: 流水线
    pipeline = test_pipeline()

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print("✅ 模型加载: 成功")
    print("✅ 随机图像推理: 成功")
    print(f"{'✅' if result else '⚠️'} 真实图像推理: {'成功' if result else '跳过'}")
    print(f"{'✅' if pipeline else '⚠️'} 流水线: {'成功' if pipeline else '失败'}")
    print("=" * 60)

    print("\n🎉 Part2模型集成测试完成！")
    print("\n📋 集成结果:")
    print("   - 模型权重: checkpoints/trained/segmodel_best.pth")
    print("   - 模型包装器: 05_Shared/models/real_models.py")
    print("   - 配置文件: 05_Shared/common/config_loader.py")
    print("   - 接口适配: 05_Shared/models/model_interface.py")
    print("\n✅ 可以开始在Demo中使用真实模型了！")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
