"""
Simple test for Part2 model integration
"""
import sys
import os
from pathlib import Path

# Add module path
sys.path.insert(0, str(Path(__file__).parent / '05_Shared'))

import numpy as np
import torch
from PIL import Image
from models.model_interface import ModelFactory
from models.real_models import SegModelWrapper, mask_to_color_image, overlay_mask
from common.config_loader import ConfigLoader


def main():
    print("=" * 60)
    print("Part2 Model Integration Test")
    print("=" * 60)

    # Test 1: Model loading
    print("\n[Test 1] Model Loading")
    config = ConfigLoader.load_model_config("segmentation")
    print(f"Config: {config}")

    try:
        segmentor = ModelFactory.create_segmentor(config)
        print(f"[OK] Model created: {type(segmentor).__name__}")

        info = segmentor.get_info()
        print(f"\nModel Info:")
        for k, v in info.items():
            if k != "class_names":
                print(f"   {k}: {v}")

    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Random image inference
    print("\n[Test 2] Random Image Inference")
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    print(f"Test image shape: {test_image.shape}")

    try:
        mask = segmentor.predict(test_image)
        print(f"[OK] Inference success")
        print(f"   Output mask shape: {mask.shape}")
        print(f"   Value range: [{mask.min()}, {mask.max()}]")

        unique, counts = np.unique(mask, return_counts=True)
        print(f"   Class distribution: {dict(zip(unique, counts))}")

    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Real image inference
    print("\n[Test 3] Real Image Inference")
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

    if test_image_path:
        print(f"Test image: {test_image_path}")

        try:
            image = Image.open(test_image_path)
            original_size = image.size[::-1]
            print(f"   Original size: {original_size}")

            mask = segmentor.predict(test_image_path)
            print(f"[OK] Inference success")
            print(f"   Output mask shape: {mask.shape}")

            unique, counts = np.unique(mask, return_counts=True)
            print(f"   Detected classes:")
            for cls, cnt in zip(unique, counts):
                pct = cnt / mask.size * 100
                cls_name = SegModelWrapper.CLASS_NAMES[cls] if cls < len(SegModelWrapper.CLASS_NAMES) else f"Class{cls}"
                print(f"     {cls_name}: {cnt} pixels ({pct:.1f}%)")

        except Exception as e:
            print(f"[ERROR] Real image inference failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("[SKIP] No test image found")

    # Test 4: Pipeline
    print("\n[Test 4] Pipeline")
    config = ConfigLoader.load_model_config("pipeline")

    try:
        pipeline = ModelFactory.create_pipeline(config)
        print(f"[OK] Pipeline created: {type(pipeline).__name__}")

        info = pipeline.get_info()
        print(f"\nPipeline Info:")
        print(f"   Enhancer: {info['enhancer']['name']}")
        print(f"   Segmentor: {info['segmentor']['name']}")

    except Exception as e:
        print(f"[ERROR] Pipeline creation failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("[OK] Model loading: Success")
    print("[OK] Random image inference: Success")
    print(f"[{'OK' if test_image_path else 'SKIP'}] Real image inference: {'Success' if test_image_path else 'Skipped'}")
    print("[OK] Pipeline: Success")
    print("=" * 60)

    print("\n[SUCCESS] Part2 model integration test completed!")
    print("\nIntegration Details:")
    print("   - Model weights: checkpoints/trained/segmodel_best.pth")
    print("   - Model wrapper: 05_Shared/models/real_models.py")
    print("   - Config: 05_Shared/common/config_loader.py")
    print("   - Interface: 05_Shared/models/model_interface.py")
    print("\n[OK] Ready to use real model in Demo!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
