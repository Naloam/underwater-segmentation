"""
CLI Demo - Command-line version for testing
"""
import sys
from pathlib import Path

# Add module paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / '05_Shared'))
sys.path.insert(0, str(project_root / '04_Demo' / 'src'))

import numpy as np
from PIL import Image

from models.model_interface import ModelFactory
from common.config_loader import ConfigLoader


def demo_test_images():
    """Test with real images"""
    print("=" * 60)
    print("CLI Demo - Underwater Image Segmentation")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading models...")
    config = ConfigLoader.load_model_config("segmentation")
    segmentor = ModelFactory.create_segmentor(config)

    model_info = segmentor.get_info()
    print(f"  Model: {model_info.get('name')}")
    print(f"  Params: {model_info.get('params_M', 0):.2f}M")
    print(f"  Device: {model_info.get('device')}")

    # Find test images
    print("\n[2/3] Finding test images...")
    data_paths = [
        r"d:\myProjects\大创(1)\SUIM_Processed\SUIM_Processed\1_raw",
        r"d:\myProjects\大创(1)\USIS10K_Processed\USIS10K_Processed\1_raw"
    ]

    test_images = []
    for path in data_paths:
        p = Path(path)
        if p.exists():
            files = list(p.glob("*.jpg")) + list(p.glob("*.png"))
            test_images.extend(files)

    if not test_images:
        print("  [!] No test images found!")
        return

    print(f"  Found {len(test_images)} test images")

    # Process images
    print("\n[3/3] Processing images...")
    from models.real_models import mask_to_color_image, overlay_mask

    output_dir = Path('./output/demo_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, img_path in enumerate(test_images[:5]):
        print(f"\n  Processing {i+1}/{min(5, len(test_images))}: {img_path.name}")

        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)

            # Inference
            mask = segmentor.predict(str(img_path))

            # Generate visualizations
            color_mask = mask_to_color_image(mask)
            overlay = overlay_mask(img_array, mask, alpha=0.5)

            # Save results
            result_path = output_dir / f"result_{i+1}_{img_path.stem}.png"

            # Create comparison image
            import cv2
            h, w = img_array.shape[:2]
            comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
            comparison[:, :w] = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            comparison[:, w:2*w] = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
            comparison[:, 2*w:] = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(result_path), comparison)

            # Analyze results
            unique, counts = np.unique(mask, return_counts=True)
            total = mask.size

            print(f"    Detected classes:")
            for cls, cnt in zip(unique, counts):
                pct = cnt / total * 100
                class_names = ["Background", "Divers", "Plants", "Wrecks",
                               "Robots", "Reefs", "Fish", "Sea floor"]
                name = class_names[cls] if cls < len(class_names) else f"Class{cls}"
                print(f"      {name}: {pct:.1f}%")

            print(f"    Saved: {result_path}")
            results.append({
                'path': img_path,
                'result': result_path,
                'classes': dict(zip(unique, counts))
            })

        except Exception as e:
            print(f"    [!] Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Demo Summary")
    print("=" * 60)
    print(f"Processed: {len(results)} images")
    print(f"Output directory: {output_dir}")
    print("\nOutput files:")
    for r in results:
        print(f"  - {r['result']}")
    print("\n[SUCCESS] CLI Demo completed!")


def demo_scenarios():
    """Test with predefined scenarios"""
    print("\n" + "=" * 60)
    print("Scenario Presets Test")
    print("=" * 60)

    scenarios = {
        "coral_reef": {
            "name": "Shallow Coral Reef",
            "description": "Detects reefs, fish, plants",
            "enhancement": "CLAHE",
            "target_classes": [0, 5, 6, 7]
        },
        "deep_sea": {
            "name": "Deep Sea Ruins",
            "description": "Detects starfish, sea cucumbers",
            "enhancement": "Histogram Equalization",
            "target_classes": [0, 5, 7]
        },
        "marine_life": {
            "name": "Marine Life Monitoring",
            "description": "Fish tracking and counting",
            "enhancement": "CLAHE",
            "target_classes": [0, 6]
        }
    }

    for key, scenario in scenarios.items():
        print(f"\n[{key}] {scenario['name']}")
        print(f"  Description: {scenario['description']}")
        print(f"  Enhancement: {scenario['enhancement']}")
        print(f"  Target classes: {scenario['target_classes']}")


if __name__ == '__main__':
    demo_test_images()
    demo_scenarios()
