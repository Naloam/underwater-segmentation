"""
Generate PNG masks for USIS10K dataset
基于颜色聚类生成粗略mask
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def create_simple_color_masks(images_dir: str, output_dir: str):
    """Generate simple color-based masks"""
    img_path = Path(images_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)

    image_files = list(img_path.glob('*.jpg')) + list(img_path.glob('*.png'))

    print(f"Generating masks for {len(image_files)} images...")

    for img_file in tqdm(image_files):
        try:
            img = Image.open(img_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = np.array(img)
            h, w = img_array.shape[:2]

            mask = np.zeros((h, w), dtype=np.uint8)

            r = img_array[:, :, 0].astype(np.int16)
            g = img_array[:, :, 1].astype(np.int16)
            b = img_array[:, :, 2].astype(np.int16)

            # Simple color segmentation
            blue_mask = (b > 100) & (b > g * 1.1) & (b > r * 1.1)
            mask[blue_mask] = 0  # background

            green_mask = (g > 80) & (g > r * 1.1) & (g > b * 0.9) & (~blue_mask)
            mask[green_mask] = 2  # plants

            red_mask = (r > 100) & (r > g * 1.2) & (r > b * 1.2) & (~blue_mask)
            mask[red_mask] = 6  # fish

            darkness = (r + g + b) / 3
            dark_mask = (darkness < 70) & (~blue_mask) & (~green_mask) & (~red_mask)
            mask[dark_mask] = 7  # rocks

            medium_mask = (~blue_mask) & (~green_mask) & (~red_mask) & (~dark_mask)
            mask[medium_mask] = 5  # other

            mask_name = img_file.stem + '.png'
            mask_img = Image.fromarray(mask)
            mask_img.save(out_path / mask_name)

        except Exception as e:
            print(f"Error: {e}")
            continue

    print(f"[OK] Generated masks for {len(image_files)} images")


if __name__ == "__main__":
    BASE_PATH = r"D:\myProjects\大创(1)\USIS10K_Processed\USIS10K_Processed"

    IMAGES_DIR = os.path.join(BASE_PATH, "1_raw")
    OUTPUT_DIR = os.path.join(BASE_PATH, "6_label", "masks")

    create_simple_color_masks(IMAGES_DIR, OUTPUT_DIR)
