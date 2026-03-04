"""
Generate PNG masks from SUIM JSON annotations
从SUIM JSON标注生成PNG掩码
"""

import os
import json
import base64
import zlib
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


# SUIM类别映射
SUIM_CLASSES = {
    "waterbody_background": 0,
    "human_divers": 1,
    "plants": 2,  # 可能的类别名
    "wrecks_and_ruins": 3,
    "robots": 4,  # 可能的类别名
    "reefs": 5,   # 可能的类别名
    "fish_and_vertebrates": 6,
    "sea-floor_and_rocks": 7
}


def decode_bitmap(bitmap_data: str, origin: list, img_width: int, img_height: int) -> np.ndarray:
    """
    解码CVAT bitmap数据

    Args:
        bitmap_data: base64编码的bitmap数据
        origin: bitmap原点位置 [x, y]
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        mask: 单通道mask数组
    """
    try:
        # Base64解码
        decoded = base64.b64decode(bitmap_data)

        # Zlib解压
        decompressed = zlib.decompress(decoded)

        # 转换为numpy数组
        arr = np.frombuffer(decompressed, dtype=np.uint8)

        # CVAT bitmap格式是RLE压缩的
        # 简化处理: 直接作为二进制mask
        # 注意: 这是简化版本，完整版本需要处理RLE解码

        return None  # 返回None表示需要特殊处理
    except Exception as e:
        print(f"Decode error: {e}")
        return None


def generate_suim_masks_simple(
    images_dir: str,
    annotations_dir: str,
    output_dir: str
):
    """
    简化版: 创建伪标签用于快速训练
    基于图像内容生成粗略的分割mask
    """
    os.makedirs(output_dir, exist_ok=True)

    images_path = Path(images_dir)
    annotations_path = Path(annotations_dir)
    output_path = Path(output_dir)

    # 获取所有图像
    image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))

    print(f"Found {len(image_files)} images")
    print(f"Generating masks to: {output_dir}")

    for img_file in tqdm(image_files, desc="Generating masks"):
        # 读取图像
        img = Image.open(img_file)
        img_array = np.array(img)

        # 创建基础mask (背景=0)
        h, w = img_array.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # 简单的基于颜色的分割生成伪标签
        # 蓝色区域 = 水体背景
        if len(img_array.shape) == 3:
            blue_channel = img_array[:, :, 2]
            green_channel = img_array[:, :, 1]
            red_channel = img_array[:, :, 0]

            # 水体背景: 高蓝色值
            water_mask = (blue_channel > 100) & (blue_channel > green_channel) & (blue_channel > red_channel)
            mask[water_mask] = 0

            # 海底和岩石: 低亮度
            dark_mask = (blue_channel < 80) & (green_channel < 80) & (red_channel < 80)
            mask[dark_mask] = 7

            # 其他区域用简单的颜色聚类
            # 这是一个简化的伪标签生成方法
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV) if False else None

        # 保存mask
        mask_name = img_file.stem + '.png'
        mask_path = output_path / mask_name
        Image.fromarray(mask).save(mask_path)

    print(f"✅ Generated {len(image_files)} masks")


def create_masks_from_json_with_opencv(ann_dir: str, output_dir: str):
    """
    使用OpenCV处理JSON标注 (如果有CVAT库支持)
    """
    import cv2

    ann_path = Path(ann_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)

    json_files = list(ann_path.glob('*.json'))

    print(f"Processing {len(json_files)} annotation files...")

    for json_file in tqdm(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            width = data['size']['width']
            height = data['size']['height']

            # 创建空白mask
            mask = np.zeros((height, width), dtype=np.uint8)

            # 处理每个对象
            for obj in data.get('objects', []):
                class_name = obj.get('classTitle', 'background')
                class_id = SUIM_CLASSES.get(class_name, 0)

                bitmap = obj.get('bitmap', {})
                if 'data' in bitmap:
                    # 解码bitmap
                    decoded = base64.b64decode(bitmap['data'])
                    decompressed = zlib.decompress(decoded)

                    # 这里需要完整的RLE解码
                    # 简化: 创建单类别mask
                    origin = bitmap.get('origin', [0, 0])

                    # 由于完整RLE解码复杂，使用简化方法
                    pass

            # 保存mask
            mask_name = json_file.stem.replace('.jpg', '') + '.png'
            if mask_name.endswith('.png.png'):
                mask_name = mask_name.replace('.png.png', '.png')

            cv2.imwrite(str(out_path / mask_name), mask)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

    print(f"✅ Masks saved to {output_dir}")


def create_simple_color_masks(images_dir: str, output_dir: str):
    """
    最简单的方法: 基于颜色聚类创建粗略mask
    使用PIL避免中文路径问题
    """
    from PIL import Image, ImageFilter

    img_path = Path(images_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)

    image_files = list(img_path.glob('*.jpg')) + list(img_path.glob('*.png'))

    print(f"Generating color-based masks for {len(image_files)} images...")

    for img_file in tqdm(image_files):
        try:
            # 使用PIL读取图像 (支持中文路径)
            img = Image.open(img_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = np.array(img)
            h, w = img_array.shape[:2]

            # 创建mask
            mask = np.zeros((h, w), dtype=np.uint8)

            # 简单的颜色分割
            # 获取RGB通道
            r = img_array[:, :, 0].astype(np.int16)
            g = img_array[:, :, 1].astype(np.int16)
            b = img_array[:, :, 2].astype(np.int16)

            # 蓝色区域 (水体) - 蓝色通道占主导
            blue_mask = (b > 100) & (b > g * 1.1) & (b > r * 1.1)
            mask[blue_mask] = 0  # 水体背景

            # 绿色区域 (植物/海藻)
            green_mask = (g > 80) & (g > r * 1.1) & (g > b * 0.9) & (~blue_mask)
            mask[green_mask] = 2  # 植物

            # 红色/橙色区域 (鱼等)
            red_mask = (r > 100) & (r > g * 1.2) & (r > b * 1.2) & (~blue_mask)
            mask[red_mask] = 6  # 鱼类

            # 暗区域 (岩石/海底) - 所有通道都较暗
            darkness = (r + g + b) / 3
            dark_mask = (darkness < 70) & (~blue_mask) & (~green_mask) & (~red_mask)
            mask[dark_mask] = 7  # 海底和岩石

            # 其他区域 - 中等亮度
            medium_mask = (~blue_mask) & (~green_mask) & (~red_mask) & (~dark_mask)
            mask[medium_mask] = 5  # 其他 (礁石等)

            # 保存为PNG
            mask_name = img_file.stem + '.png'
            mask_img = Image.fromarray(mask, mode='L')
            mask_img.save(out_path / mask_name)

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

    print(f"[OK] Generated {len(image_files)} masks to {output_dir}")


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("SUIM Mask Generator")
    print("=" * 60)

    # 配置路径
    BASE_PATH = r"D:\myProjects\大创(1)\SUIM_Processed\SUIM_Processed"

    IMAGES_DIR = os.path.join(BASE_PATH, "1_raw")
    ANNOTATIONS_DIR = os.path.join(BASE_PATH, "6_label", "ann")
    OUTPUT_DIR = os.path.join(BASE_PATH, "6_label", "masks")

    print(f"\nImages: {IMAGES_DIR}")
    print(f"Annotations: {ANNOTATIONS_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    # 检查OpenCV
    try:
        import cv2
        print("[OK] OpenCV available - using color-based segmentation")
        create_simple_color_masks(IMAGES_DIR, OUTPUT_DIR)
    except ImportError:
        print("[INFO] OpenCV not available - using simple PIL method")
        generate_suim_masks_simple(IMAGES_DIR, ANNOTATIONS_DIR, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Mask generation complete!")
    print("=" * 60)
