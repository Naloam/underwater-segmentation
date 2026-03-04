"""
通用工具函数

包含图像处理、路径操作等常用功能。
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List
import base64


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    加载图像

    Args:
        image_path: 图像路径

    Returns:
        图像数组 [H, W, 3] RGB格式
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, save_path: Union[str, Path]):
    """
    保存图像

    Args:
        image: 图像数组 [H, W, 3] RGB格式
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 转换为BGR格式（OpenCV默认）
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), image_bgr)


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = None,
                 max_size: int = None, keep_aspect: bool = True) -> np.ndarray:
    """
    调整图像大小

    Args:
        image: 输入图像
        target_size: 目标尺寸 (width, height)
        max_size: 最大尺寸（等比例缩放）
        keep_aspect: 是否保持宽高比

    Returns:
        调整后的图像
    """
    h, w = image.shape[:2]

    if target_size:
        if keep_aspect:
            scale = min(target_size[0] / w, target_size[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = target_size
    elif max_size:
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            return image
    else:
        return image

    return cv2.resize(image, (new_w, new_h))


def image_to_base64(image: np.ndarray, format: str = 'png') -> str:
    """
    将图像转换为base64编码

    Args:
        image: 图像数组 [H, W, 3] RGB格式
        format: 图像格式

    Returns:
        base64编码的字符串
    """
    # 转换为BGR格式
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(f'.{format}', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(base64_str: str) -> np.ndarray:
    """
    将base64编码转换为图像

    Args:
        base64_str: base64编码的字符串

    Returns:
        图像数组 [H, W, 3] RGB格式
    """
    buffer = base64.b64decode(base64_str)
    image = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def create_comparison_grid(images: List[np.ndarray], labels: List[str] = None,
                           rows: int = 1, cols: int = None) -> np.ndarray:
    """
    创建图像对比网格

    Args:
        images: 图像列表
        labels: 标签列表
        rows: 行数
        cols: 列数（如果为None则自动计算）

    Returns:
        拼接后的图像
    """
    n = len(images)
    if cols is None:
        cols = (n + rows - 1) // rows

    # 确保所有图像尺寸相同
    h, w = images[0].shape[:2]
    resized_images = [cv2.resize(img, (w, h)) for img in images]

    # 计算网格尺寸
    grid_h = h * rows
    grid_w = w * cols

    # 创建空白画布
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    # 填充图像
    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        y_start = row * h
        x_start = col * w
        grid[y_start:y_start+h, x_start:x_start+w] = img

        # 添加标签
        if labels and idx < len(labels):
            cv2.putText(grid, labels[idx], (x_start + 10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                       cv2.LINE_AA)
            cv2.putText(grid, labels[idx], (x_start + 10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1,
                       cv2.LINE_AA)

    return grid


def get_image_files(directory: Union[str, Path],
                    extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> List[Path]:
    """
    获取目录中的所有图像文件

    Args:
        directory: 目录路径
        extensions: 图像扩展名列表

    Returns:
        图像文件路径列表
    """
    directory = Path(directory)
    files = []

    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))

    return sorted(files)


def ensure_dir(directory: Union[str, Path]):
    """确保目录存在"""
    Path(directory).mkdir(parents=True, exist_ok=True)


class Timer:
    """简单的计时器"""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0

    def start(self):
        """开始计时"""
        import time
        self.start_time = time.time()

    def stop(self) -> float:
        """停止计时并返回耗时（秒）"""
        import time
        if self.start_time:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
