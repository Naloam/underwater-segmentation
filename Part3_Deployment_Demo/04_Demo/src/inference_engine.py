"""
推理引擎

负责图像增强和分割的推理。
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Union
import numpy as np
from PIL import Image
import yaml

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / '05_Shared'))

from models.model_interface import ModelFactory
from common.config_loader import ConfigLoader


class InferenceEngine:
    """
    推理引擎

    支持图像增强和全景分割推理。
    """

    def __init__(self, device='auto'):
        """
        初始化推理引擎

        Args:
            device: 设备 ('auto', 'cuda', 'cpu')
        """
        if device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.segmentor = None
        self.enhancer = None
        self.model_info = {}

        self._load_models()

    def _load_models(self):
        """加载模型"""
        # 加载分割模型
        seg_config = ConfigLoader.load_model_config("segmentation")
        seg_config['device'] = self.device
        self.segmentor = ModelFactory.create_segmentor(seg_config)
        self.model_info['segmentor'] = self.segmentor.get_info()

        # 加载增强模型
        enhance_config = ConfigLoader.load_model_config("enhancement")
        self.enhancer = ModelFactory.create_enhancer(enhance_config)
        self.model_info['enhancer'] = self.enhancer.get_info()

        print(f"[InferenceEngine] Models loaded on {self.device}")

    def process(self, image: Union[np.ndarray, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理单张图像（增强 + 分割）

        Args:
            image: 输入图像，可以是numpy数组或图像路径

        Returns:
            enhanced: 增强后图像 [H, W, 3]
            mask: 分割mask [H, W]
        """
        # 增强
        enhanced = self.enhancer.enhance(image)

        # 分割
        mask = self.segmentor.predict(enhanced)

        return enhanced, mask

    def process_batch(self, images: list) -> Tuple[list, list]:
        """
        批量处理图像

        Args:
            images: 图像列表

        Returns:
            enhanced_list: 增强后图像列表
            mask_list: 分割mask列表
        """
        enhanced_list = self.enhancer.enhance_batch(images)
        mask_list = self.segmentor.predict_batch(enhanced_list)

        return enhanced_list, mask_list

    def benchmark(self, image: np.ndarray, num_runs: int = 10) -> Dict[str, Any]:
        """
        性能测试

        Args:
            image: 测试图像
            num_runs: 运行次数

        Returns:
            性能统计
        """
        times = []

        # 预热
        self.process(image)

        # 测试
        for _ in range(num_runs):
            start = time.time()
            self.process(image)
            end = time.time()
            times.append(end - start)

        avg_time = np.mean(times)
        fps = 1.0 / avg_time

        return {
            'avg_time': avg_time,
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'fps': fps,
            'device': self.device
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.model_info


class ScenarioManager:
    """
    场景管理器

    管理不同场景的预设配置。
    """

    # 预设场景
    SCENARIOS = {
        "浅海珊瑚礁": {
            "scene": "浅海珊瑚礁",
            "description": "适用于浅水环境，光线充足，色彩丰富",
            "enhance_method": "clahe",
            "enhance_params": {
                "clip_limit": 2.0,
                "tile_grid_size": (8, 8)
            },
            "segmentation_params": {
                "confidence_threshold": 0.7,
                "target_classes": [5, 6]  # Reefs, Fish
            },
            "target_fps": 15
        },
        "深海遗迹": {
            "scene": "深海遗迹",
            "description": "适用于深水环境，低光照，高散射",
            "enhance_method": "histogram_equalization",
            "enhance_params": {
                "clip_limit": 3.0
            },
            "segmentation_params": {
                "confidence_threshold": 0.6,
                "target_classes": [3, 7]  # Wrecks, Sea floor
            },
            "target_fps": 10
        },
        "海洋生物监测": {
            "scene": "海洋生物监测",
            "description": "动态场景，需要高帧率",
            "enhance_method": "clahe",
            "enhance_params": {
                "clip_limit": 2.5,
                "tile_grid_size": (4, 4)  # 更快处理
            },
            "segmentation_params": {
                "confidence_threshold": 0.7,
                "target_classes": [1, 6]  # Divers, Fish
            },
            "target_fps": 30
        }
    }

    @classmethod
    def list_scenarios(cls) -> list:
        """列出所有场景"""
        return list(cls.SCENARIOS.keys())

    @classmethod
    def get_scenario(cls, scene_name: str) -> Dict[str, Any]:
        """
        获取场景配置

        Args:
            scene_name: 场景名称

        Returns:
            场景配置字典
        """
        return cls.SCENARIOS.get(scene_name, {})

    @classmethod
    def apply_scenario(cls, engine: InferenceEngine, scene_name: str):
        """
        应用场景配置到推理引擎

        Args:
            engine: 推理引擎
            scene_name: 场景名称
        """
        scenario = cls.get_scenario(scene_name)

        if not scenario:
            raise ValueError(f"未知场景: {scene_name}")

        # 这里可以根据场景调整模型参数
        # 例如调整增强方法、置信度阈值等

        print(f"[ScenarioManager] Applied scenario: {scene_name}")
        print(f"  Description: {scenario['description']}")
        print(f"  Enhance method: {scenario['enhance_method']}")
        print(f"  Target FPS: {scenario['target_fps']}")


def create_inference_engine(device='auto') -> InferenceEngine:
    """
    创建推理引擎

    Args:
        device: 设备

    Returns:
        InferenceEngine实例
    """
    return InferenceEngine(device=device)


if __name__ == '__main__':
    # 测试推理引擎
    print("Testing InferenceEngine...")

    engine = create_inference_engine(device='cpu')

    print("\nModel Info:")
    for k, v in engine.get_model_info().items():
        print(f"  {k}: {v}")

    print("\nScenarios:")
    for scenario in ScenarioManager.list_scenarios():
        print(f"  - {scenario}")

    print("\n[OK] InferenceEngine test completed")
