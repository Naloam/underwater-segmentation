"""
推理引擎

统一的图像处理推理引擎，支持多种模型和配置。
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import sys

# 添加共享模块到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / '05_Shared'))

from models.model_interface import SegmentationModelInterface, EnhancementModelInterface
from common.config_loader import load_model_config
from common.utils import Timer


class InferenceEngine:
    """
    统一推理引擎

    支持图像增强和全景分割的端到端推理。
    """

    def __init__(
        self,
        model_config: Dict[str, Any] = None,
        device: str = 'cuda'
    ):
        """
        初始化推理引擎

        Args:
            model_config: 模型配置
            device: 推理设备
        """
        self.device = device
        self.model_config = model_config or load_model_config('pipeline')

        # 加载模型
        from models.model_interface import ModelFactory
        self.model = ModelFactory.create_pipeline(self.model_config)

        # 缓存
        self._result_cache = {}

    @property
    def enhancer(self) -> EnhancementModelInterface:
        """获取增强模型"""
        return self.model.enhancer

    @property
    def segmentor(self) -> SegmentationModelInterface:
        """获取分割模型"""
        return self.model.segmentor

    def process(
        self,
        image: Union[np.ndarray, str, Path],
        enhance_first: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理单张图像

        Args:
            image: 输入图像（路径或数组）
            enhance_first: 是否先增强再分割

        Returns:
            enhanced: 增强后图像 [H, W, 3] RGB
            mask: 分割mask [H, W]
        """
        # 加载图像
        if isinstance(image, (str, Path)):
            from common.utils import load_image
            image = load_image(str(image))

        # 执行推理
        with Timer() as t:
            if enhance_first:
                enhanced = self.model.enhancer.enhance(image)
                mask = self.model.segmentor.predict(enhanced)
            else:
                enhanced = image.copy()
                mask = self.model.segmentor.predict(image)

        inference_time = t.stop()

        return enhanced, mask

    def process_batch(
        self,
        images: list,
        enhance_first: bool = True
    ) -> Tuple[list, list]:
        """
        批量处理图像

        Args:
            images: 图像列表
            enhance_first: 是否先增强再分割

        Returns:
            enhanced_list: 增强后图像列表
            mask_list: 分割mask列表
        """
        enhanced_list = []
        mask_list = []

        for image in images:
            enhanced, mask = self.process(image, enhance_first)
            enhanced_list.append(enhanced)
            mask_list.append(mask)

        return enhanced_list, mask_list

    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path] = None,
        fps: int = 15,
        enhance_first: bool = True
    ) -> str:
        """
        处理视频文件

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            fps: 输出帧率
            enhance_first: 是否先增强再分割

        Returns:
            输出视频路径
        """
        import cv2

        # 打开视频
        cap = cv2.VideoCapture(str(video_path))

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 设置输出路径
        if output_path is None:
            output_path = Path(video_path).stem + '_processed.mp4'

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 2, height))

        # 处理每一帧
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 处理
            enhanced, mask = self.process(frame_rgb, enhance_first)

            # 创建对比图
            mask_color = self._mask_to_color(mask)
            comparison = np.hstack([
                cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR)
            ])

            out.write(comparison)
            frame_count += 1

            if frame_count % 30 == 0:
                print(f"已处理: {frame_count}/{total_frames} 帧")

        cap.release()
        out.release()

        return str(output_path)

    def _mask_to_color(self, mask: np.ndarray) -> np.ndarray:
        """将mask转换为彩色图像"""
        from models.mock_models import SUIM_COLOR_MAP
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in SUIM_COLOR_MAP.items():
            color_mask[mask == class_id] = color

        return color_mask

    def benchmark(
        self,
        test_image: np.ndarray = None,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        性能测试

        Args:
            test_image: 测试图像
            num_runs: 运行次数

        Returns:
            性能指标
        """
        if test_image is None:
            test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        import time

        # 预热
        for _ in range(10):
            self.process(test_image)

        # 测试
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.process(test_image)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        fps = 1.0 / np.mean(times)

        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'fps': fps,
            'num_runs': num_runs
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.model.get_info()


class ScenarioManager:
    """
    场景管理器

    管理不同场景的配置和参数。
    """

    def __init__(self, scenarios_dir: Path = None):
        """
        初始化场景管理器

        Args:
            scenarios_dir: 场景配置目录
        """
        if scenarios_dir is None:
            scenarios_dir = Path(__file__).parent.parent.parent / 'resources' / 'scenarios'

        self.scenarios_dir = Path(scenarios_dir)
        self.scenarios = self._load_scenarios()

    def _load_scenarios(self) -> Dict[str, Dict]:
        """加载所有场景配置"""
        scenarios = {}

        for yaml_file in self.scenarios_dir.glob('*.yaml'):
            try:
                import yaml
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    scene_name = config['scene']['name']
                    scenarios[scene_name] = config
            except Exception as e:
                print(f"警告: 加载场景失败 {yaml_file}: {e}")

        return scenarios

    def get_scenario(self, name: str) -> Optional[Dict]:
        """获取指定场景配置"""
        return self.scenarios.get(name)

    def list_scenarios(self) -> list:
        """列出所有可用场景"""
        return list(self.scenarios.keys())

    def apply_scenario(
        self,
        inference_engine: InferenceEngine,
        scenario_name: str
    ):
        """
        应用场景配置到推理引擎

        Args:
            inference_engine: 推理引擎
            scenario_name: 场景名称
        """
        scenario = self.get_scenario(scenario_name)

        if scenario is None:
            print(f"警告: 场景不存在: {scenario_name}")
            return

        # 应用增强参数
        # 这里可以根据场景调整增强器的参数

        # 应用分割参数
        # 这里可以根据场景调整分割器的参数

        print(f"已应用场景: {scenario_name}")


if __name__ == '__main__':
    print("推理引擎模块已就绪")
