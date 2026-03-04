"""
配置加载器

用于加载和管理项目配置。
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ConfigLoader:
    """
    配置加载器类

    支持YAML和JSON格式的配置文件。
    """

    def __init__(self, config_path: Union[str, Path] = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = {}

        if self.config_path and self.config_path.exists():
            self.load()

    def load(self, config_path: Union[str, Path] = None) -> Dict[str, Any]:
        """
        加载配置文件

        Args:
            config_path: 配置文件路径，如果为None则使用初始化时的路径

        Returns:
            配置字典
        """
        if config_path:
            self.config_path = Path(config_path)

        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        suffix = self.config_path.suffix.lower()

        if suffix in ['.yaml', '.yml']:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif suffix == '.json':
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {suffix}")

        return self.config

    def save(self, config_path: Union[str, Path] = None):
        """
        保存配置到文件

        Args:
            config_path: 保存路径，如果为None则使用原路径
        """
        save_path = Path(config_path) if config_path else self.config_path

        if not save_path:
            raise ValueError("未指定保存路径")

        suffix = save_path.suffix.lower()

        if suffix in ['.yaml', '.yml']:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True)
        elif suffix == '.json':
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的配置文件格式: {suffix}")

    def get(self, key: str, default=None):
        """获取配置项"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def set(self, key: str, value: Any):
        """设置配置项"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    @staticmethod
    def load_model_config(model_type: str = "segmentation") -> Dict[str, Any]:
        """
        加载模型配置

        Args:
            model_type: 模型类型 ("segmentation", "enhancement", "pipeline")

        Returns:
            模型配置字典
        """
        # 检测设备
        if HAS_TORCH:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cpu"

        # 默认配置
        default_configs = {
            "segmentation": {
                "use_mock": False,
                "model_type": "segmodel",  # 使用基础模型
                "num_classes": 8,
                "input_size": [256, 256],
                "weight_path": "checkpoints/trained/segmodel_best.pth",
                "device": device
            },
            "enhancement": {
                "use_mock": False,
                "enhance_method": "clahe"
            },
            "pipeline": {
                "use_mock": False,
                "model_type": "segmodel",
                "num_classes": 8,
                "weight_path": "checkpoints/trained/segmodel_best.pth",
                "device": device
            }
        }

        return default_configs.get(model_type, {})


# 全局配置实例
_global_config = ConfigLoader()


def get_global_config() -> ConfigLoader:
    """获取全局配置实例"""
    return _global_config
