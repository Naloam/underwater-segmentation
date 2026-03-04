"""
嵌入式部署模块

包含Jetson部署相关工具和模拟器。
"""

from .deployment_simulator import JetsonSimulator, TEST_SCENARIOS

__all__ = ['JetsonSimulator', 'TEST_SCENARIOS']
