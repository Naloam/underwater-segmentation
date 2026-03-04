"""
通信模块

包含串口通信和网络通信接口，用于与水下机器人交互。
"""

from .serial_interface import AUVSerialInterface, AUV_COMMANDS
from .network_interface import AUVNetworkInterface, VideoStreamReceiver

__all__ = [
    'AUVSerialInterface',
    'AUV_COMMANDS',
    'AUVNetworkInterface',
    'VideoStreamReceiver'
]
