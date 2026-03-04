"""
串口通信接口

预留用于与水下机器人（AUV/ROV）的串口通信。
"""

import serial
import serial.tools.list_ports
from pathlib import Path
from typing import Optional, Callable, Any
import threading
import queue


class AUVSerialInterface:
    """
    水下机器人串口接口

    用于接收传感器数据和控制指令。
    """

    # 常用波特率
    BAUD_RATES = [9600, 19200, 38400, 57600, 115200]

    def __init__(self, port: str = None, baudrate: int = 115200):
        """
        初始化串口接口

        Args:
            port: 串口名称（如 'COM3' 或 '/dev/ttyUSB0'）
            baudrate: 波特率
        """
        self.port = port
        self.baudrate = baudrate
        self.connection: Optional[serial.Serial] = None
        self.is_connected = False

        # 数据队列
        self.rx_queue = queue.Queue()
        self.tx_queue = queue.Queue()

        # 回调函数
        self.on_data_received: Optional[Callable] = None

        # 接收线程
        self.rx_thread = None
        self.running = False

    @staticmethod
    def list_available_ports() -> list:
        """列出所有可用的串口"""
        ports = serial.tools.list_ports.comports()
        return [
            {
                'device': port.device,
                'description': port.description,
                'hwid': port.hwid
            }
            for port in ports
        ]

    def connect(self, port: str = None) -> bool:
        """
        连接串口

        Args:
            port: 串口名称

        Returns:
            是否连接成功
        """
        if port:
            self.port = port

        if not self.port:
            raise ValueError("未指定串口")

        try:
            self.connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            self.is_connected = True

            # 启动接收线程
            self.running = True
            self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
            self.rx_thread.start()

            print(f"串口已连接: {self.port}")
            return True

        except serial.SerialException as e:
            print(f"串口连接失败: {e}")
            return False

    def disconnect(self):
        """断开串口连接"""
        self.running = False

        if self.rx_thread:
            self.rx_thread.join(timeout=2)

        if self.connection and self.connection.is_open:
            self.connection.close()
            self.is_connected = False
            print("串口已断开")

    def send_data(self, data: bytes):
        """
        发送数据

        Args:
            data: 要发送的数据
        """
        if not self.is_connected:
            raise RuntimeError("串口未连接")

        self.connection.write(data)

    def send_command(self, command: str, args: dict = None):
        """
        发送控制命令

        Args:
            command: 命令名称
            args: 命令参数
        """
        import json

        message = {
            'cmd': command,
            'args': args or {}
        }

        # 编码为JSON并添加帧头帧尾
        json_str = json.dumps(message)
        data = json_str.encode('utf-8')

        # 简单的帧格式: [0xAA][0x55][长度][数据][校验]
        frame = bytearray()
        frame.extend([0xAA, 0x55])
        frame.append(len(data))
        frame.extend(data)
        frame.append(self._checksum(data))

        self.send_data(bytes(frame))

    def _rx_loop(self):
        """接收循环"""
        buffer = bytearray()

        while self.running:
            try:
                if self.connection.in_waiting > 0:
                    data = self.connection.read(self.connection.in_waiting)
                    buffer.extend(data)

                    # 解析帧
                    while len(buffer) >= 4:
                        # 查找帧头
                        if buffer[0] == 0xAA and buffer[1] == 0x55:
                            length = buffer[2]
                            total_length = 4 + length + 1

                            if len(buffer) >= total_length:
                                # 提取数据
                                data_bytes = bytes(buffer[4:4+length])
                                checksum = buffer[4+length]

                                # 验证校验和
                                if checksum == self._checksum(data_bytes):
                                    self._process_received_data(data_bytes)

                                # 移除已处理的帧
                                buffer = buffer[total_length:]
                                continue

                        # 帧头不匹配，移除第一个字节
                        buffer = buffer[1:]

            except Exception as e:
                print(f"接收错误: {e}")

    def _process_received_data(self, data: bytes):
        """处理接收到的数据"""
        try:
            import json
            message = json.loads(data.decode('utf-8'))

            # 放入队列
            self.rx_queue.put(message)

            # 触发回调
            if self.on_data_received:
                self.on_data_received(message)

        except Exception as e:
            print(f"数据处理错误: {e}")

    @staticmethod
    def _checksum(data: bytes) -> int:
        """计算校验和（XOR）"""
        checksum = 0
        for byte in data:
            checksum ^= byte
        return checksum


# 预定义的AUV命令
AUV_COMMANDS = {
    # 运动控制
    'MOVE_FORWARD': {'description': '前进', 'params': ['speed']},
    'MOVE_BACKWARD': {'description': '后退', 'params': ['speed']},
    'TURN_LEFT': {'description': '左转', 'params': ['angle']},
    'TURN_RIGHT': {'description': '右转', 'params': ['angle']},
    'DIVE': {'description': '下潜', 'params': ['depth']},
    'ASCEND': {'description': '上浮', 'params': ['depth']},
    'STOP': {'description': '停止', 'params': []},

    # 设备控制
    'LIGHTS_ON': {'description': '开启灯光', 'params': ['intensity']},
    'LIGHTS_OFF': {'description': '关闭灯光', 'params': []},
    'CAMERA_START': {'description': '启动相机', 'params': ['resolution']},
    'CAMERA_STOP': {'description': '停止相机', 'params': []},

    # 状态查询
    'GET_STATUS': {'description': '获取状态', 'params': []},
    'GET_BATTERY': {'description': '获取电量', 'params': []},
    'GET_DEPTH': {'description': '获取深度', 'params': []},
}


if __name__ == '__main__':
    print("串口通信接口模块已就绪")
    print("可用串口:", AUVSerialInterface.list_available_ports())
