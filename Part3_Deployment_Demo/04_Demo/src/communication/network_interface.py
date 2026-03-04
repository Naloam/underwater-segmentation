"""
网络通信接口

支持TCP/UDP通信，用于与水下机器人进行数据传输。
"""

import socket
import threading
import queue
import json
from typing import Optional, Callable, Any
from pathlib import Path


class AUVNetworkInterface:
    """
    水下机器人网络接口

    支持TCP和UDP协议，用于接收实时视频流和控制指令。
    """

    def __init__(
        self,
        protocol: str = 'tcp',
        host: str = 'localhost',
        port: int = 8888
    ):
        """
        初始化网络接口

        Args:
            protocol: 协议类型 ('tcp' 或 'udp')
            host: 主机地址
            port: 端口号
        """
        self.protocol = protocol.lower()
        self.host = host
        self.port = port

        self.socket: Optional[socket.socket] = None
        self.is_connected = False

        # 数据队列
        self.rx_queue = queue.Queue()
        self.tx_queue = queue.Queue()

        # 接收线程
        self.rx_thread = None
        self.running = False

        # 回调函数
        self.on_data_received: Optional[Callable] = None
        self.on_video_frame: Optional[Callable] = None

    def connect(self, host: str = None, port: int = None) -> bool:
        """
        连接到服务器

        Args:
            host: 主机地址
            port: 端口号

        Returns:
            是否连接成功
        """
        if host:
            self.host = host
        if port:
            self.port = port

        try:
            # 创建socket
            if self.protocol == 'tcp':
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            else:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            self.socket.connect((self.host, self.port))
            self.is_connected = True

            # 启动接收线程
            self.running = True
            self.rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
            self.rx_thread.start()

            print(f"已连接到 {self.host}:{self.port} ({self.protocol.upper()})")
            return True

        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        self.running = False

        if self.rx_thread:
            self.rx_thread.join(timeout=2)

        if self.socket:
            self.socket.close()
            self.is_connected = False
            print("网络连接已断开")

    def send_data(self, data: bytes):
        """
        发送数据

        Args:
            data: 要发送的数据
        """
        if not self.is_connected:
            raise RuntimeError("未连接")

        self.socket.sendall(data)

    def send_json(self, message: dict):
        """
        发送JSON消息

        Args:
            message: 消息字典
        """
        json_str = json.dumps(message)
        data = json_str.encode('utf-8')

        # 添加长度头（4字节大端序）
        header = len(data).to_bytes(4, byteorder='big')

        self.send_data(header + data)

    def send_command(self, command: str, **kwargs):
        """
        发送控制命令

        Args:
            command: 命令名称
            **kwargs: 命令参数
        """
        message = {
            'type': 'command',
            'command': command,
            'params': kwargs
        }
        self.send_json(message)

    def _rx_loop(self):
        """接收循环"""
        if self.protocol == 'tcp':
            self._tcp_rx_loop()
        else:
            self._udp_rx_loop()

    def _tcp_rx_loop(self):
        """TCP接收循环"""
        while self.running:
            try:
                # 接收消息长度
                header = self._recv_exact(4)
                if not header:
                    break

                length = int.from_bytes(header, byteorder='big')

                # 接收消息体
                data = self._recv_exact(length)
                if not data:
                    break

                # 处理消息
                self._process_message(data)

            except Exception as e:
                print(f"TCP接收错误: {e}")
                break

    def _udp_rx_loop(self):
        """UDP接收循环"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(65536)
                self._process_message(data)

            except Exception as e:
                print(f"UDP接收错误: {e}")

    def _recv_exact(self, length: int) -> Optional[bytes]:
        """接收指定长度的数据"""
        data = bytearray()
        while len(data) < length:
            chunk = self.socket.recv(length - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def _process_message(self, data: bytes):
        """处理接收到的消息"""
        try:
            message = json.loads(data.decode('utf-8'))

            msg_type = message.get('type', 'data')

            if msg_type == 'data':
                self.rx_queue.put(message)

                if self.on_data_received:
                    self.on_data_received(message)

            elif msg_type == 'video_frame':
                # 视频帧（base64编码）
                if self.on_video_frame:
                    self.on_video_frame(message)

        except Exception as e:
            print(f"消息处理错误: {e}")

    def start_server(self):
        """
        启动服务器模式

        等待水下机器人连接。
        """
        if self.protocol == 'tcp':
            self._start_tcp_server()
        else:
            self._start_udp_server()

    def _start_tcp_server(self):
        """启动TCP服务器"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)

        print(f"TCP服务器启动，监听 {self.host}:{self.port}")

        # 等待连接
        conn, addr = self.socket.accept()
        self.socket = conn
        self.is_connected = True

        print(f"客户端已连接: {addr}")

        # 启动接收线程
        self.running = True
        self.rx_thread = threading.Thread(target=self._tcp_rx_loop, daemon=True)
        self.rx_thread.start()

    def _start_udp_server(self):
        """启动UDP服务器"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))

        self.is_connected = True
        print(f"UDP服务器启动，监听 {self.host}:{self.port}")

        # 启动接收线程
        self.running = True
        self.rx_thread = threading.Thread(target=self._udp_rx_loop, daemon=True)
        self.rx_thread.start()


class VideoStreamReceiver:
    """
    视频流接收器

    接收来自水下机器人的实时视频流。
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        """
        初始化视频流接收器

        Args:
            host: 绑定地址
            port: 端口号
        """
        self.host = host
        self.port = port
        self.running = False

        # 回调函数
        self.on_frame: Optional[Callable] = None

    def start(self):
        """启动接收"""
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))

        self.running = True
        print(f"视频流接收器启动，监听 {self.host}:{self.port}")

        buffer = bytearray()
        while self.running:
            try:
                data, addr = sock.recvfrom(65536)
                buffer.extend(data)

                # 查找JPEG帧结束标记
                jpeg_end = buffer.find(b'\xff\xd9')
                if jpeg_end != -1:
                    jpeg_start = buffer.find(b'\xff\xd8')
                    if jpeg_start != -1:
                        # 提取JPEG帧
                        jpeg_data = bytes(buffer[jpeg_start:jpeg_end+2])
                        buffer = buffer[jpeg_end+2:]

                        # 触发回调
                        if self.on_frame:
                            self.on_frame(jpeg_data)

            except Exception as e:
                if self.running:
                    print(f"接收错误: {e}")

        sock.close()

    def stop(self):
        """停止接收"""
        self.running = False


if __name__ == '__main__':
    print("网络通信接口模块已就绪")
