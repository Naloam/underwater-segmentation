"""
Demo主程序

水下图像增强 + 全景分割系统的入口点。
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# 添加共享模块到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / '05_Shared'))

from ui.main_window import MainWindow


def main():
    """主函数"""
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("水下图像增强+全景分割系统")
    app.setApplicationVersion("1.0.0")

    # 设置应用样式
    app.setStyle("Fusion")

    # 设置全局字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    # 创建并显示主窗口
    window = MainWindow()
    window.show()

    # 运行应用
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
