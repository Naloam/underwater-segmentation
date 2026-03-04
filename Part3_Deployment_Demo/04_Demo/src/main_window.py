"""
PyQt5 Demo - 水下图像增强+全景分割系统

主窗口界面
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QSlider,
    QTabWidget, QGroupBox, QScrollArea, QMessageBox, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
import numpy as np
from PIL import Image
import cv2

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / '05_Shared'))

from models.model_interface import ModelFactory
from common.config_loader import ConfigLoader


class ImageDisplayWidget(QLabel):
    """图像显示组件"""
    def __init__(self, title="", placeholder=True):
        super().__init__()
        self.title = title
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                background-color: #2a2a2a;
                color: #aaa;
                font-size: 14px;
            }
        """)
        if placeholder:
            self.setText(f"{title}\n(等待加载)")

    def update_image(self, image):
        """更新显示图像"""
        if image is None:
            return

        # 转换图像格式
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # mask
                # 转换为彩色显示
                image = self._mask_to_color(image)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w = image.shape[:2]
        bytes_per_line = 3 * w

        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # 缩放到合适大小
        scaled_pixmap = pixmap.scaled(
            self.width() - 10, self.height() - 10,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.setPixmap(scaled_pixmap)

    def _mask_to_color(self, mask):
        """将mask转换为彩色图像"""
        h, w = mask.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)

        # SUIM类别颜色
        colors = [
            (0, 0, 0),       # Background
            (255, 0, 0),     # Divers
            (0, 255, 0),     # Plants
            (0, 0, 255),     # Wrecks
            (255, 255, 0),   # Robots
            (255, 0, 255),   # Reefs
            (0, 255, 255),   # Fish
            (128, 128, 128)  # Sea floor
        ]

        for class_id, color in enumerate(colors):
            color_img[mask == class_id] = color

        return cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_image_path = None
        self.segmentor = None
        self.enhancer = None
        self.processing = False

        self.init_ui()
        self.load_models()
        self.setup_scenarios()

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("水下图像增强+全景分割系统")
        self.setGeometry(100, 100, 1400, 800)

        # 设置暗色主题
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ddd;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5a8c;
            }
            QGroupBox {
                color: #ddd;
                font-weight: bold;
                border: 1px solid #444;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QComboBox {
                background-color: #2a2a2a;
                color: #ddd;
                border: 1px solid #444;
                padding: 5px;
            }
            QStatusBar {
                background-color: #0e0e0e;
                color: #ddd;
            }
        """)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # 左侧面板
        left_panel = self.create_left_panel()
        main_layout.addLayout(left_panel, 1)

        # 右侧面板
        right_panel = self.create_right_panel()
        main_layout.addLayout(right_panel, 1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def create_left_panel(self):
        """创建左侧面板"""
        layout = QVBoxLayout()

        # 输入图像组
        input_group = QGroupBox("输入图像")
        input_layout = QVBoxLayout()

        self.input_display = ImageDisplayWidget("原始图像")
        input_layout.addWidget(self.input_display)

        # 按钮行
        btn_layout = QGridLayout()
        self.load_btn = QPushButton("加载图像")
        self.load_video_btn = QPushButton("加载视频")
        self.camera_btn = QPushButton("摄像头")

        self.load_btn.clicked.connect(self.load_image)
        self.load_video_btn.clicked.connect(self.load_video)
        self.camera_btn.clicked.connect(self.start_camera)

        btn_layout.addWidget(self.load_btn, 0, 0)
        btn_layout.addWidget(self.load_video_btn, 0, 1)
        btn_layout.addWidget(self.camera_btn, 1, 0, 1, 2)
        input_layout.addLayout(btn_layout)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout()

        # 场景选择
        scene_layout = QHBoxLayout()
        scene_layout.addWidget(QLabel("场景预设:"))
        self.scene_combo = QComboBox()
        scene_layout.addWidget(self.scene_combo)
        control_layout.addLayout(scene_layout)

        # 增强方法
        enhance_layout = QHBoxLayout()
        enhance_layout.addWidget(QLabel("增强方法:"))
        self.enhance_combo = QComboBox()
        self.enhance_combo.addItems(["CLAHE", "直方图均衡", "无增强"])
        enhance_layout.addWidget(self.enhance_combo)
        control_layout.addLayout(enhance_layout)

        # 处理按钮
        self.process_btn = QPushButton("开始处理")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #2da042;
                font-size: 16px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #3cb553;
            }
        """)
        control_layout.addWidget(self.process_btn)

        # 保存按钮
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_results)
        control_layout.addWidget(self.save_btn)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        return layout

    def create_right_panel(self):
        """创建右侧面板"""
        layout = QVBoxLayout()

        # 输出选项卡
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #ddd;
                padding: 8px 16px;
            }
            QTabBar::tab:selected {
                background-color: #0e639c;
            }
        """)

        # 增强结果
        enhanced_tab = QWidget()
        enhanced_layout = QVBoxLayout()
        self.enhanced_display = ImageDisplayWidget("增强后图像")
        enhanced_layout.addWidget(self.enhanced_display)
        enhanced_tab.setLayout(enhanced_layout)
        self.tab_widget.addTab(enhanced_tab, "增强结果")

        # 分割结果
        segment_tab = QWidget()
        segment_layout = QVBoxLayout()
        self.segment_display = ImageDisplayWidget("分割结果")
        segment_layout.addWidget(self.segment_display)
        segment_tab.setLayout(segment_layout)
        self.tab_widget.addTab(segment_tab, "分割结果")

        # 叠加结果
        overlay_tab = QWidget()
        overlay_layout = QVBoxLayout()
        self.overlay_display = ImageDisplayWidget("叠加结果")
        overlay_layout.addWidget(self.overlay_display)
        overlay_tab.setLayout(overlay_layout)
        self.tab_widget.addTab(overlay_tab, "叠加结果")

        # 对比视图
        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout()
        self.comparison_display = ImageDisplayWidget("对比视图 (原始 | 增强 | 分割)")
        self.comparison_display.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                background-color: #2a2a2a;
                color: #aaa;
                font-size: 12px;
            }
        """)
        comparison_layout.addWidget(self.comparison_display)

        # 统计信息
        self.stats_label = QLabel("处理统计信息将显示在这里")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                padding: 10px;
                border-radius: 4px;
            }
        """)
        comparison_layout.addWidget(self.stats_label)

        comparison_tab.setLayout(comparison_layout)
        self.tab_widget.addTab(comparison_tab, "对比视图")

        layout.addWidget(self.tab_widget)

        return layout

    def setup_scenarios(self):
        """设置场景预设"""
        scenarios = {
            "浅海珊瑚礁": {
                "description": "适用于浅水环境，光线充足，色彩丰富",
                "enhance_method": "CLAHE",
                "enhance_params": {"clip_limit": 2.0}
            },
            "深海遗迹": {
                "description": "适用于深水环境，低光照，高散射",
                "enhance_method": "直方图均衡",
                "enhance_params": {"clip_limit": 3.0}
            },
            "海洋生物监测": {
                "description": "动态场景，需要高帧率",
                "enhance_method": "CLAHE",
                "enhance_params": {"clip_limit": 2.5}
            }
        }

        for scenario_name in scenarios.keys():
            self.scene_combo.addItem(scenario_name)

        self.scenarios = scenarios

    def load_models(self):
        """加载模型"""
        try:
            self.status_bar.showMessage("正在加载模型...")

            config = ConfigLoader.load_model_config("pipeline")
            self.pipeline = ModelFactory.create_pipeline(config)

            self.status_bar.showMessage("模型加载完成")
        except Exception as e:
            QMessageBox.warning(self, "模型加载失败", f"无法加载模型: {str(e)}")
            self.pipeline = None

    def load_image(self):
        """加载图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp)"
        )

        if file_path:
            self.current_image_path = file_path
            self.current_image = np.array(Image.open(file_path))

            # 显示原始图像
            self.input_display.update_image(self.current_image)
            self.status_bar.showMessage(f"已加载: {Path(file_path).name}")

    def load_video(self):
        """加载视频"""
        QMessageBox.information(self, "功能提示", "视频处理功能开发中...")

    def start_camera(self):
        """启动摄像头"""
        QMessageBox.information(self, "功能提示", "摄像头功能开发中...")

    def process_image(self):
        """处理图像"""
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return

        if self.pipeline is None:
            QMessageBox.warning(self, "警告", "模型未加载")
            return

        try:
            self.status_bar.showMessage("正在处理...")
            self.processing = True

            # 处理图像
            enhanced, mask = self.pipeline.process(self.current_image)

            # 显示结果
            self.enhanced_display.update_image(enhanced)
            self.segment_display.update_image(mask)

            # 创建叠加
            overlay = self.create_overlay(self.current_image, mask)
            self.overlay_display.update_image(overlay)

            # 创建对比视图
            comparison = self.create_comparison(self.current_image, enhanced, mask)
            self.comparison_display.update_image(comparison)

            # 统计信息
            unique, counts = np.unique(mask, return_counts=True)
            stats_text = "检测到的类别:\n"
            for cls, cnt in zip(unique, counts):
                pct = cnt / mask.size * 100
                stats_text += f"  类别{cls}: {cnt}像素 ({pct:.1f}%)\n"

            self.stats_label.setText(stats_text)

            self.status_bar.showMessage("处理完成")
            self.processing = False

        except Exception as e:
            QMessageBox.critical(self, "处理失败", f"错误: {str(e)}")
            self.status_bar.showMessage("处理失败")
            self.processing = False

    def create_overlay(self, image, mask):
        """创建叠加图像"""
        from models.real_models import mask_to_color_image

        color_mask = mask_to_color_image(mask)

        # 调整尺寸
        if color_mask.shape != image.shape:
            color_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]))

        overlay = (image * 0.5 + color_mask * 0.5).astype(np.uint8)
        return overlay

    def create_comparison(self, original, enhanced, mask):
        """创建对比视图 (三联图)"""
        from models.real_models import mask_to_color_image

        h, w = original.shape[:2]

        # 调整尺寸
        if enhanced.shape[:2] != (h, w):
            enhanced = cv2.resize(enhanced, (w, h))
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 转换mask为彩色
        color_mask = mask_to_color_image(mask)

        # 水平拼接
        comparison = np.hstack([original, enhanced, color_mask])

        # 添加标签
        cv2.putText(comparison, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Enhanced", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Segmented", (2 * w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return comparison

    def save_results(self):
        """保存结果"""
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return

        directory = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if directory:
            import uuid
            base_name = f"result_{uuid.uuid4().hex[:8]}"

            try:
                # 保存各个结果
                Path(directory, f"{base_name}_original.png").write_bytes(
                    cv2.imencode('.png', cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))[1].tobytes()
                )

                QMessageBox.information(self, "成功", f"结果已保存到:\n{directory}")
                self.status_bar.showMessage(f"已保存: {base_name}")

            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"错误: {str(e)}")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
