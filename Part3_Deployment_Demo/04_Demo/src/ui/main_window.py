"""
PyQt5 Demo - 主窗口

水下图像增强 + 全景分割系统的主界面。
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox,
    QSlider, QGroupBox, QStatusBar, QProgressBar,
    QMessageBox, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont
from typing import Optional

# 添加共享模块到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / '05_Shared'))

from common.utils import load_image, create_comparison_grid


class ProcessingThread(QThread):
    """图像处理线程"""
    finished = pyqtSignal(np.ndarray, np.ndarray)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, image: np.ndarray, model):
        super().__init__()
        self.image = image
        self.model = model

    def run(self):
        try:
            self.progress.emit(10)

            # 图像增强
            enhanced = self.model.enhance.enhance(self.image)
            self.progress.emit(50)

            # 图像分割
            mask = self.model.segmentor.predict(enhanced)
            self.progress.emit(90)

            self.finished.emit(enhanced, mask)
            self.progress.emit(100)
        except Exception as e:
            self.error.emit(str(e))


class ImageDisplayWidget(QLabel):
    """图像显示组件"""

    def __init__(self, title: str = "Image"):
        super().__init__()
        self.title = title
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(450, 350)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                border-radius: 5px;
                background-color: #2b2b2b;
                color: #888;
                font-size: 14px;
            }
        """)
        self.setText(f"{title}\n(暂无图像)")

    def display_image(self, image: np.ndarray):
        """显示图像"""
        if image is None:
            return

        # 转换为QPixmap
        if len(image.shape) == 3:
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            h, w = image.shape
            q_image = QImage(image.data, w, h, w, QImage.Format_Grayscale8)

        # 缩放到合适大小
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.width() - 10, self.height() - 30,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.setPixmap(scaled_pixmap)


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_enhanced = None
        self.current_mask = None
        self.processing_thread = None

        # 加载模型
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / '05_Shared'))
        from models.mock_models import MockPipeline
        self.model = MockPipeline(num_classes=8)

        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("水下图像增强 + 全景分割系统")
        self.setGeometry(100, 100, 1400, 750)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧：输入区域
        left_panel = self.create_input_panel()
        main_layout.addLayout(left_panel, stretch=1)

        # 中间：控制面板
        center_panel = self.create_control_panel()
        main_layout.addLayout(center_panel, stretch=0)

        # 右侧：输出区域
        right_panel = self.create_output_panel()
        main_layout.addLayout(right_panel, stretch=1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def create_input_panel(self) -> QVBoxLayout:
        """创建输入面板"""
        layout = QVBoxLayout()

        # 标题
        title = QLabel("输入图像")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)

        # 图像显示
        self.input_display = ImageDisplayWidget("原始图像")
        layout.addWidget(self.input_display)

        # 加载按钮
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("加载图像")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.load_btn.clicked.connect(self.load_image)
        btn_layout.addWidget(self.load_btn)
        layout.addLayout(btn_layout)

        # 加载示例按钮
        self.load_sample_btn = QPushButton("加载示例")
        self.load_sample_btn.clicked.connect(self.load_sample_image)
        layout.addWidget(self.load_sample_btn)

        layout.addStretch()
        return layout

    def create_output_panel(self) -> QVBoxLayout:
        """创建输出面板"""
        layout = QVBoxLayout()

        # 使用标签页
        tab_widget = QTabWidget()

        # 增强结果标签页
        enhanced_tab = QWidget()
        enhanced_layout = QVBoxLayout(enhanced_tab)
        enhanced_title = QLabel("增强结果")
        enhanced_title.setFont(QFont("Arial", 12, QFont.Bold))
        enhanced_layout.addWidget(enhanced_title)
        self.enhanced_display = ImageDisplayWidget("增强后图像")
        enhanced_layout.addWidget(self.enhanced_display)
        tab_widget.addTab(enhanced_tab, "增强")

        # 分割结果标签页
        mask_tab = QWidget()
        mask_layout = QVBoxLayout(mask_tab)
        mask_title = QLabel("分割结果")
        mask_title.setFont(QFont("Arial", 12, QFont.Bold))
        mask_layout.addWidget(mask_title)
        self.mask_display = ImageDisplayWidget("分割Mask")
        mask_layout.addWidget(self.mask_display)
        tab_widget.addTab(mask_tab, "分割")

        # 叠加结果标签页
        overlay_tab = QWidget()
        overlay_layout = QVBoxLayout(overlay_tab)
        overlay_title = QLabel("叠加结果")
        overlay_title.setFont(QFont("Arial", 12, QFont.Bold))
        overlay_layout.addWidget(overlay_title)
        self.overlay_display = ImageDisplayWidget("叠加显示")
        overlay_layout.addWidget(self.overlay_display)
        tab_widget.addTab(overlay_tab, "叠加")

        layout.addWidget(tab_widget)

        # 保存按钮
        self.save_btn = QPushButton("保存结果")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_results)
        layout.addWidget(self.save_btn)

        layout.addStretch()
        return layout

    def create_control_panel(self) -> QVBoxLayout:
        """创建控制面板"""
        layout = QVBoxLayout()
        layout.setSpacing(15)

        # 场景选择
        scene_group = QGroupBox("场景预设")
        scene_layout = QVBoxLayout()
        self.scene_combo = QComboBox()
        self.scene_combo.addItems([
            "浅海珊瑚礁",
            "深海遗迹",
            "海洋生物监测"
        ])
        scene_layout.addWidget(self.scene_combo)
        scene_group.setLayout(scene_layout)
        layout.addWidget(scene_group)

        # 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "完整流水线",
            "仅增强",
            "仅分割"
        ])
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # 参数调整
        param_group = QGroupBox("参数调整")
        param_layout = QVBoxLayout()

        # 增强强度
        enh_label = QLabel("增强强度")
        param_layout.addWidget(enh_label)
        self.enh_slider = QSlider(Qt.Horizontal)
        self.enh_slider.setRange(0, 100)
        self.enh_slider.setValue(50)
        param_layout.addWidget(self.enh_slider)

        # 分割阈值
        seg_label = QLabel("分割阈值")
        param_layout.addWidget(seg_label)
        self.seg_slider = QSlider(Qt.Horizontal)
        self.seg_slider.setRange(0, 100)
        self.seg_slider.setValue(70)
        param_layout.addWidget(self.seg_slider)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 处理按钮
        self.process_btn = QPushButton("开始处理")
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 15px 25px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.process_btn.clicked.connect(self.process_image)
        layout.addWidget(self.process_btn)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        return layout

    def load_image(self):
        """加载图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp)"
        )

        if file_path:
            try:
                self.current_image = load_image(file_path)
                self.input_display.display_image(self.current_image)
                self.process_btn.setEnabled(True)
                self.status_bar.showMessage(f"已加载: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图像失败: {str(e)}")

    def load_sample_image(self):
        """加载示例图像"""
        # 从数据集中加载示例图像
        sample_path = Path("d:/myProjects/大创(1)/SUIM_Processed/SUIM_Processed/1_raw")
        sample_files = list(sample_path.glob("*.jpg"))

        if sample_files:
            try:
                self.current_image = load_image(sample_files[0])
                self.input_display.display_image(self.current_image)
                self.process_btn.setEnabled(True)
                self.status_bar.showMessage(f"已加载示例: {sample_files[0].name}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载示例失败: {str(e)}")
        else:
            QMessageBox.warning(self, "警告", "未找到示例图像")

    def process_image(self):
        """处理图像"""
        if self.current_image is None:
            return

        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("处理中...")

        # 启动处理线程
        self.processing_thread = ProcessingThread(self.current_image, self.model)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.progress.connect(self.progress_bar.setValue)
        self.processing_thread.start()

    def on_processing_finished(self, enhanced: np.ndarray, mask: np.ndarray):
        """处理完成回调"""
        self.current_enhanced = enhanced
        self.current_mask = mask

        # 显示结果
        self.enhanced_display.display_image(enhanced)
        self.mask_display.display_image(self._mask_to_color(mask))

        # 生成叠加图
        from models.mock_models import overlay_mask
        overlay = overlay_mask(enhanced, mask, alpha=0.4)
        self.overlay_display.display_image(overlay)

        # 更新UI状态
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.save_btn.setEnabled(True)
        self.status_bar.showMessage("处理完成!")

    def on_processing_error(self, error_msg: str):
        """处理错误回调"""
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "错误", f"处理失败: {error_msg}")
        self.status_bar.showMessage("处理失败")

    def _mask_to_color(self, mask: np.ndarray) -> np.ndarray:
        """将mask转换为彩色图像"""
        from models.mock_models import SUIM_COLOR_MAP
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in SUIM_COLOR_MAP.items():
            color_mask[mask == class_id] = color

        return color_mask

    def save_results(self):
        """保存结果"""
        if self.current_enhanced is None:
            return

        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if save_dir:
            try:
                base_name = "result"
                # 保存增强图
                enhanced_path = Path(save_dir) / f"{base_name}_enhanced.png"
                cv2.imwrite(str(enhanced_path), cv2.cvtColor(self.current_enhanced, cv2.COLOR_RGB2BGR))

                # 保存分割mask
                mask_path = Path(save_dir) / f"{base_name}_mask.png"
                cv2.imwrite(str(mask_path), self.current_mask)

                # 保存叠加图
                from models.mock_models import overlay_mask
                overlay = overlay_mask(self.current_enhanced, self.current_mask, alpha=0.4)
                overlay_path = Path(save_dir) / f"{base_name}_overlay.png"
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                QMessageBox.information(self, "成功", f"结果已保存到:\n{save_dir}")
                self.status_bar.showMessage("结果已保存")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
