"""
Demo测试脚本

测试PyQt5 Demo的各项功能。
"""

import sys
import unittest
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / '05_Shared'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestMockModels(unittest.TestCase):
    """测试Mock模型"""

    def setUp(self):
        from models.mock_models import MockSegmentor, MockEnhancer, MockPipeline
        self.segmentor = MockSegmentor(num_classes=8)
        self.enhancer = MockEnhancer()
        self.pipeline = MockPipeline(num_classes=8)

    def test_segmentor_forward(self):
        """测试分割模型前向传播"""
        import torch
        x = torch.randn(1, 3, 256, 256)
        output = self.segmentor(x)
        self.assertEqual(output.shape, (1, 8, 256, 256))

    def test_enhancer_forward(self):
        """测试增强模型前向传播"""
        import torch
        x = torch.randn(1, 3, 256, 256)
        output = self.enhancer(x)
        self.assertEqual(output.shape, x.shape)

    def test_pipeline_process(self):
        """测试流水线处理"""
        import numpy as np
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        enhanced, mask = self.pipeline.process(image)
        self.assertEqual(enhanced.shape, image.shape)
        self.assertEqual(mask.shape, (256, 256))


class TestInferenceEngine(unittest.TestCase):
    """测试推理引擎"""

    def setUp(self):
        from core.inference_engine import InferenceEngine
        self.engine = InferenceEngine(device='cpu')

    def test_process_random_image(self):
        """测试处理随机图像"""
        import numpy as np
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        enhanced, mask = self.engine.process(image)
        self.assertIsNotNone(enhanced)
        self.assertIsNotNone(mask)

    def test_benchmark(self):
        """测试性能测试"""
        import numpy as np
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        stats = self.engine.benchmark(image, num_runs=10)
        self.assertIn('fps', stats)
        self.assertGreater(stats['fps'], 0)


class TestScenarioManager(unittest.TestCase):
    """测试场景管理器"""

    def setUp(self):
        from core.inference_engine import ScenarioManager
        self.manager = ScenarioManager()

    def test_list_scenarios(self):
        """测试列出示例"""
        scenarios = self.manager.list_scenarios()
        self.assertIsInstance(scenarios, list)
        self.assertGreater(len(scenarios), 0)

    def test_get_scenario(self):
        """测试获取场景"""
        scenario = self.manager.get_scenario("浅海珊瑚礁")
        self.assertIsNotNone(scenario)
        self.assertIn('scene', scenario)


class TestVisualization(unittest.TestCase):
    """测试可视化功能"""

    def test_mask_to_color(self):
        """测试mask颜色转换"""
        from models.mock_models import mask_to_color_image
        import numpy as np

        mask = np.random.randint(0, 8, (256, 256), dtype=np.uint8)
        color = mask_to_color_image(mask)
        self.assertEqual(color.shape, (256, 256, 3))

    def test_overlay_mask(self):
        """测试mask叠加"""
        from models.mock_models import overlay_mask
        import numpy as np

        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.random.randint(0, 8, (256, 256), dtype=np.uint8)
        overlay = overlay_mask(image, mask, alpha=0.5)
        self.assertEqual(overlay.shape, image.shape)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestMockModels))
    suite.addTests(loader.loadTestsFromTestCase(TestInferenceEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestScenarioManager))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualization))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
