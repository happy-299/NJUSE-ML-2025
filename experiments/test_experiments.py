"""
实验代码单元测试 - 验证基本功能
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.hypothesis_testing import HypothesisTest


class TestHypothesisTesting(unittest.TestCase):
    """假设检验工具测试"""

    def setUp(self):
        """准备测试数据"""
        # 模拟 3 个算法在 5 个数据集上的性能（R2 分数）
        np.random.seed(42)
        self.data = np.array(
            [
                [0.85, 0.80, 0.75],  # 数据集 1
                [0.90, 0.88, 0.82],  # 数据集 2
                [0.78, 0.76, 0.70],  # 数据集 3
                [0.88, 0.85, 0.80],  # 数据集 4
                [0.82, 0.79, 0.74],  # 数据集 5
            ]
        )
        self.algorithm_names = ["Algorithm A", "Algorithm B", "Algorithm C"]

    def test_friedman_test(self):
        """测试 Friedman test"""
        stat, p_value = HypothesisTest.friedman_test(self.data)

        # 检查返回值类型
        self.assertIsInstance(stat, (int, float))
        self.assertIsInstance(p_value, (int, float))

        # p 值应在 [0, 1] 范围内
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

        print(f"\nFriedman Test: statistic={stat:.4f}, p-value={p_value:.6f}")

    def test_nemenyi_test(self):
        """测试 Nemenyi test"""
        p_matrix, mean_ranks, cd = HypothesisTest.nemenyi_test(self.data, alpha=0.05)

        # 检查形状
        n_algorithms = self.data.shape[1]
        self.assertEqual(p_matrix.shape, (n_algorithms, n_algorithms))
        self.assertEqual(len(mean_ranks), n_algorithms)

        # 对角线应为 1（自己与自己比较）
        np.testing.assert_array_equal(np.diag(p_matrix), np.ones(n_algorithms))

        # 矩阵应对称
        np.testing.assert_array_almost_equal(p_matrix, p_matrix.T)

        # CD 应为正数
        self.assertGreater(cd, 0)

        print(f"\nNemenyi Test: CD={cd:.4f}")
        print(f"Mean ranks: {mean_ranks}")

    def test_kruskal_wallis_test(self):
        """测试 Kruskal-Wallis test"""
        stat, p_value = HypothesisTest.kruskal_wallis_test(self.data)

        self.assertIsInstance(stat, (int, float))
        self.assertIsInstance(p_value, (int, float))
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

        print(f"\nKruskal-Wallis Test: statistic={stat:.4f}, p-value={p_value:.6f}")

    def test_dunn_test(self):
        """测试 Dunn's test"""
        p_matrix = HypothesisTest.dunn_test(self.data, alpha=0.05)

        n_algorithms = self.data.shape[1]
        self.assertEqual(p_matrix.shape, (n_algorithms, n_algorithms))

        # 对角线应为 1
        np.testing.assert_array_equal(np.diag(p_matrix), np.ones(n_algorithms))

        # 矩阵应对称
        np.testing.assert_array_almost_equal(p_matrix, p_matrix.T)

        print(f"\nDunn's Test: pairwise p-values computed")


class TestAblationConfig(unittest.TestCase):
    """消融实验配置测试"""

    def test_variants_defined(self):
        """测试消融变体是否定义"""
        from experiments.ablation_study import AblationConfig

        variants = AblationConfig.VARIANTS

        # 至少应该有 full 变体
        self.assertIn("full", variants)

        # 每个变体应该有必要的字段
        for name, config in variants.items():
            self.assertIn("name", config)
            self.assertIn("description", config)
            self.assertIn("modifications", config)

        print(f"\n定义的消融变体: {list(variants.keys())}")


class TestCrossProjectConfig(unittest.TestCase):
    """跨项目实验配置测试"""

    def test_scenarios_defined(self):
        """测试实验场景是否定义"""
        from experiments.cross_project import CrossProjectConfig

        scenarios = CrossProjectConfig.SCENARIOS

        # 应该有基本场景
        expected_scenarios = ["within_project", "cross_project", "mixed_training"]
        for scenario in expected_scenarios:
            self.assertIn(scenario, scenarios)

        print(f"\n定义的实验场景: {list(scenarios.keys())}")


def run_basic_tests():
    """运行基础测试"""
    print("=" * 60)
    print("运行实验代码单元测试")
    print("=" * 60)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestHypothesisTesting))
    suite.addTests(loader.loadTestsFromTestCase(TestAblationConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossProjectConfig))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回是否成功
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)
