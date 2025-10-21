"""
运行跨项目实验并进行模型间比较的完整脚本
这个脚本会为两个模型分别运行跨项目实验，然后进行统计比较
"""

import os
import sys
import subprocess


def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"命令: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n错误: {description} 失败")
        return False

    print(f"\n{description} 完成")
    return True


def main():
    # 要使用的项目列表（越多越好，建议5+个）
    projects = ["django", "react", "tensorflow", "moby", "opencv"]

    print("=" * 70)
    print("跨项目实验 + 假设检验完整流程")
    print("=" * 70)
    print(f"\n使用的项目: {', '.join(projects)}")
    print(f"每个模型将在 {len(projects)} 个项目上测试")
    print(f"产生 {len(projects)} 个独立样本用于统计检验\n")

    # 步骤1: MMoE 跨项目实验
    if not os.path.exists("outputs/cross_project_mmoe/cross_project_summary.json"):
        print("\n步骤 1/4: 运行 MMoE 跨项目实验")
        if not run_command(
            [
                sys.executable,
                "experiments/cross_project.py",
                "--config",
                "configs/django_real_mmoe.yaml",
                "--output",
                "outputs/cross_project_mmoe",
                "--projects",
            ]
            + projects,
            "MMoE 跨项目实验",
        ):
            return
    else:
        print("\n步骤 1/4: MMoE 跨项目实验 [已缓存，跳过]")

    # 步骤2: Shared-Bottom 跨项目实验
    if not os.path.exists("outputs/cross_project_shared/cross_project_summary.json"):
        print("\n步骤 2/4: 运行 Shared-Bottom 跨项目实验")
        if not run_command(
            [
                sys.executable,
                "experiments/cross_project.py",
                "--config",
                "configs/django_real_shared.yaml",
                "--output",
                "outputs/cross_project_shared",
                "--projects",
            ]
            + projects,
            "Shared-Bottom 跨项目实验",
        ):
            return
    else:
        print("\n步骤 2/4: Shared-Bottom 跨项目实验 [已缓存，跳过]")

    # 步骤3: 比较 R2
    if not os.path.exists("outputs/hypothesis_test_cross_project/comparison_R2.json"):
        print("\n步骤 3/4: 假设检验 - 回归 R2")
        if not run_command(
            [
                sys.executable,
                "experiments/hypothesis_test.py",
                "--result_dirs",
                "outputs/cross_project_mmoe",
                "outputs/cross_project_shared",
                "--output",
                "outputs/hypothesis_test_cross_project",
                "--metric",
                "R2",
            ],
            "假设检验 - R2 (跨项目)",
        ):
            return
    else:
        print("\n步骤 3/4: 假设检验 - 回归 R2 [已缓存，跳过]")

    # 步骤4: 比较 F1
    if not os.path.exists("outputs/hypothesis_test_cross_project/comparison_F1.json"):
        print("\n步骤 4/4: 假设检验 - 分类 F1")
        if not run_command(
            [
                sys.executable,
                "experiments/hypothesis_test.py",
                "--result_dirs",
                "outputs/cross_project_mmoe",
                "outputs/cross_project_shared",
                "--output",
                "outputs/hypothesis_test_cross_project",
                "--metric",
                "F1",
            ],
            "假设检验 - F1 (跨项目)",
        ):
            return
    else:
        print("\n步骤 4/4: 假设检验 - 分类 F1 [已缓存，跳过]")

    print("\n" + "=" * 70)
    print("所有实验完成！")
    print("=" * 70)
    print("\n结果保存在:")
    print("  - MMoE 跨项目结果: outputs/cross_project_mmoe/")
    print("  - Shared-Bottom 跨项目结果: outputs/cross_project_shared/")
    print("  - 假设检验结果: outputs/hypothesis_test_cross_project/")
    print("\n假设检验输出文件:")
    print("  - comparison_R2.json (包含统计检验结果)")
    print("  - comparison_R2.png (可视化图表)")
    print("  - comparison_F1.json")
    print("  - comparison_F1.png")
    print(f"\n每个模型有 {len(projects)} 个独立样本，可以进行有效的统计检验！")


if __name__ == "__main__":
    main()
