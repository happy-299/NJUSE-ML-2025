"""
运行所有level2实验的统一脚本
包括: 消融实验、跨项目实验、假设检验
"""

import os
import sys
import subprocess
import argparse


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


def output_has_cache(output_dir: str, markers: list) -> bool:
    """检查输出目录中是否存在用于判定已完成的标志文件

    Args:
        output_dir: 输出目录路径
        markers: 标志文件相对路径列表（相对于 output_dir ）

    Returns:
        True 如果所有标志文件存在
    """
    if not os.path.exists(output_dir):
        return False
    for m in markers:
        if not os.path.exists(os.path.join(output_dir, m)):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="运行所有level2实验")
    parser.add_argument("--skip_ablation", action="store_true", help="跳过消融实验")
    parser.add_argument(
        "--skip_cross_project", action="store_true", help="跳过跨项目实验"
    )
    parser.add_argument("--skip_hypothesis", action="store_true", help="跳过假设检验")
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/django_real_mmoe.yaml",
        help="基础配置文件路径",
    )

    args = parser.parse_args()

    # 1. 消融实验
    if not args.skip_ablation:
        print("\n" + "=" * 70)
        print("第一步: 模型组件消融实验")
        print("=" * 70)

        # MMoE消融实验 - 若已有缓存则跳过
        ablation_mmoe_dir = "outputs/ablation_mmoe"
        ablation_mmoe_markers = ["ablation_summary.json"]
        if output_has_cache(ablation_mmoe_dir, ablation_mmoe_markers):
            print(f"发现消融实验缓存，跳过 MMoE 消融实验（{ablation_mmoe_dir}）")
        else:
            if not run_command(
                [
                    sys.executable,
                    "experiments/ablation_study.py",
                    "--config",
                    args.base_config,
                    "--output",
                    ablation_mmoe_dir,
                    "--model_type",
                    "mmoe",
                ],
                "MMoE 消融实验",
            ):
                return

        # SharedBottom消融实验
        config_shared = args.base_config.replace("mmoe", "shared")
        ablation_shared_dir = "outputs/ablation_shared"
        ablation_shared_markers = ["ablation_summary.json"]
        if os.path.exists(config_shared):
            if output_has_cache(ablation_shared_dir, ablation_shared_markers):
                print(
                    f"发现消融实验缓存，跳过 SharedBottom 消融实验（{ablation_shared_dir}）"
                )
            else:
                if not run_command(
                    [
                        sys.executable,
                        "experiments/ablation_study.py",
                        "--config",
                        config_shared,
                        "--output",
                        ablation_shared_dir,
                        "--model_type",
                        "shared_bottom",
                    ],
                    "SharedBottom 消融实验",
                ):
                    return

    # 2. 跨项目泛化性演示实验（单个模型）
    if not args.skip_cross_project:
        print("\n" + "=" * 70)
        print("第二步: 跨项目泛化性演示实验（单个模型）")
        print("=" * 70)
        print("说明: 使用单个模型演示跨项目泛化能力")

        cross_project_dir = "outputs/cross_project"
        # 检查是否有任何跨项目实验结果文件（使用通配符模式）
        cross_project_markers = ["train_"]  # 简化的标记检查

        # 更灵活的缓存检查：只要目录存在且有train_开头的json文件即可
        has_cache = False
        if os.path.exists(cross_project_dir):
            try:
                files = os.listdir(cross_project_dir)
                has_cache = any(
                    f.startswith("train_") and f.endswith(".json") for f in files
                )
            except:
                pass

        if has_cache:
            print(f"发现跨项目泛化性实验缓存，跳过（{cross_project_dir}）")
        else:
            # 使用3个项目进行演示性跨项目实验
            if not run_command(
                [
                    sys.executable,
                    "experiments/cross_project.py",
                    "--config",
                    args.base_config,
                    "--output",
                    cross_project_dir,
                    "--projects",
                    "django",
                    "react",
                    "tensorflow",
                ],
                "跨项目泛化性实验",
            ):
                return

    # 3. 跨项目模型比较实验（level2重点）
    if not args.skip_cross_project:
        print("\n" + "=" * 70)
        print("第三步: 跨项目模型比较实验")
        print("=" * 70)
        print("说明: 运行两个模型的跨项目实验，然后进行假设检验比较")

        # 要使用的项目列表（与run_cross_project_comparison.py保持一致）
        projects = ["django", "react", "tensorflow", "moby", "opencv"]
        print(f"使用的项目: {', '.join(projects)}")
        print(f"每个模型将在 {len(projects)} 个项目上测试（leave-one-out）")

        # 3.1 MMoE 跨项目实验
        cross_project_mmoe_dir = "outputs/cross_project_mmoe"
        cross_project_mmoe_markers = [
            f"train_{'_'.join([p for p in projects if p != test])}_test_{test}.json"
            for test in projects
        ]

        if output_has_cache(cross_project_mmoe_dir, cross_project_mmoe_markers):
            print(f"\n发现 MMoE 跨项目实验缓存，跳过（{cross_project_mmoe_dir}）")
        else:
            print("\n子步骤 3.1: 运行 MMoE 跨项目实验")
            if not run_command(
                [
                    sys.executable,
                    "experiments/cross_project.py",
                    "--config",
                    "configs/django_real_mmoe.yaml",
                    "--output",
                    cross_project_mmoe_dir,
                    "--projects",
                ]
                + projects,
                "MMoE 跨项目实验",
            ):
                return

        # 3.2 Shared-Bottom 跨项目实验
        cross_project_shared_dir = "outputs/cross_project_shared"
        cross_project_shared_markers = [
            f"train_{'_'.join([p for p in projects if p != test])}_test_{test}.json"
            for test in projects
        ]

        if output_has_cache(cross_project_shared_dir, cross_project_shared_markers):
            print(
                f"\n发现 Shared-Bottom 跨项目实验缓存，跳过（{cross_project_shared_dir}）"
            )
        else:
            print("\n子步骤 3.2: 运行 Shared-Bottom 跨项目实验")
            if not run_command(
                [
                    sys.executable,
                    "experiments/cross_project.py",
                    "--config",
                    "configs/django_real_shared.yaml",
                    "--output",
                    cross_project_shared_dir,
                    "--projects",
                ]
                + projects,
                "Shared-Bottom 跨项目实验",
            ):
                return

    # 4. 假设检验
    if not args.skip_hypothesis:
        print("\n" + "=" * 70)
        print("第四步: 假设检验")
        print("=" * 70)

        # 检验1: 基于单项目的模型比较（如果有数据的话）
        result_dirs_single = ["outputs/django_real_mmoe", "outputs/django_real_shared"]
        hypothesis_dir_single = "outputs/hypothesis_test"
        hypothesis_markers_single = ["comparison_R2.json", "comparison_F1.json"]

        if output_has_cache(hypothesis_dir_single, hypothesis_markers_single):
            print(f"\n发现单项目假设检验缓存，跳过（{hypothesis_dir_single}）")
        else:
            if all(os.path.exists(d) for d in result_dirs_single):
                print(
                    "\n子步骤 4.1: 单项目模型比较（基于django_real_mmoe vs django_real_shared）"
                )
                print("警告: 单项目比较样本数量有限，统计效力较弱")

                # 回归指标检验
                if not run_command(
                    [sys.executable, "experiments/hypothesis_test.py", "--result_dirs"]
                    + result_dirs_single
                    + ["--output", hypothesis_dir_single, "--metric", "R2"],
                    "假设检验 - 单项目回归R2",
                ):
                    return

                # 分类指标检验
                if not run_command(
                    [sys.executable, "experiments/hypothesis_test.py", "--result_dirs"]
                    + result_dirs_single
                    + ["--output", hypothesis_dir_single, "--metric", "F1"],
                    "假设检验 - 单项目分类F1",
                ):
                    return
            else:
                print("注意: 未找到单项目结果目录，跳过单项目模型比较")

        # 检验2: 跨项目模型比较（推荐，样本量充足）
        cross_project_mmoe_dir = "outputs/cross_project_mmoe"
        cross_project_shared_dir = "outputs/cross_project_shared"
        hypothesis_dir_cross = "outputs/hypothesis_test_cross_project"
        hypothesis_markers_cross = ["comparison_R2.json", "comparison_F1.json"]

        if output_has_cache(hypothesis_dir_cross, hypothesis_markers_cross):
            print(f"\n发现跨项目假设检验缓存，跳过（{hypothesis_dir_cross}）")
        else:
            if os.path.exists(cross_project_mmoe_dir) and os.path.exists(
                cross_project_shared_dir
            ):
                print("\n子步骤 4.2: 跨项目模型比较（MMoE vs Shared-Bottom）")
                projects = ["django", "react", "tensorflow", "moby", "opencv"]
                print(f"基于 {len(projects)} 个测试项目的结果进行统计检验")
                print("这是推荐的假设检验方式，样本量充足，结论可靠")

                # 回归指标检验
                if not run_command(
                    [
                        sys.executable,
                        "experiments/hypothesis_test.py",
                        "--result_dirs",
                        cross_project_mmoe_dir,
                        cross_project_shared_dir,
                        "--output",
                        hypothesis_dir_cross,
                        "--metric",
                        "R2",
                    ],
                    "假设检验 - 跨项目回归R2（MMoE vs Shared-Bottom）",
                ):
                    return

                # 分类指标检验
                if not run_command(
                    [
                        sys.executable,
                        "experiments/hypothesis_test.py",
                        "--result_dirs",
                        cross_project_mmoe_dir,
                        cross_project_shared_dir,
                        "--output",
                        hypothesis_dir_cross,
                        "--metric",
                        "F1",
                    ],
                    "假设检验 - 跨项目分类F1（MMoE vs Shared-Bottom）",
                ):
                    return
            else:
                print("警告: 未找到跨项目实验结果，跳过跨项目模型比较")

    print("\n" + "=" * 70)
    print("所有实验完成！")
    print("=" * 70)

    # 5. 生成可视化图表
    print("\n" + "=" * 70)
    print("第五步: 生成可视化图表")
    print("=" * 70)

    viz_output = "outputs/figures_level2"
    os.makedirs(viz_output, exist_ok=True)

    # 消融实验可视化
    if os.path.exists("outputs/ablation_mmoe") or os.path.exists(
        "outputs/ablation_shared"
    ):
        ablation_dirs = []
        if os.path.exists("outputs/ablation_mmoe"):
            ablation_dirs.append("outputs/ablation_mmoe")
        if os.path.exists("outputs/ablation_shared"):
            ablation_dirs.append("outputs/ablation_shared")

        if ablation_dirs:
            run_command(
                [
                    sys.executable,
                    "experiments/visualize_results.py",
                    "--ablation_dirs",
                ]
                + ablation_dirs
                + ["--output", viz_output],
                "生成消融实验图表",
            )

    # 跨项目实验可视化（演示性实验）
    cross_project_dir = "outputs/cross_project"
    if os.path.exists(cross_project_dir):
        try:
            files = os.listdir(cross_project_dir)
            if any(f.startswith("train_") and f.endswith(".json") for f in files):
                run_command(
                    [
                        sys.executable,
                        "experiments/visualize_results.py",
                        "--cross_project_dir",
                        cross_project_dir,
                        "--output",
                        viz_output,
                    ],
                    "生成跨项目实验图表（演示）",
                )
        except:
            pass

    # 跨项目模型比较可视化
    cross_project_mmoe_dir = "outputs/cross_project_mmoe"
    cross_project_shared_dir = "outputs/cross_project_shared"
    if os.path.exists(cross_project_mmoe_dir) and os.path.exists(
        cross_project_shared_dir
    ):
        run_command(
            [
                sys.executable,
                "experiments/visualize_results.py",
                "--cross_project_dir",
                cross_project_mmoe_dir,
                "--output",
                viz_output,
                "--model_name",
                "MMoE",
            ],
            "生成跨项目实验图表（MMoE）",
        )
        run_command(
            [
                sys.executable,
                "experiments/visualize_results.py",
                "--cross_project_dir",
                cross_project_shared_dir,
                "--output",
                viz_output,
                "--model_name",
                "SharedBottom",
            ],
            "生成跨项目实验图表（Shared-Bottom）",
        )

    print("\n" + "=" * 70)
    print("完成！所有结果和图表已生成")
    print("=" * 70)
    print("\n实验结果保存在以下目录:")
    print("  - 消融实验: outputs/ablation_mmoe, outputs/ablation_shared")
    print("  - 跨项目泛化演示: outputs/cross_project")
    print(
        "  - 跨项目模型比较: outputs/cross_project_mmoe, outputs/cross_project_shared"
    )
    print("  - 假设检验（单项目）: outputs/hypothesis_test")
    print("  - 假设检验（跨项目）: outputs/hypothesis_test_cross_project")
    print(f"  - 可视化图表: {viz_output}/")
    print("\n请查看各目录下的JSON文件和图表获取详细结果。")


if __name__ == "__main__":
    main()
