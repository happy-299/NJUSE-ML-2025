"""
实验运行脚本 - 统一运行所有实验
"""

import os
import sys
import argparse
import subprocess
from typing import List

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_command(cmd: List[str], description: str):
    """运行命令并打印状态"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} 失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ {description} 出错: {e}")
        return False


def run_all_experiments(config: str, output_base: str = "./outputs"):
    """运行所有实验"""
    success_count = 0
    total_count = 0

    experiments = [
        {
            "name": "消融实验",
            "cmd": [
                "python",
                "experiments/ablation_study.py",
                "--config",
                config,
                "--output",
                os.path.join(output_base, "ablation"),
            ],
        },
        {
            "name": "泛化性实验",
            "cmd": [
                "python",
                "experiments/cross_project.py",
                "--config",
                config,
                "--output",
                os.path.join(output_base, "cross_project"),
                "--projects",
                "django",
            ],
        },
    ]

    for exp in experiments:
        total_count += 1
        if run_command(exp["cmd"], exp["name"]):
            success_count += 1

    # 假设检验（需要先有结果）
    print(f"\n{'='*60}")
    print("假设检验需要在消融实验完成后运行")
    print(
        "请运行: python experiments/hypothesis_testing.py --summaries outputs/ablation/ablation_summary.json"
    )
    print(f"{'='*60}\n")

    # 汇总
    print(f"\n{'#'*60}")
    print(f"# 实验完成汇总")
    print(f"# 成功: {success_count}/{total_count}")
    print(f"{'#'*60}\n")

    return success_count == total_count


def main():
    parser = argparse.ArgumentParser(description="运行所有实验")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/django_shared.yaml",
        help="基础配置文件",
    )
    parser.add_argument("--output", type=str, default="./outputs", help="输出基础目录")

    args = parser.parse_args()

    success = run_all_experiments(args.config, args.output)

    if success:
        print("\n所有实验运行成功！")
        sys.exit(0)
    else:
        print("\n部分实验失败，请查看日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
