# main.py
# 训练和测试入口示例

import argparse
from config import cfg


def train():
    print("训练流程占位")


def test():
    print("测试流程占位")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    args = parser.parse_args()
    if args.mode == "train":
        train()
    else:
        test()
