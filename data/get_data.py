# data/get_data.py
# 示例：用于获取和预处理数据的函数

import os
from typing import Tuple

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """加载 CSV 数据为 pandas DataFrame"""
    return pd.read_csv(path)


def prepare_dataset(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """示例预处理：返回特征和标签"""
    df = load_csv(path)
    # 假设最后一列为标签
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y
