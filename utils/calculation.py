# utils/calculation.py
# 计算类工具函数示例

import numpy as np


def accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()
