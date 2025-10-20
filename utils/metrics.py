from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算回归指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        包含回归指标的字典
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        y_true: 真实类别
        y_prob: 预测概率
        
    Returns:
        包含分类指标的字典
    """
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "F1_macro": f1_macro}