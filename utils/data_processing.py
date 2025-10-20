import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DataConfig:
    """数据配置类"""
    file: str
    sheet_name: Optional[str] = None
    numeric_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    target_reg: str = "pr_close_time_hours"
    target_cls: str = "pr_merged"
    one_hot_categorical: bool = False
    split: Dict[str, Any] = None


class PRDataset(Dataset):
    """PR数据集类，用于PyTorch数据加载器"""
    def __init__(self,
                 X_num: Optional[np.ndarray],
                 X_cat: Optional[np.ndarray],
                 y_reg: np.ndarray,
                 y_cls: np.ndarray,
                 one_hot_categorical: bool = False):
        self.X_num = torch.from_numpy(X_num).float() if X_num is not None else None
        # one-hot 时为 float，否则为 Long 索引
        if X_cat is not None:
            if one_hot_categorical:
                self.X_cat = torch.from_numpy(X_cat).float()
            else:
                self.X_cat = torch.from_numpy(X_cat).long()
        else:
            self.X_cat = None
        self.y_reg = torch.from_numpy(y_reg).float().view(-1, 1)
        self.y_cls = torch.from_numpy(y_cls).float().view(-1, 1)

    def __len__(self):
        return self.y_reg.shape[0]

    def __getitem__(self, idx):
        x_num = self.X_num[idx] if self.X_num is not None else torch.empty(0)
        x_cat = self.X_cat[idx] if self.X_cat is not None else torch.empty(0, dtype=torch.long)
        return x_num, x_cat, self.y_reg[idx], self.y_cls[idx]


class FeatureProcessor:
    """特征处理器，处理数值特征和类别特征"""
    def __init__(self, numeric_features: List[str], categorical_features: List[str], one_hot_categorical: bool):
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.one_hot_categorical = one_hot_categorical
        self.scaler = StandardScaler() if self.numeric_features else None
        # 对于 embedding 方案：记录每个类别列的词表映射
        self.cat_vocab: Dict[str, Dict[Any, int]] = {}
        # 对于 one-hot 方案：固定列名顺序
        self.one_hot_columns: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame):
        """拟合特征处理器"""
        if self.numeric_features:
            self.scaler.fit(df[self.numeric_features])
        if self.categorical_features and not self.one_hot_categorical:
            for col in self.categorical_features:
                uniques = pd.Series(df[col].astype(str).fillna("<NA>").unique())
                # 保留 0 作为未知/填充
                vocab = {v: i + 1 for i, v in enumerate(sorted(uniques))}
                self.cat_vocab[col] = vocab
        if self.categorical_features and self.one_hot_categorical:
            dummies = pd.get_dummies(df[self.categorical_features].astype(str).fillna("<NA>"), drop_first=False)
            # 固定列顺序
            self.one_hot_columns = dummies.columns.tolist()

    def transform(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """转换特征数据"""
        X_num = None
        X_cat = None
        if self.numeric_features:
            X_num = self.scaler.transform(df[self.numeric_features]).astype(np.float32)
        if self.categorical_features:
            if self.one_hot_categorical:
                # 使用 pandas.get_dummies 进行 one-hot
                dummies = pd.get_dummies(df[self.categorical_features].astype(str).fillna("<NA>"), drop_first=False)
                if self.one_hot_columns is not None:
                    dummies = dummies.reindex(columns=self.one_hot_columns, fill_value=0)
                X_cat = dummies.values.astype(np.float32)
            else:
                # 将类别映射到索引
                cat_arrays = []
                for col in self.categorical_features:
                    vocab = self.cat_vocab.get(col, {})
                    indices = df[col].astype(str).fillna("<NA>").map(lambda x: vocab.get(x, 0)).astype(int).values
                    cat_arrays.append(indices)
                X_cat = np.stack(cat_arrays, axis=1).astype(np.int64)
        return X_num, X_cat

    def get_output_dims(self) -> Tuple[int, List[int], int]:
        """获取输出维度"""
        num_dim = len(self.numeric_features) if self.numeric_features else 0
        if self.one_hot_categorical:
            # 计算 one-hot 后的总维度
            cat_dim = len(self.one_hot_columns) if self.one_hot_columns is not None else 0
            return num_dim, [], cat_dim
        else:
            cat_cardinalities = []
            for col in self.categorical_features:
                vocab = self.cat_vocab.get(col, {})
                # +1 for unknown idx 0
                cat_cardinalities.append(len(vocab) + 1)
            return num_dim, cat_cardinalities, 0


def load_dataframe(file: str, sheet_name: Optional[str]) -> pd.DataFrame:
    """加载数据文件"""
    ext = os.path.splitext(file)[1].lower()
    if ext in [".xlsx", ".xls"]:
        # 当 sheet_name 为空时默认读取第一个工作表（0），避免返回 dict
        if sheet_name is None:
            return pd.read_excel(file, sheet_name=0)
        return pd.read_excel(file, sheet_name=sheet_name)
    elif ext == ".csv":
        return pd.read_csv(file)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def prepare_dataloaders(cfg: DataConfig,
                        batch_size: int,
                        num_workers: int = 0,
                        seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """准备数据加载器"""
    df = load_dataframe(cfg.file, cfg.sheet_name)

    # 处理缺失并确保目标列存在
    if cfg.target_reg not in df.columns or cfg.target_cls not in df.columns:
        raise ValueError(f"缺少目标列: {cfg.target_reg}, {cfg.target_cls}")

    # 默认数值/类别列推断（如未提供）
    numeric_features = cfg.numeric_features or df.select_dtypes(include=[np.number]).columns.tolist()
    # 去掉目标列
    numeric_features = [c for c in numeric_features if c not in [cfg.target_reg, cfg.target_cls]]

    categorical_features = cfg.categorical_features or [
        c for c in df.columns
        if c not in numeric_features + [cfg.target_reg, cfg.target_cls]
    ]

    # 简单缺失处理
    df[numeric_features] = df[numeric_features].fillna(0)
    for c in categorical_features:
        df[c] = df[c].astype(str).fillna("<NA>")

    y_reg = df[cfg.target_reg].astype(float).values
    y_cls_raw = df[cfg.target_cls].values
    # 将分类目标转换为 0/1
    if df[cfg.target_cls].dtype == bool:
        y_cls = y_cls_raw.astype(np.float32)
    else:
        # 尝试智能映射各种取值到 0/1
        mapping = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, 1: 1.0, 0: 0.0, "yes": 1.0, "no": 0.0}
        def to01(v):
            if isinstance(v, str):
                v_low = v.strip().lower()
                return mapping.get(v_low, float(v)) if v_low in mapping else float(v)
            return mapping.get(v, float(v))
        y_cls = np.array([to01(v) for v in y_cls_raw], dtype=np.float32)

    # 划分数据集
    test_size = float(cfg.split.get("test_size", 0.1)) if cfg.split else 0.1
    val_size = float(cfg.split.get("val_size", 0.1)) if cfg.split else 0.1
    stratify_by = cfg.split.get("stratify_by") if cfg.split else None
    stratify_vals = df[stratify_by] if stratify_by in df.columns else None

    X_train, X_temp, y_reg_train, y_reg_temp, y_cls_train, y_cls_temp = train_test_split(
        df, y_reg, y_cls, test_size=test_size + val_size, random_state=seed,
        stratify=stratify_vals if stratify_vals is not None else None
    )

    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_reg_val, y_reg_test, y_cls_val, y_cls_test = train_test_split(
        X_temp, y_reg_temp, y_cls_temp, test_size=1 - rel_val, random_state=seed,
        stratify=(X_temp[stratify_by] if stratify_vals is not None else None)
    )

    # 拟合特征处理器
    processor = FeatureProcessor(numeric_features, categorical_features, cfg.one_hot_categorical)
    processor.fit(X_train)

    Xnum_tr, Xcat_tr = processor.transform(X_train)
    Xnum_v, Xcat_v = processor.transform(X_val)
    Xnum_te, Xcat_te = processor.transform(X_test)

    train_ds = PRDataset(Xnum_tr, Xcat_tr, y_reg_train.astype(np.float32), y_cls_train.astype(np.float32), cfg.one_hot_categorical)
    val_ds = PRDataset(Xnum_v, Xcat_v, y_reg_val.astype(np.float32), y_cls_val.astype(np.float32), cfg.one_hot_categorical)
    test_ds = PRDataset(Xnum_te, Xcat_te, y_reg_test.astype(np.float32), y_cls_test.astype(np.float32), cfg.one_hot_categorical)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_dim, cat_cardinalities, cat_onehot_dim = processor.get_output_dims()

    meta = {
        "numeric_dim": num_dim,
        "cat_cardinalities": cat_cardinalities,  # [] if one-hot
        "cat_onehot_dim": cat_onehot_dim,        # 0 if embedding
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "one_hot_categorical": cfg.one_hot_categorical,
    }

    return train_loader, val_loader, test_loader, meta