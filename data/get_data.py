import os
import shutil
from typing import Optional

import pandas as pd


def load_data(file: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    加载数据文件，支持CSV和Excel格式
    
    Args:
        file: 文件路径
        sheet_name: Excel工作表名称，如果为None则读取第一个工作表
        
    Returns:
        pandas DataFrame 对象
    """
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


def copy_data_to_local(source_path: str, destination_dir: str = "./data") -> str:
    """
    将数据文件复制到本地数据目录
    
    Args:
        source_path: 源文件路径
        destination_dir: 目标目录
        
    Returns:
        目标文件路径
    """
    os.makedirs(destination_dir, exist_ok=True)
    filename = os.path.basename(source_path)
    destination_path = os.path.join(destination_dir, filename)
    
    shutil.copy2(source_path, destination_path)
    print(f"文件已复制到: {destination_path}")
    return destination_path


if __name__ == "__main__":
    # 示例用法
    # 1. 复制数据文件到本地
    if os.path.exists("engineered/synth.csv"):
        local_path = copy_data_to_local("engineered/synth.csv", "./data")
    
    # 2. 加载数据
    if os.path.exists("./data/synth.csv"):
        df = load_data("./data/synth.csv")
        print(f"数据加载成功，形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")