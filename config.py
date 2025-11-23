import os
from pathlib import Path

# ==================== 路径配置 ====================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 数据集子目录
DIFF_QUALITY_DIR = RAW_DATA_DIR / "Diff_Quality_Estimation"
COMMENT_GEN_DIR = RAW_DATA_DIR / "Comment_Generation"
CODE_REFINE_DIR = RAW_DATA_DIR / "Code_Refinement"

# 任务四数据文件直接在 raw 目录下
TASK4_TRAIN_FILE = RAW_DATA_DIR / "ref-train.jsonl"
TASK4_VALID_FILE = RAW_DATA_DIR / "ref-valid.jsonl"
TASK4_TEST_FILE = RAW_DATA_DIR / "ref-test.jsonl"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LEVEL1_OUTPUT = OUTPUT_DIR / "level1"
LEVEL2_OUTPUT = OUTPUT_DIR / "level2"
LEVEL3_OUTPUT = OUTPUT_DIR / "level3"

# Level 1 模型检查点
LEVEL1_CHECKPOINT_DIR = PROJECT_ROOT / "level1" / "checkpoints"

# ==================== Level 1 配置 ====================

# CodeReviewer 模型配置
CODEREVIEWER_MODEL_NAME = "microsoft/codereviewer"
CODEREVIEWER_MAX_SOURCE_LENGTH = 512
CODEREVIEWER_MAX_TARGET_LENGTH = 128

# 训练参数
LEVEL1_TRAIN_CONFIG = {
    "batch_size": 12,
    "learning_rate": 3e-4,
    "num_epochs": 30,
    "gradient_accumulation_steps": 3,
    "warmup_steps": 1000,
    "save_steps": 3600,
    "log_steps": 100,
    "max_grad_norm": 1.0,
    "seed": 42,
}

# ==================== Level 2 配置 ====================

# LLM API 配置
LLM_CONFIG = {
    "provider": "openai",  # openai, anthropic, or custom
    "model": "gpt-4o-mini",  # gpt-4, gpt-3.5-turbo, claude-3-opus, etc.
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 1.0,
}

# API 密钥 (建议使用环境变量)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Level 2 推理配置
LEVEL2_INFERENCE_CONFIG = {
    "batch_size": 1,  # LLM API 通常逐条推理
    "retry_attempts": 3,
    "timeout": 30,
}

# ==================== Level 3 配置 ====================

# AI Agent 配置
AGENT_CONFIG = {
    "agent_type": "langchain",  # langchain, autogen, or custom
    "llm_model": "gpt-4",
    "max_iterations": 10,
    "verbose": True,
}

# Agent 工具配置
AGENT_TOOLS = [
    "code_analyzer",
    "quality_estimator",
    "localizer",
    "comment_generator",
    "code_refiner",
]

# ==================== 评估配置 ====================

# 评估指标
METRICS_CONFIG = {
    "task1": ["accuracy", "precision", "recall", "f1_macro"],
    "task2": ["accuracy", "precision", "recall", "f1_macro", "mrr"],
    "task3": ["bleu4", "rouge_l", "bert_score"],
    "task4": ["exact_match", "code_bleu"],
}

# ==================== 设备配置 ====================

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# ==================== 日志配置 ====================

LOG_DIR = OUTPUT_DIR / "logs"
LOG_LEVEL = "INFO"

# ==================== 工具函数 ====================


def ensure_dirs():
    """确保所有必要的目录存在"""
    dirs = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        LEVEL1_OUTPUT,
        LEVEL2_OUTPUT,
        LEVEL3_OUTPUT,
        LEVEL1_CHECKPOINT_DIR,
        LOG_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_data_path(task, split="train"):
    """获取数据集路径

    Args:
        task: 任务类型 ("task1", "task2", "task3", "task4")
        split: 数据集划分 ("train", "valid", "test")

    Returns:
        数据文件路径
    """
    task_map = {
        "task1": DIFF_QUALITY_DIR,
        "task2": None,  # Task 2 可能需要从其他任务数据中提取
        "task3": COMMENT_GEN_DIR,
        "task4": CODE_REFINE_DIR,
    }

    if task not in task_map:
        raise ValueError(f"Unknown task: {task}")

    base_dir = task_map[task]
    if base_dir is None:
        return None

    # 文件命名规则
    prefix_map = {
        "task1": "cls",
        "task3": "msg",
        "task4": "ref",
    }
    prefix = prefix_map.get(task, "data")

    if split == "train":
        # 训练集可能有多个文件
        train_files = list(base_dir.glob(f"{prefix}-train*.jsonl"))
        return train_files if train_files else None
    else:
        return base_dir / f"{prefix}-{split}.jsonl"


if __name__ == "__main__":
    # 测试配置
    ensure_dirs()
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
    print(f"Num GPUs: {NUM_GPUS}")
