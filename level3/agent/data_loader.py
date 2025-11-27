"""
数据加载器
加载和处理实验数据集
"""

import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器"""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        if data_dir is None:
            from .config import get_config
            config = get_config()
            self.data_dir = config.data_dir
        else:
            self.data_dir = Path(data_dir)

        self.code_refinement_dir = self.data_dir / "Code_Refinement"
        self.comment_generation_dir = self.data_dir / "Comment_Generation"
        self.diff_quality_dir = self.data_dir / "Diff_Quality_Estimation"

    def load_jsonl(self,
                   file_path: Path,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        加载JSONL文件
        
        Args:
            file_path: 文件路径
            limit: 限制加载条数
            
        Returns:
            数据列表
        """
        data = []
        try:
            with jsonlines.open(file_path) as reader:
                for i, obj in enumerate(reader):
                    if limit and i >= limit:
                        break
                    data.append(obj)

            logger.info(f"加载了 {len(data)} 条数据from {file_path.name}")
            return data

        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
            return []

    def load_code_refinement(
            self,
            split: str = "test",
            limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        加载Code Refinement数据集
        
        Args:
            split: 数据集划分 (train/valid/test)
            limit: 限制条数
            
        Returns:
            数据列表
        """
        file_path = self.code_refinement_dir / f"ref-{split}.jsonl"
        return self.load_jsonl(file_path, limit)

    def load_comment_generation(
            self,
            split: str = "test",
            limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        加载Comment Generation数据集
        
        Args:
            split: 数据集划分 (train/valid/test)
            limit: 限制条数
            
        Returns:
            数据列表
        """
        file_path = self.comment_generation_dir / f"msg-{split}.jsonl"
        return self.load_jsonl(file_path, limit)

    def load_diff_quality(self,
                          split: str = "test",
                          limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        加载Diff Quality Estimation数据集
        
        Args:
            split: 数据集划分 (train/valid/test)
            limit: 限制条数
            
        Returns:
            数据列表
        """
        if split == "train":
            # 训练集分块，需要合并
            data = []
            for i in range(4):  # 0-3
                file_path = self.diff_quality_dir / f"cls-train-chunk-{i}.jsonl"
                if file_path.exists():
                    chunk_data = self.load_jsonl(file_path, limit)
                    data.extend(chunk_data)
                    if limit and len(data) >= limit:
                        data = data[:limit]
                        break
            return data
        else:
            file_path = self.diff_quality_dir / f"cls-{split}.jsonl"
            return self.load_jsonl(file_path, limit)

    def prepare_review_input(self,
                             data_item: Dict[str, Any],
                             task_type: str = "refinement") -> Dict[str, Any]:
        """
        准备代码审查输入
        
        Args:
            data_item: 数据项
            task_type: 任务类型 (refinement/comment/quality)
            
        Returns:
            格式化的输入字典，包含参考答案用于评价指标计算
        """
        if task_type == "refinement":
            return {
                "old_code": data_item.get("oldf", ""),
                "new_code": data_item.get("new", ""),
                "diff": data_item.get("hunk", ""),
                "language": data_item.get("lang", "unknown"),
                "context": {
                    "repo": data_item.get("repo", ""),
                    "comment": data_item.get("comment", "")  # 参考答案
                }
            }

        elif task_type == "comment":
            return {
                "code": data_item.get("oldf", ""),
                "diff": data_item.get("patch", ""),
                "language": data_item.get("lang", "unknown"),
                "context": {
                    "project": data_item.get("proj", ""),
                    "expected_comment": data_item.get("msg", "")  # 参考答案
                }
            }

        elif task_type == "quality":
            return {
                "code": data_item.get("oldf", ""),
                "diff": data_item.get("patch", ""),
                "language": data_item.get("lang", "unknown"),
                "quality_label": data_item.get("y", 0),  # 参考标签
                "context": {
                    "project": data_item.get("proj", ""),
                    "message": data_item.get("msg", "")
                }
            }

        else:
            raise ValueError(f"未知任务类型: {task_type}")

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Returns:
            统计信息字典
        """
        stats = {}

        for task, method in [("code_refinement", self.load_code_refinement),
                             ("comment_generation",
                              self.load_comment_generation),
                             ("diff_quality", self.load_diff_quality)]:
            stats[task] = {}
            for split in ["train", "valid", "test"]:
                try:
                    data = method(split=split)
                    stats[task][split] = len(data)
                except:
                    stats[task][split] = 0

        return stats
