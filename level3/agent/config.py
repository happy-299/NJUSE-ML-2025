"""
配置管理模块
加载和管理环境变量配置
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    """配置管理类"""

    def __init__(self, env_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            env_path: .env文件路径，默认为level3/.env
        """
        if env_path is None:
            # 默认路径：level3/.env
            env_path = Path(__file__).parent.parent / ".env"

        # 加载环境变量
        if Path(env_path).exists():
            load_dotenv(env_path)
            logger.info(f"已加载配置文件: {env_path}")
        else:
            logger.warning(f"配置文件不存在: {env_path}")

        # LLM配置
        self.llm_type = os.getenv("LLM_TYPE", "local")  # local, openai, ollama
        self.local_model_type = os.getenv("LOCAL_MODEL_TYPE", "codereviewer")
        self.local_model_path = os.getenv("LOCAL_MODEL_PATH",
                                          "microsoft/codereviewer")
        self.device = os.getenv("DEVICE", "auto")  # cuda, cpu, auto

        # OpenAI配置
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api_base = os.getenv("OPENAI_API_BASE",
                                         "https://api.openai.com/v1")

        # Ollama配置
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL",
                                         "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "codellama")

        # 模型参数
        self.max_length = int(os.getenv("MAX_LENGTH", "512"))
        self.num_beams = int(os.getenv("NUM_BEAMS", "5"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))

        # 其他配置
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.timeout = int(os.getenv("TIMEOUT", "60"))

        # 数据路径 - 指向项目根目录的 data/raw
        project_root = Path(__file__).parent.parent.parent  # NJUSE-ML-2025/
        self.data_dir = project_root / "data" / "raw"
        self.code_refinement_dir = self.data_dir / "Code_Refinement"
        self.comment_generation_dir = self.data_dir / "Comment_Generation"
        self.diff_quality_dir = self.data_dir / "Diff_Quality_Estimation"
        self.codereviewer_code_dir = self.data_dir / "CodeReviewer" / "code"

        # 更新日志级别
        logging.getLogger().setLevel(self.log_level)

    def validate(self) -> bool:
        """
        验证配置有效性
        
        Returns:
            配置是否有效
        """
        if self.llm_type == "openai" and not self.openai_api_key:
            logger.error("使用OpenAI时必须设置OPENAI_API_KEY")
            return False

        if not self.data_dir.exists():
            logger.error(f"数据目录不存在: {self.data_dir}")
            return False

        return True

    def __repr__(self) -> str:
        """字符串表示"""
        return f"""Config(
    llm_type={self.llm_type},
    local_model_type={self.local_model_type},
    local_model_path={self.local_model_path},
    device={self.device},
    max_length={self.max_length},
    data_dir={self.data_dir}
)"""


# 全局配置实例
_config: Optional[Config] = None


def get_config(env_path: Optional[str] = None) -> Config:
    """
    获取全局配置实例（单例模式）
    
    Args:
        env_path: .env文件路径
        
    Returns:
        配置实例
    """
    global _config
    if _config is None:
        _config = Config(env_path)
    return _config
