"""
LLM工厂模块
支持加载多种LLM：本地CodeReviewer、OpenAI、Ollama等
"""

import torch
from typing import Optional, Any
from langchain_core.language_models.llms import BaseLLM
from langchain_core.language_models import BaseChatModel
import logging

from .config import get_config

logger = logging.getLogger(__name__)


class CodeReviewerLLM(BaseLLM):
    """自定义CodeReviewer LLM包装器"""

    model: Any = None
    tokenizer: Any = None
    device: str = "cuda"
    max_length: int = 512
    num_beams: int = 5

    def __init__(self,
                 model_path: str = "microsoft/codereviewer",
                 device: str = "auto",
                 **kwargs):
        """
        初始化CodeReviewer模型
        
        Args:
            model_path: 模型路径（HuggingFace模型名或本地路径）
            device: 设备（cuda/cpu/auto）
            **kwargs: 其他参数
        """
        super().__init__()

        # 设置设备 - 检查 CUDA 是否真正可用
        if device == "auto":
            if torch.cuda.is_available():
                try:
                    # 尝试初始化 CUDA
                    torch.cuda.init()
                    self.device = "cuda"
                except Exception as e:
                    logger.warning(f"CUDA 初始化失败，回退到 CPU: {e}")
                    self.device = "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.max_length = kwargs.get("max_length", 512)
        self.num_beams = kwargs.get("num_beams", 5)

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """加载CodeReviewer模型"""
        try:
            from transformers import RobertaTokenizer, T5ForConditionalGeneration

            logger.info(f"正在加载CodeReviewer模型: {model_path}")
            logger.info(f"使用设备: {self.device}")

            # CodeReviewer使用RoBERTa tokenizer和T5模型
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16
                if self.device == "cuda" else torch.float32)

            self.model.to(self.device)
            self.model.eval()

            logger.info("模型加载完成")

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        return "codereviewer"

    def _call(self,
              prompt: str,
              stop: Optional[list[str]] = None,
              **kwargs) -> str:
        """
        执行推理
        
        Args:
            prompt: 输入提示
            stop: 停止词
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        try:
            # Tokenize
            inputs = self.tokenizer(prompt,
                                    return_tensors="pt",
                                    max_length=self.max_length,
                                    truncation=True,
                                    padding=True).to(self.device)

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(**inputs,
                                              max_length=kwargs.get(
                                                  "max_length", 256),
                                              num_beams=kwargs.get(
                                                  "num_beams", self.num_beams),
                                              early_stopping=True,
                                              temperature=kwargs.get(
                                                  "temperature", 0.7))

            # Decode
            result = self.tokenizer.decode(outputs[0],
                                           skip_special_tokens=True)
            return result

        except Exception as e:
            logger.error(f"推理失败: {e}")
            return f"Error: {str(e)}"

    def _generate(self,
                  prompts: list[str],
                  stop: Optional[list[str]] = None,
                  **kwargs) -> Any:
        """
        批量生成（实现抽象方法）
        
        Args:
            prompts: 提示列表
            stop: 停止词
            **kwargs: 其他参数
            
        Returns:
            生成结果
        """
        from langchain_core.outputs import LLMResult, Generation

        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, **kwargs)
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)


class LLMFactory:
    """LLM工厂类"""

    @staticmethod
    def create_llm(config: Optional[Any] = None) -> BaseLLM:
        """
        根据配置创建LLM实例
        
        Args:
            config: 配置对象，如果为None则使用全局配置
            
        Returns:
            LLM实例
        """
        if config is None:
            config = get_config()

        llm_type = config.llm_type.lower()

        logger.info(f"创建LLM: {llm_type}")

        if llm_type == "local":
            # 使用本地模型
            if config.local_model_type == "codereviewer":
                return CodeReviewerLLM(model_path=config.local_model_path,
                                       device=config.device,
                                       max_length=config.max_length,
                                       num_beams=config.num_beams)
            else:
                # 使用HuggingFace Pipeline
                from langchain_community.llms import HuggingFacePipeline
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

                logger.info(f"加载HuggingFace模型: {config.local_model_path}")

                tokenizer = AutoTokenizer.from_pretrained(
                    config.local_model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    config.local_model_path)

                pipe = pipeline("text-generation",
                                model=model,
                                tokenizer=tokenizer,
                                max_length=config.max_length,
                                device=0 if config.device == "cuda"
                                and torch.cuda.is_available() else -1)

                return HuggingFacePipeline(pipeline=pipe)

        elif llm_type == "openai":
            # 使用OpenAI API
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(api_key=config.openai_api_key,
                              base_url=config.openai_api_base,
                              temperature=config.temperature,
                              max_tokens=config.max_length)

        elif llm_type == "ollama":
            # 使用Ollama
            from langchain_community.llms import Ollama

            return Ollama(base_url=config.ollama_base_url,
                          model=config.ollama_model,
                          temperature=config.temperature)

        else:
            raise ValueError(f"不支持的LLM类型: {llm_type}")


def get_llm(config: Optional[Any] = None) -> BaseLLM:
    """
    获取LLM实例（便捷函数）
    
    Args:
        config: 配置对象
        
    Returns:
        LLM实例
    """
    return LLMFactory.create_llm(config)
