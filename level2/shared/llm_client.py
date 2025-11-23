"""
Level 2 共享模块: LLM 客户端封装

支持多种 LLM API (OpenAI, Anthropic, 等)
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional
import openai

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM API 客户端封装"""

    def __init__(self, provider="openai", model="gpt-4o-mini", api_key=None, **kwargs):
        """
        初始化 LLM 客户端

        Args:
            provider: API 提供商 ("openai", "anthropic", 等)
            model: 模型名称
            api_key: API 密钥
            **kwargs: 其他参数 (temperature, max_tokens, 等)
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.config = kwargs

        if not self.api_key:
            raise ValueError(f"API key not found for provider: {provider}")

        # 初始化客户端
        if provider == "openai":
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
        elif provider == "anthropic":
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retry_attempts: int = 3,
        retry_delay: int = 2,
    ) -> str:
        """
        调用聊天补全 API

        Args:
            messages: 消息列表,格式 [{"role": "user/assistant/system", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大 token 数
            retry_attempts: 重试次数
            retry_delay: 重试延迟(秒)

        Returns:
            LLM 响应文本
        """
        temperature = temperature or self.config.get("temperature", 0.7)
        max_tokens = max_tokens or self.config.get("max_tokens", 2048)

        for attempt in range(retry_attempts):
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=self.config.get("top_p", 1.0),
                    )
                    return response.choices[0].message.content

                elif self.provider == "anthropic":
                    # 将 messages 转换为 Anthropic 格式
                    system_msg = next(
                        (m["content"] for m in messages if m["role"] == "system"), ""
                    )
                    user_msgs = [m for m in messages if m["role"] != "system"]

                    response = self.client.messages.create(
                        model=self.model,
                        system=system_msg,
                        messages=user_msgs,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.content[0].text

            except Exception as e:
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{retry_attempts}): {e}"
                )
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
                else:
                    raise

    def extract_json(self, text: str) -> Optional[Dict]:
        """
        从 LLM 响应中提取 JSON

        Args:
            text: LLM 响应文本

        Returns:
            解析后的 JSON 字典,或 None
        """
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 块
        import re

        json_pattern = r"```json\s*(.*?)\s*```"
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取 {...} 块
        json_pattern = r"\{.*\}"
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to extract JSON from response: {text[:200]}")
        return None


def create_llm_client(provider=None, model=None, api_key=None, **kwargs):
    """
    创建 LLM 客户端的工厂函数

    Args:
        provider: API 提供商
        model: 模型名称
        api_key: API 密钥
        **kwargs: 其他参数

    Returns:
        LLMClient 实例
    """
    # 从 config 读取默认配置
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from config import LLM_CONFIG

    provider = provider or LLM_CONFIG["provider"]
    model = model or LLM_CONFIG["model"]

    config = {**LLM_CONFIG, **kwargs}

    return LLMClient(provider=provider, model=model, api_key=api_key, **config)


if __name__ == "__main__":
    # 测试
    client = create_llm_client()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in JSON format with a 'message' field."},
    ]

    response = client.chat_completion(messages)
    print(f"Response: {response}")

    data = client.extract_json(response)
    print(f"Extracted JSON: {data}")
