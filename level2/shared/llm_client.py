"""
Level 2 Shared Module: LLM Client Wrapper

Support multiple LLM APIs (OpenAI, DeepSeek, Anthropic, etc.)
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM API Client Wrapper"""

    def __init__(
        self,
        provider="deepseek",
        model="deepseek-chat",
        api_key=None,
        base_url=None,
        **kwargs,
    ):
        """
        Initialize LLM client

        Args:
            provider: API provider ("openai", "deepseek", "anthropic", etc.)
            model: Model name
            api_key: API key
            base_url: API base URL (for custom endpoints like DeepSeek)
            **kwargs: Other parameters (temperature, max_tokens, etc.)
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.config = kwargs

        if not self.api_key:
            raise ValueError(f"API key not found for provider: {provider}")

        # Initialize client based on provider
        if provider == "deepseek":
            self.client = OpenAI(
                api_key=self.api_key, base_url=base_url or "https://api.deepseek.com"
            )
        elif provider == "openai":
            self.client = OpenAI(api_key=self.api_key)
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
        Call chat completion API

        Args:
            messages: Message list, format [{"role": "user/assistant/system", "content": "..."}]
            temperature: Temperature parameter
            max_tokens: Max tokens
            retry_attempts: Retry attempts
            retry_delay: Retry delay (seconds)

        Returns:
            LLM response text
        """
        temperature = temperature or self.config.get("temperature", 0.7)
        max_tokens = max_tokens or self.config.get("max_tokens", 2048)

        for attempt in range(retry_attempts):
            try:
                if self.provider in ["openai", "deepseek"]:
                    # Both OpenAI and DeepSeek use the same API format
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,
                    )
                    return response.choices[0].message.content

                elif self.provider == "anthropic":
                    # Convert messages to Anthropic format
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


def create_llm_client(provider=None, model=None, api_key=None, base_url=None, **kwargs):
    """
    Factory function to create LLM client

    Args:
        provider: API provider
        model: Model name
        api_key: API key
        base_url: API base URL
        **kwargs: Other parameters

    Returns:
        LLMClient instance
    """
    # Read default config from config.py
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from config import LLM_CONFIG, DEEPSEEK_API_KEY

    provider = provider or LLM_CONFIG["provider"]
    model = model or LLM_CONFIG["model"]
    base_url = base_url or LLM_CONFIG.get("base_url")

    # Use DEEPSEEK_API_KEY from config if not provided
    if provider == "deepseek" and not api_key:
        api_key = DEEPSEEK_API_KEY

    # Merge configs, but remove keys that are already passed as explicit arguments
    config = {**LLM_CONFIG, **kwargs}
    # Remove keys that will be passed explicitly to avoid duplicate keyword arguments
    for key in ["provider", "model", "api_key", "base_url"]:
        config.pop(key, None)

    return LLMClient(
        provider=provider, model=model, api_key=api_key, base_url=base_url, **config
    )


if __name__ == "__main__":
    # Test
    client = create_llm_client()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in JSON format with a 'message' field."},
    ]

    response = client.chat_completion(messages)
    print(f"Response: {response}")

    data = client.extract_json(response)
    print(f"Extracted JSON: {data}")
