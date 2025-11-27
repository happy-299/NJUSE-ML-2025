"""Test DeepSeek API connection"""

import sys

sys.path.append(r"d:\CollegeCourse\grade-3-fall\ML\lab3\NJUSE-ML-2025")

from level2.shared.llm_client import create_llm_client

print("Creating LLM client...")
client = create_llm_client()

print(f"Provider: {client.provider}")
print(f"Model: {client.model}")

print("\nSending test message...")
response = client.chat_completion(
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Say 'Hello, DeepSeek is working!' in exactly those words.",
        },
    ]
)

print(f"\nResponse: {response}")
print("\n✅ DeepSeek API is working!")
