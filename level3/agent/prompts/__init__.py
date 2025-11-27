"""Prompts模块"""

from .simple_prompts import (get_detection_prompt, get_suggestion_prompt,
                             GENERAL_DETECTION_PROMPT,
                             GENERAL_SUGGESTION_PROMPT)

__all__ = [
    'get_detection_prompt', 'get_suggestion_prompt',
    'GENERAL_DETECTION_PROMPT', 'GENERAL_SUGGESTION_PROMPT'
]
