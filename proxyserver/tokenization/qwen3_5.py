"""Qwen3.5: ChatML, qwen3_coder tools, and think-tag reasoning.

The bare assistant header lets the checkpoint sample its own think block.
TokenStreamManager supplies missing assistant-ending tokens on continuation.
"""

from .base import ModelProfile, ThinkReasoning
from .tool_parser import Qwen3CoderToolParser


class Qwen3_5(ModelProfile):
    name = "qwen3_5"
    tokenizer_dir = "Qwen3.5"
    generation_prompt = "<|im_start|>assistant\n"
    assistant_turn_end = "<|im_end|>\n"
    tool_call_parser = Qwen3CoderToolParser
    reasoning = ThinkReasoning()


PROFILE = Qwen3_5()
