"""Qwen3.6 model profile.

Qwen3.6 uses the same ChatML turn markers, ``qwen3_coder`` XML
tool-call dialect, and ``<think>...</think>`` reasoning format as Qwen3.5.
The proxy supplies a bare assistant header so the checkpoint samples its
own think block.  The bundled chat template retains Qwen3.6's upstream
assistant-history behavior with the two strict-TITO tool-delta patches
documented in ``tokenization/README.md``; the pristine upstream template is
kept beside it as ``chat_template-legacy.jinja``.
"""

from __future__ import annotations
from .base import ModelProfile, ThinkReasoning
from .tool_parser import Qwen3CoderToolParser


class Qwen3_6(ModelProfile):
    name = "qwen3_6"
    tokenizer_dir = "Qwen3.6"
    generation_prompt = "<|im_start|>assistant\n"
    assistant_turn_end = "<|im_end|>\n"
    tool_call_parser = Qwen3CoderToolParser
    reasoning = ThinkReasoning()


PROFILE = Qwen3_6()
