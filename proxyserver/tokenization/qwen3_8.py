"""Qwen3.8 model profile.

The proxy supplies a bare assistant header so the checkpoint samples its
own think block, as in the Qwen3.5 integration.

The bundled template preserves Qwen3.8's reasoning-effort system prefix
and assistant-history behavior. TokenStreamManager measures the implicit
prefix and strips it from continuation deltas. Two patches allow tool-only
deltas: omit the missing-user-query rejection and open the user block when
the first message is a tool result. The original is retained alongside it
as chat_template-legacy.jinja.
"""

from __future__ import annotations
from .base import ModelProfile, ThinkReasoning
from .tool_parser import Qwen3CoderToolParser


class Qwen3_8(ModelProfile):
    name = "qwen3_8"
    tokenizer_dir = "Qwen3.8"
    generation_prompt = "<|im_start|>assistant\n"
    assistant_turn_end = "<|im_end|>\n"
    tool_call_parser = Qwen3CoderToolParser
    reasoning = ThinkReasoning()


PROFILE = Qwen3_8()
