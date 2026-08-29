"""Qwen3.5 / Qwen3-Coder model profile.

ChatML chat format, the ``qwen3_coder`` XML tool-call dialect, and
``<think>...</think>`` chain-of-thought.  Serves every ``Qwen3.5-*`` model
name — they all share this one tokenizer and format — so ``mapping.json``
points each of those names at this module.

Why the generation prompt is the **bare** ChatML assistant header
(``<|im_start|>assistant\\n``), not the template's own
``add_generation_prompt=True`` output: the bundled Qwen3.5 template pre-fills a
dangling ``<think>\\n`` there, but the served RL checkpoint samples its own
think block (usually the empty ``<think>\\n\\n</think>\\n\\n``, a full one when
it reasons).  Pre-filled, it goes off-distribution — it never emits
``</think>`` and loops on repeated pseudo-reasoning until ``max_tokens``.
Appending only the header keeps the template file stock (byte-identical to the
upstream model release) while giving the engine the header-only prompt the
checkpoint expects.  A *headerless* prompt is not an option either: the
checkpoint then samples the header itself, unreliably (it sometimes emits
``<|im_start|><think>`` with no role and runs away), and the forced-format
header tokens would pollute the recorded completion.

The assistant turn ends with ``<|im_end|>\\n`` — the ``<|im_end|>`` the model
samples plus the newline the template writes after it; the manager's turn-gap
supplies whichever part of that ending the sampled completion did not itself
carry (see :class:`~proxyserver.token_stream.TokenStreamManager`).
"""

from __future__ import annotations
from .base import ModelProfile, ThinkReasoning
from .tool_parser import Qwen3CoderToolParser


class Qwen3_5(ModelProfile):
    name = "qwen3_5"
    tokenizer_dir = "Qwen3.5"
    # ChatML markers — see the module docstring for why a *bare* assistant
    # header (no <think> pre-fill) and the <|im_end|>\n turn end.
    generation_prompt = "<|im_start|>assistant\n"
    assistant_turn_end = "<|im_end|>\n"
    tool_call_parser = Qwen3CoderToolParser
    reasoning = ThinkReasoning()


#: The profile :func:`proxyserver.tokenization.base.get_profile` loads for the
#: name ``"qwen3_5"``.
PROFILE = Qwen3_5()
