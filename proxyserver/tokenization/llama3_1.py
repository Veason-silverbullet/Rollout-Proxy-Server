"""Llama 3.1 model profile.

Llama-3 header/``<|eot_id|>`` chat format, the ``llama3_json`` bare-JSON
tool-call dialect, and no chain-of-thought.  Serves the ``Llama-3.1-*``
instruct names, which share this tokenizer and format.

The template also prepends a fixed boilerplate to every render (a ``bos`` token
and a dated default system block); that implicit prefix is *measured and
stripped* by :class:`~proxyserver.token_stream.TokenStreamManager` (a wrong
measurement fails loudly rather than corrupting), so the profile need not state
it — only the two markers a stream's byte-exactness hinges on.
"""

from __future__ import annotations
from .base import ModelProfile, NoReasoning
from .tool_parser import Llama3JsonToolParser


class Llama3_1(ModelProfile):
    name = "llama3_1"
    tokenizer_dir = "Llama-3.1"
    generation_prompt = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    assistant_turn_end = "<|eot_id|>"
    tool_call_parser = Llama3JsonToolParser
    reasoning = NoReasoning()


#: The profile :func:`proxyserver.tokenization.base.get_profile` loads for the
#: name ``"llama3_1"``.
PROFILE = Llama3_1()
