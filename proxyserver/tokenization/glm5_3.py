"""GLM-5.3-Flash's text chat format and GLM XML tool-call dialect.

The bare assistant header lets the model sample its own think block.
There is no fixed assistant ending in the template. A sampled user or
observation stop token already opens the next message; the token-stream
manager must not append that header twice.
"""

from collections.abc import Iterator
from .base import ModelProfile, ThinkReasoning
from .tool_parser import GLMToolParser


class GLMReasoning(ThinkReasoning):
    """GLM ends reasoning at a tool opener; tool bodies are literal data."""

    @staticmethod
    def _segments(text: str) -> Iterator[tuple[str, bool]]:
        """Yield (text, is_tool) segments, keeping tool bodies untouched."""
        pos = 0
        while True:
            start = text.find("<tool_call>", pos)
            if start < 0:
                yield text[pos:], False
                return
            before = text[pos:start]
            if before.rfind("<think>") > before.rfind("</think>"):
                # Close implicit reasoning only in the parsing view.
                before += "</think>"
            yield before, False
            end = text.find("</tool_call>", start + len("<tool_call>"))
            if end < 0:
                # A truncated call still owns its entire remaining body.
                yield text[start:], True
                return
            pos = end + len("</tool_call>")
            yield text[start:pos], True

    def extract(self, text: str) -> str | None:
        for part, is_tool in self._segments(text):
            if not is_tool and "</think>" in part:
                return super().extract(part)
        return None

    def strip(self, text: str) -> str:
        strip_reasoning = super().strip
        return "".join(part if is_tool else strip_reasoning(part)
                       for part, is_tool in self._segments(text))


class GLM5_3(ModelProfile):
    name = "glm5_3"
    tokenizer_dir = "GLM-5.3-Flash"
    reasoning_efforts = ("low", "high", "max")
    default_reasoning_effort = "max"
    generation_prompt = "<|assistant|>"
    assistant_turn_end = ""
    continuation_headers = ("<|user|>", "<|observation|>")
    tool_call_parser = GLMToolParser
    reasoning = GLMReasoning()


PROFILE = GLM5_3()
