"""Read gpt-oss assistant completions without discarding Harmony boundaries.

The prompt ends in ``<|start|>assistant``; sampled tokens finish that header.
One completion may contain several analysis/commentary messages followed by a
final answer or a function handoff. The recipient can precede or follow the
channel. Header metadata is structural; JSON bodies are kept verbatim.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Iterable, Iterator
from typing import Any
import re

# Structural IDs from the bundled GPT-OSS tokenizer. Ordinary vocabulary tokens
# can decode to the same spellings, so decoded strings are not delimiters.
_START, _END, _MESSAGE, _CHANNEL = 200006, 200007, 200008, 200005
_CONSTRAIN, _RETURN, _CALL, _END_OF_TEXT = 200003, 200002, 200012, 199999
_MARKERS = {
    _START: "<|start|>", _END: "<|end|>", _MESSAGE: "<|message|>",
    _CHANNEL: "<|channel|>", _CONSTRAIN: "<|constrain|>",
    _RETURN: "<|return|>", _CALL: "<|call|>", _END_OF_TEXT: "<|endoftext|>",
}
_BOUNDARIES = {_START, _END, _RETURN, _CALL, _END_OF_TEXT}
_ACTION_ENDINGS = {_RETURN, _CALL, _END_OF_TEXT}
_RECIPIENT = re.compile(r"(?:^|\s)to=([^\s<>]+)")


@dataclass(frozen=True)
class HarmonyMessage:
    channel: str | None
    recipient: str | None
    body: str
    ending: str | None


def _assistant_messages(parts: list[str | int]) -> Iterator[HarmonyMessage]:
    """Read text spans separated by structural marker IDs, never spellings."""
    if not parts or parts[0] != _START:
        parts = [_START, "assistant", *parts]
    pos = 0
    while pos < len(parts) and parts[pos] == _START:
        pos += 1
        header = []
        channel = None
        while pos < len(parts) and parts[pos] != _MESSAGE:
            part = parts[pos]
            if isinstance(part, int):
                # An unfinished header cannot borrow the next message's body.
                if part in _BOUNDARIES:
                    return
                header.append(" ")
                if part == _CHANNEL and pos + 1 < len(parts) and isinstance(parts[pos + 1], str):
                    fields = parts[pos + 1].split()
                    channel = fields[0] if fields else None
            else:
                header.append(part)
            pos += 1
        if pos == len(parts):
            return
        fields = "".join(header).split()
        if not fields or fields[0] != "assistant":
            return
        recipient = _RECIPIENT.search(" ".join(fields))
        pos += 1
        body = []
        while pos < len(parts) and parts[pos] not in _BOUNDARIES:
            part = parts[pos]
            body.append(_MARKERS[part] if isinstance(part, int) else part)
            pos += 1
        ending = parts[pos] if pos < len(parts) else None
        yield HarmonyMessage(
            channel, recipient.group(1) if recipient else None,
            "".join(body), _MARKERS[ending] if ending is not None and ending != _START else None,
        )
        if ending is None or ending in _ACTION_ENDINGS:
            return
        if ending != _START:
            pos += 1


def assistant_messages_from_tokens(token_ids: list[int], tokenizer: Any) -> Iterator[HarmonyMessage]:
    """Parse sampled IDs, decoding text only after locating structural tokens."""
    parts: list[str | int] = []
    start = 0
    for pos, token in enumerate(token_ids):
        if token in _MARKERS:
            if pos > start:
                parts.append(tokenizer.decode(token_ids[start:pos], skip_special_tokens=False))
            parts.append(token)
            start = pos + 1
    if start < len(token_ids):
        parts.append(tokenizer.decode(token_ids[start:], skip_special_tokens=False))
    yield from _assistant_messages(parts)


def assistant_messages(text: str) -> Iterator[HarmonyMessage]:
    """Read explicitly serialized Harmony text (not a decoded sampled stream).

    Text alone cannot distinguish literal spellings from special tokens. All
    sampled completions must use ``assistant_messages_from_tokens`` instead.
    """
    markers = {value: key for key, value in _MARKERS.items()}
    parts = re.split("(" + "|".join(map(re.escape, markers)) + ")", text)
    yield from _assistant_messages([markers.get(part, part) for part in parts if part])


class HarmonyReasoning:
    """Separate analysis from visible commentary/final messages and tool data."""

    def extract(self, text: str) -> str | None:
        return self._extract(assistant_messages(text))

    def extract_tokens(self, token_ids: list[int], tokenizer: Any) -> str | None:
        return self._extract(assistant_messages_from_tokens(token_ids, tokenizer))

    @staticmethod
    def _extract(messages: Iterable[HarmonyMessage]) -> str | None:
        return "\n\n".join(m.body.strip() for m in messages
                           if m.channel == "analysis" and m.recipient is None and m.body.strip()) or None

    def strip(self, text: str) -> str:
        return "\n\n".join(m.body for m in assistant_messages(text)
                           if m.channel != "analysis" and m.recipient is None)
