"""Per-model profiles — the model-format-specific half of serving one model
family, declared in one place instead of inferred or scattered.

A :class:`ModelProfile` states everything the strict-TITO machinery needs to
know about a model family that it cannot know generically:

* ``tokenizer_dir`` — the bundled tokenizer under ``proxyserver/tokenization/``;
* ``generation_prompt`` / ``assistant_turn_end`` — the chat-format markers the
  ``TokenStreamManager`` appends and closes turns with, **stated** rather than
  derived from the template (byte-exactness is too important to infer);
* ``tool_call_parser`` — the tool-call dialect (a ``BaseToolParser`` subclass,
  or ``None`` to disable parsing);
* ``reasoning`` — how the model delimits chain-of-thought, used to strip
  reasoning before tool parsing.

The strict-TITO *algorithm* (delta construction, the turn-gap, ``commit``,
``cached_turn``) stays in :class:`~proxyserver.token_stream.TokenStreamManager`;
a profile supplies only the model-specific data and hooks it consumes.  A new
model family is therefore a small declarative module, not an edit to the shared
core — and the two format strings a stream's byte-exactness hinges on are
stated by a human who knows the model, not guessed from its template.

``tokenization/mapping.json`` maps each served model *name* to a profile
module: ``{"Llama-3.1-8B-Instruct": "llama3_1", ...}`` loads
``tokenization/llama3_1.py`` and reads its module-level ``PROFILE``.

This module is a dependency **sink**: it imports only the standard library, so
``tool_parser`` (for reasoning stripping) and ``model_registry`` (to build the
tokenizer / parser around a profile) can import it without an import cycle.
The concrete profile modules — not this one — import the tool-parser classes.
"""

from __future__ import annotations
import importlib
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: The directory holding this module and the bundled tokenizer subdirectories.
TOKENIZATION_DIR = Path(__file__).resolve().parent

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _drop_think_blocks(text: str) -> str:
    """Remove every complete ``<think>...</think>`` span from ``text``.

    An *unterminated* opener is left alone: the completion was truncated
    mid-reasoning, and dropping everything after it would hide the only content
    there is.
    """
    parts: list[str] = []
    pos = 0
    while True:
        open_at = text.find(_THINK_OPEN, pos)
        if open_at < 0:
            break
        close_at = text.find(_THINK_CLOSE, open_at + len(_THINK_OPEN))
        if close_at < 0:
            break
        parts.append(text[pos:open_at])
        pos = close_at + len(_THINK_CLOSE)
        if text.startswith("\n\n", pos):  # the template's "</think>\n\n"
            pos += 2
    parts.append(text[pos:])
    return "".join(parts)


@runtime_checkable
class ReasoningParser(Protocol):
    """How a model delimits chain-of-thought reasoning in a completion.

    Two operations: :meth:`strip` for the tool parser (which must not parse
    tool calls out of a reasoning block), :meth:`extract` for consumers that
    restore a completion's reasoning from its recorded ``completion_text``.
    """

    def extract(self, text: str) -> str | None:
        """The reasoning the model emitted, or ``None`` if none / undelimited."""

    def strip(self, text: str) -> str:
        """``text`` with reasoning removed — the agent-visible content."""


class ThinkReasoning:
    """``<think>...</think>`` chain-of-thought (Qwen3.5 / DeepSeek-R1 dialect).

    :meth:`extract` returns the leading think block; :meth:`strip` drops the
    leading reasoning *and* every later complete ``<think>...</think>`` span —
    a checkpoint that reasons again mid-completion must not leak that reasoning
    to the agent, nor have the tool calls it merely weighed there executed.
    """

    def extract(self, text: str) -> str | None:
        before, sep, _ = text.partition(_THINK_CLOSE)
        if not sep:
            return None
        reasoning = before.rsplit(_THINK_OPEN, 1)[-1].strip()
        return reasoning or None

    def strip(self, text: str) -> str:
        _, sep, tail = text.partition(_THINK_CLOSE)
        if not sep:
            return text
        return _drop_think_blocks(tail).lstrip("\n")


class NoReasoning:
    """Models that emit no delimited chain-of-thought (Llama 3.1)."""

    def extract(self, text: str) -> str | None:
        return None

    def strip(self, text: str) -> str:
        return text


class ModelProfile:
    """Base for a model family's profile.

    Subclasses set the class attributes below; this base provides only the
    small derived helper (:attr:`tokenizer_path`).  Kept deliberately thin — a
    profile is *data*, and the shared machinery (tokenizer loading, tool-parser
    factory wiring, the whole strict-TITO flow) reads it rather than living
    here, so a profile can never drift from that flow by re-implementing it.
    """

    #: Profile id — the value ``mapping.json`` points a model name at, and the
    #: module basename (``qwen3_5`` -> ``tokenization/qwen3_5.py``).
    name: str = ""
    #: Bundled tokenizer directory under :data:`TOKENIZATION_DIR`.
    tokenizer_dir: str = ""
    #: Chat-format markers the ``TokenStreamManager`` appends / closes turns
    #: with — stated, never derived from the template.
    generation_prompt: str = ""
    assistant_turn_end: str = ""
    #: Tool-call dialect: a ``BaseToolParser`` subclass, or ``None`` to disable
    #: parsing (annotated ``Any`` so this sink module need not import the class).
    tool_call_parser: Any = None
    #: Chain-of-thought format (a shared, stateless instance).
    reasoning: ReasoningParser = NoReasoning()

    @property
    def tokenizer_path(self) -> str:
        """Absolute path passed to ``AutoTokenizer.from_pretrained``."""
        return str(TOKENIZATION_DIR / self.tokenizer_dir)

    def __repr__(self) -> str:
        return f"<ModelProfile {self.name!r} tokenizer={self.tokenizer_dir!r}>"


def get_profile(name: str) -> ModelProfile:
    """Load the profile from ``tokenization/{name}.py``.

    The module must expose a module-level ``PROFILE`` that is a
    :class:`ModelProfile`.  Any failure — a missing module, a wrong ``PROFILE``
    type — raises here, at load time: unlike an inferred chat format, a missing
    or malformed profile fails loudly instead of ever serving a stream that is
    silently one token off.
    """
    try:
        module = importlib.import_module(f"{__package__}.{name}")
    except ImportError as e:
        raise ValueError(
            f"No model profile module {name!r} (expected tokenization/{name}.py): {e}"
        ) from e
    profile = getattr(module, "PROFILE", None)
    if not isinstance(profile, ModelProfile):
        raise ValueError(
            f"tokenization/{name}.py must define a module-level PROFILE that is "
            f"a ModelProfile; got {type(profile).__name__}"
        )
    return profile
