"""Shared helpers for the offline test scripts.

``test/common.py`` resolves live engine endpoints at import time, so the
offline tests share their pass/fail reporting, real-tokenizer skip
scaffolding, and proxy-stack fakes (``FakeTokenizer`` / ``ScriptedProvider``,
used by ``test-recovery.py`` and ``test-multi-agent.py``) here instead.
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Any
from proxyserver.rollout_provider import BaseRolloutProvider


def check(label: str, cond: bool) -> None:
    """One assertion in the ladder's ``  ok  <label>`` output convention."""
    assert cond, f"FAIL: {label}"
    print(f"  ok  {label}")


class FakeTokenizer:
    """Deterministic stand-in: render length scales with the messages."""

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=True):
        if not messages:
            raise ValueError("empty conversation")  # Qwen-style template
        return [100 + i for i in range(2 * len(messages))]

    def encode(self, text, add_special_tokens=False):
        # TokenStreamManager tokenizes its generation prompt through this
        # (see token_stream.GENERATION_PROMPT); a tokenizer without it fails
        # every build_prompt.
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(f"<{t}>" for t in token_ids)


class ScriptedProvider(BaseRolloutProvider):
    """Real strict-TITO generate flow over a scripted engine transport."""

    def __init__(self) -> None:
        super().__init__(tokenizer_loader=lambda path: FakeTokenizer())
        self.delay = 0.0
        self.engine_calls = 0
        self.next_tokens = [900, 901, 902]
        self.weight_version: str | None = None

    async def _call_engine(self, prompt_ids, sampling_params, session_id):
        self.engine_calls += 1
        await asyncio.sleep(self.delay)
        tokens = list(self.next_tokens)
        meta = {"weight_version": self.weight_version} if self.weight_version is not None else {}
        return tokens, [-0.1] * len(tokens), "stop", meta


class GatedProvider(ScriptedProvider):
    """Hold scripted engine responses until a test releases or cancels them."""

    def __init__(self) -> None:
        super().__init__()
        self.gate: asyncio.Event | None = None
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.finished = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def _call_engine(self, prompt_ids, sampling_params, session_id):
        output = await super()._call_engine(prompt_ids, sampling_params, session_id)
        if self.gate is not None:
            self.started.put_nowait(session_id)
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            finally:
                self.finished.set()
        return output


def load_real_tokenizer(model: str) -> Any | None:
    """Require bundled tokenizers; skip unavailable optional external models."""
    bundled = Path(model).resolve().is_relative_to(
        Path(__file__).resolve().parents[1] / "proxyserver" / "tokenization"
    )
    try:
        from transformers import AutoTokenizer
    except ImportError:
        if bundled:
            raise
        print("  skip  transformers not installed")
        return None
    try:
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception as e:
        if bundled:
            raise
        print(f"  skip  cannot load tokenizer ({type(e).__name__})")
        return None
