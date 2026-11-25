"""Offline test reporting, tokenizer loading, and scripted provider fixtures."""

from __future__ import annotations
import asyncio
import tempfile
from pathlib import Path
from typing import Any
from proxyserver.model_registry import TOKENIZATION_DIR
from proxyserver.rollout_provider import BaseRolloutProvider


def check(label: str, cond: bool) -> None:
    """One assertion in the ladder's ``  ok  <label>`` output convention."""
    assert cond, f"FAIL: {label}"
    print(f"  ok  {label}")


def load_yaml_config(body: str):
    """Load a temporary config file, exercising errors that include its path."""
    from proxyserver.config import load_config

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(body)
        path = f.name
    try:
        return load_config(path)
    finally:
        Path(path).unlink()


class FakeResponse:
    """Successful JSON response from a scripted engine endpoint."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self._payload


class FakeHttp:
    """Record POST requests and return a scripted JSON payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, json: dict[str, Any], headers: Any = None) -> FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers})
        return FakeResponse(self._payload)


class FakeTokenizer:
    """Deterministic stand-in: render length scales with the messages."""

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=True):
        if not messages:
            raise ValueError("empty conversation")  # Qwen-style template
        return [100 + i for i in range(2 * len(messages))]

    def encode(self, text, add_special_tokens=False):
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


class RecordingProvider(ScriptedProvider):
    """Keep the exact engine prompts for token-reconstruction assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[list[int]] = []

    async def _call_engine(self, prompt_ids, sampling_params, session_id):
        self.prompts.append(list(prompt_ids))
        return await super()._call_engine(prompt_ids, sampling_params, session_id)


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
    bundled = Path(model).resolve().is_relative_to(TOKENIZATION_DIR)
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
