"""Engine selection and shared finish-reason normalization.

Both vLLM and SGLang report stop, length, or abort. Explicit reasons are
authoritative, including EOS sampled on the final permitted token. Only
missing or unknown reasons fall back to the requested token budget.
"""

from __future__ import annotations
import os
from typing import Final

VLLM: Final = "vllm"
SGLANG: Final = "sglang"
ENGINES: Final = (VLLM, SGLANG)

#: Environment variable consulted when ``inference_engine`` is not passed.
ENGINE_ENV_VAR: Final = "INFERENCE_ENGINE"

#: Completion-token budget for a request that names none.
DEFAULT_MAX_TOKENS: Final = 65536

# Normalized finish reasons, in OpenAI's vocabulary.
STOP: Final = "stop"
LENGTH: Final = "length"
ABORT: Final = "abort"

_NORMALIZED: Final = frozenset({STOP, LENGTH, ABORT})


class EngineError(RuntimeError):
    """The rollout engine returned a result the proxy cannot record."""


class EngineAbort(EngineError):
    """The engine aborted the request; its tokens are partial or empty."""


def normalize_finish_reason(stop_reason: str | None, *, num_tokens: int, max_tokens: int) -> str:
    """Keep native finish reasons authoritative for either engine."""
    if stop_reason in _NORMALIZED:
        return stop_reason
    if max_tokens and num_tokens >= max_tokens:
        return LENGTH
    return STOP


def resolve_engine(inference_engine: str | None = None) -> str:
    """Resolve and validate an engine name.

    Falls back to ``$INFERENCE_ENGINE`` and then to ``"vllm"``.
    """
    name = (inference_engine or os.getenv(ENGINE_ENV_VAR) or VLLM).strip().lower()
    if name not in ENGINES:
        raise ValueError(f"Unknown inference_engine {name!r}; supported engines: {', '.join(ENGINES)}")
    return name
