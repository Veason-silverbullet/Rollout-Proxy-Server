"""Rollout-engine adaptation for the rollout server's ``TokenOutput`` contract.

Under ``transport_mode: verl`` the proxy never talks to vLLM or SGLang directly.
It calls the training framework's rollout server actor, and both backends implement the same
``generate(prompt_ids, sampling_params, request_id) -> TokenOutput`` interface.
The framework also accepts vLLM-style ``max_tokens`` and ``logprobs``
on *both* backends and translates them internally, so the
sampling params the proxy sends need no per-engine branching.

Exactly one field of ``TokenOutput`` is backend-dependent: ``stop_reason``.
Taking verl — the most popular RL training framework — as the reference
implementation of this contract:

``verl/workers/rollout/vllm_rollout/vllm_async_server.py``::

    if finish_reason == "abort":              stop_reason = "aborted"
    elif finish_reason in ("stop", "length"): stop_reason = "completed"
    else:                                     stop_reason = finish_reason

``verl/workers/rollout/sglang_rollout/async_sglang_server.py``::

    finish_reason = meta_info.get("finish_reason")
    finish_reason = finish_reason["type"] if finish_reason else None
    return TokenOutput(..., stop_reason=finish_reason, ...)   # "stop"|"length"|"abort"

So SGLang reports truncation authoritatively while vLLM collapses "stop" and
"length" into a single ``"completed"``.  On vLLM the proxy must recover
truncation from the token count; on SGLang it must *not*, because a response
that legitimately emits EOS on its final permitted token would otherwise be
mislabeled ``"length"``.  That asymmetry is what :class:`EngineAdapter`
encodes, and it is why the engine has to be named rather than sniffed.

Under ``transport_mode: direct`` (and slime) the proxy speaks to the engines
itself, so ``stop_reason`` arrives in each engine's **own** vocabulary
(``"stop"``/``"length"``/``"abort"``) rather than verl's.  The adapters
accept that too: an already-normalized reason passes through verbatim, and
because the vllm adapter cannot assume truncation reporting, a raw vLLM
``"stop"`` is still double-checked against the token count while a raw
``"length"`` is trusted.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Final, Mapping

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


@dataclass(frozen=True)
class EngineAdapter:
    """Normalizes one rollout engine's ``TokenOutput.stop_reason``."""

    name: str

    #: Whether this engine's ``stop_reason`` distinguishes truncation from a
    #: natural stop.  When False, truncation is recovered from the token count.
    reports_truncation: bool

    #: Engine-specific ``stop_reason`` -> normalized finish reason.
    stop_reasons: Mapping[str, str]

    def finish_reason(self, stop_reason: str | None, *, num_tokens: int, max_tokens: int) -> str:
        """Normalize a ``TokenOutput.stop_reason`` to ``stop``/``length``/``abort``.

        ``max_tokens`` is the cap the proxy requested. Note that the framework
        (e.g. verl) may clamp it down to the remaining context window before
        calling the engine, so on a
        non-``reports_truncation`` engine a response truncated by that clamp
        rather than by our cap is reported as ``stop``.
        """
        normalized = self.stop_reasons.get(stop_reason)
        if normalized is None and stop_reason in _NORMALIZED:
            # The framework (e.g. verl) passes stop reasons it does not recognize through verbatim.
            normalized = stop_reason

        if normalized == ABORT:
            return ABORT
        if normalized == LENGTH:
            return LENGTH
        if normalized == STOP and self.reports_truncation:
            return STOP
        # Either the engine cannot distinguish truncation (vLLM's "completed")
        # or it reported nothing at all: fall back to the token count.
        if max_tokens and num_tokens >= max_tokens:
            return LENGTH
        return STOP


_VLLM_ADAPTER: Final = EngineAdapter(
    name=VLLM,
    reports_truncation=False,
    stop_reasons={"completed": STOP, "aborted": ABORT},
)

_SGLANG_ADAPTER: Final = EngineAdapter(
    name=SGLANG,
    reports_truncation=True,
    stop_reasons={"stop": STOP, "length": LENGTH, "abort": ABORT},
)

_ADAPTERS: Final[dict[str, EngineAdapter]] = {
    VLLM: _VLLM_ADAPTER,
    SGLANG: _SGLANG_ADAPTER,
}


def get_adapter(inference_engine: str | None = None) -> EngineAdapter:
    """Resolve an engine name to its adapter.

    Falls back to ``$INFERENCE_ENGINE`` and then to ``"vllm"``.
    """
    name = (inference_engine or os.getenv(ENGINE_ENV_VAR) or VLLM).strip().lower()
    try:
        return _ADAPTERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown inference_engine {name!r}; supported engines: {', '.join(ENGINES)}"
        ) from None
