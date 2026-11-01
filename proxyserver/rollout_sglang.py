"""SGLang /generate provider and protocol helpers.

The generation and context-discovery helpers are also used by direct mode.
Shared rollout semantics and transport controls live in rollout_base.
"""

from __future__ import annotations
import asyncio
import base64
import logging
from typing import Any, Callable
import httpx
import numpy as np
from .engines import SGLANG, EngineError, resolve_engine
from .rollout_base import (
    BaseRolloutProvider, _RouterTransport, _clamp_to_context_window,
    _engine_meta, _finite_logprob, engine_http_limits,
)

logger = logging.getLogger(__name__)


async def _sglang_generate(
    http: Any,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    session_id: str | None,
    return_routed_experts: bool = False,
    routed_experts_prior: tuple[int, int] | None = None,
) -> tuple[list[int], list[float], str | None, dict[str, Any]]:
    """One generation in SGLang's native ``/generate`` dialect.

    Shared by :class:`SGLangRolloutProvider` (directly or through a transparent
    router) and :class:`DirectRolloutProvider`. Logprobs that do not line up
    with the sampled ids yield an empty result, which ``generate()`` refuses.
    ``routed_experts_prior`` is the recorder's ``(rows, cols)`` for this
    stream; the engine then returns only the new routing rows.
    """
    params = dict(sampling_params)
    params["max_new_tokens"] = params.pop("max_tokens", None)
    if "seed" in params:
        params["sampling_seed"] = params.pop("seed")
    return_logprob = bool(params.pop("logprobs", False))
    params = {k: v for k, v in params.items() if v is not None}

    body: dict[str, Any] = {
        "input_ids": prompt_ids,
        "sampling_params": params,
        "return_logprob": return_logprob,
    }
    if return_routed_experts:
        body["return_routed_experts"] = True
        if routed_experts_prior and routed_experts_prior[0] > 0:
            body["routed_experts_start_len"] = routed_experts_prior[0]
    resp = await http.post(
        "/generate",
        json=body,
        headers=_routing_headers(session_id),
    )
    resp.raise_for_status()
    if return_routed_experts:
        output = await asyncio.to_thread(resp.json)
    else:
        output = resp.json()

    meta_info = output.get("meta_info", {})
    finish_reason = meta_info.get("finish_reason")
    stop_reason = finish_reason["type"] if finish_reason else None
    meta = _engine_meta(meta_info)

    output_token_logprobs = meta_info.get("output_token_logprobs") or []
    token_ids = output.get("output_ids")
    if token_ids is None:
        token_ids = []
    if not isinstance(token_ids, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in token_ids
    ):
        raise EngineError(f"Session {session_id}: SGLang returned invalid output_ids")
    logprob_token_ids: list[int] = []
    log_probs: list[float] = []
    if return_logprob or (not token_ids and output_token_logprobs):
        if not isinstance(output_token_logprobs, list):
            raise EngineError(f"Session {session_id}: SGLang returned invalid output_token_logprobs")
        for index, entry in enumerate(output_token_logprobs):
            if (
                not isinstance(entry, (list, tuple)) or len(entry) < 2
                or isinstance(entry[1], bool) or not isinstance(entry[1], int) or entry[1] < 0
                or (return_logprob and (
                    isinstance(entry[0], bool) or not isinstance(entry[0], (int, float))
                ))
            ):
                raise EngineError(f"Session {session_id}: SGLang returned an invalid logprob entry at position {index}")
            logprob_token_ids.append(entry[1])
            if return_logprob:
                log_probs.append(_finite_logprob(entry[0], session_id))
    if not token_ids and output_token_logprobs:
        # With SGLang's tokenizer initialized, output_ids may be absent.
        # The logprob triples retain the sampled IDs without re-tokenizing text.
        token_ids = logprob_token_ids

    if return_logprob and logprob_token_ids != token_ids:
        logger.error(
            "Session %s: SGLang logprob/token ID mismatch (%d vs %d tokens)",
            session_id, len(logprob_token_ids), len(token_ids),
        )
        return [], [], stop_reason, meta

    if return_routed_experts:
        meta["routed_experts"] = await asyncio.to_thread(
            _pack_routed_experts, meta_info, len(prompt_ids), len(token_ids), session_id, routed_experts_prior,
        )

    return token_ids, log_probs, stop_reason, meta


def _routing_headers(session_id: str | None) -> dict[str, str] | None:
    """Per-stream prefix-cache affinity for ``consistent_hashing`` routers;
    other router policies ignore the header."""
    return {"X-SMG-Routing-Key": session_id} if session_id else None


def _pack_routed_experts(
    meta_info: dict[str, Any],
    prompt_len: int,
    completion_len: int,
    session_id: str | None,
    prior: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """SGLang's ``meta_info["routed_experts"]``, repacked as a uint8 record blob.

    The engine returns base64 little-endian int32 rows of ``num_layers * topk``
    covering ``[start, prompt + completion - 1)``, where ``start`` is the
    prior's row count. A full-range answer from an engine or router that
    ignored ``routed_experts_start_len`` is sliced down to the delta. Missing
    or misaligned payloads raise ``EngineError`` instead of being recorded.
    """
    encoded = meta_info.get("routed_experts")
    if not encoded:
        raise EngineError(
            f"Session {session_id}: return_routed_experts was requested but the "
            f"engine reported no routed_experts — were the SGLang servers "
            f"launched with enable_return_routed_experts (slime's "
            f"--use-rollout-routing-replay)?"
        )
    values = np.frombuffer(base64.b64decode(encoded), dtype="<i4")
    full_rows = prompt_len + completion_len - 1
    start, cols_hint = prior or (0, None)
    rows = full_rows - start
    if start and cols_hint and values.size == full_rows * cols_hint:
        values = values[-rows * cols_hint:]
    if rows <= 0 or values.size == 0 or values.size % rows != 0 or (
        cols_hint and values.size // rows != cols_hint
    ):
        raise EngineError(
            f"Session {session_id}: routed_experts payload does not line up with "
            f"the stream ({values.size} values for {rows} rows = prompt "
            f"{prompt_len} + completion {completion_len} - 1 - start {start}"
            f"{f', expected {cols_hint} cols' if cols_hint else ''})"
        )
    if values.min() < 0 or values.max() > 255:
        raise EngineError(
            f"Session {session_id}: routed_experts ids outside [0, 256) "
            f"(min {values.min()}, max {values.max()}) do not fit uint8"
        )
    return {
        "data": base64.b64encode(values.astype(np.uint8).tobytes()).decode("ascii"),
        "rows": rows,
        "cols": values.size // rows,
        "dtype": "uint8",
        "start": start,
    }


def _served_context_length(info: Any) -> int | None:
    """The context window out of a ``GET /get_server_info`` payload.

    Accepts a flat ServerArgs payload or a router's list of worker payloads.
    ``context_length`` is null on a default launch, so the scheduler's
    ``max_req_input_len`` is the fallback. Anything unrecognized is ``None``.
    """
    if isinstance(info, list):
        info = next((entry for entry in info if isinstance(entry, dict)), None)
    if not isinstance(info, dict):
        return None
    for key in ("context_length", "max_req_input_len"):
        try:
            value = int(info.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


class _SGLangContextWindow:
    """Cache one endpoint's context window using the provider's discovery lock.

    Failed fetches warn once and remain retryable. A successful response is
    cached even when it names no usable window (None disables clamping).
    A configured context length bypasses discovery entirely.
    """

    def __init__(self, lock: asyncio.Lock, context_length: int | None = None) -> None:
        self._lock = lock
        self._length = context_length
        self._known = context_length is not None
        self._fetch_warned = False

    async def get(self, http: Any, endpoint: str, *, unknown_hint: str = "") -> int | None:
        if self._known:
            return self._length
        async with self._lock:
            if self._known:
                return self._length
            try:
                resp = await http.get("/get_server_info")
                resp.raise_for_status()
                info = resp.json()
            except Exception as e:
                if not self._fetch_warned:
                    self._fetch_warned = True
                    logger.warning(
                        "%s: could not fetch /get_server_info (%s); running without "
                        "the context-window clamp until it answers — meanwhile the "
                        "engine itself rejects prompt+max_tokens overruns",
                        endpoint, e,
                    )
                return None
            self._length = _served_context_length(info)
            self._known = True
            if self._length is None:
                logger.warning(
                    "%s: /get_server_info names neither context_length nor max_req_input_len; "
                    "the context-window clamp stays off%s", endpoint, unknown_hint,
                )
            else:
                logger.info("%s: engine context window is %d tokens (from /get_server_info)", endpoint, self._length)
            return self._length


class SGLangRolloutProvider(_RouterTransport, BaseRolloutProvider):
    """Native SGLang HTTP ``/generate``, used by ``sglang_generate``.

    Sends token-ID prompts to a SGLang server or through a transparent router
    such as Slime's ``sgl-router``, which owns placement and balancing.
    """

    def __init__(
        self,
        router_url: str,
        inference_engine: str | None = None,
        api_key: str | None = None,
        request_timeout: float = 900.0,
        tokenizer_loader: Callable[[str], Any] | None = None,
        sampling_overrides: dict[str, Any] | None = None,
        context_length: int | None = None,
        return_routed_experts: bool = False,
    ) -> None:
        """
        Args:
            router_url: Base URL of the router or a bare SGLang server.
            inference_engine: Must resolve to ``"sglang"``; defaults to it
                without consulting ``$INFERENCE_ENGINE``.
            api_key: Optional bearer token sent to the router.
            request_timeout: Per-request HTTP timeout (seconds).
            tokenizer_loader: Optional tokenizer-loading override.
            context_length: The engines' context window. Slime's router
                answers ``/get_server_info`` itself, so set this when using
                it; ``None`` keeps discovery against a bare server.
            return_routed_experts: Config baseline for R3 capture; requires
                engines launched with ``enable_return_routed_experts``.
        """
        engine = resolve_engine(inference_engine or SGLANG)
        if engine != SGLANG:
            raise ValueError(
                f"SGLangRolloutProvider requires inference_engine='sglang' "
                f"for the native /generate protocol, got {engine!r}"
            )
        if context_length is not None:
            context_length = int(context_length)
            if context_length <= 0:
                raise ValueError(f"context_length must be a positive number of tokens, got {context_length}")

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self.router_url = router_url.rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=self.router_url,
            headers=headers,
            timeout=request_timeout,
            limits=engine_http_limits(),
        )
        self._context_window = _SGLangContextWindow(asyncio.Lock(), context_length)
        # Runtime layer (PUT /routed_experts) wins over config; None means cleared.
        self._config_return_routed_experts = bool(return_routed_experts)
        self._runtime_return_routed_experts: bool | None = None
        super().__init__(engine, tokenizer_loader, sampling_overrides=sampling_overrides)
        logger.info("SGLang rollout provider routing via %s", self.router_url)
        if self._config_return_routed_experts:
            logger.info(
                "%s: routed-experts capture on by config (R3) — /generate "
                "requests carry return_routed_experts", self.router_url,
            )
        if context_length is not None:
            logger.info(
                "%s: engine context window pinned to %d tokens by config "
                "(/get_server_info not consulted)", self.router_url, context_length,
            )

    async def _engine_context_length(self) -> int | None:
        """The configured or discovered engine window; None disables clamping."""
        return await self._context_window.get(
            self._http, self.router_url,
            unknown_hint=(
                " — set context_length in the proxy config to pin the window "
                "(slime's sgl-router answers this endpoint from its own "
                "RouterManager stub instead of forwarding a worker's payload)"
            ),
        )

    async def _call_engine(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        session_id: str | None,
        routed_experts_prior: tuple[int, int] | None = None,
    ) -> tuple[list[int], list[float], str | None, dict[str, Any]]:
        _clamp_to_context_window(
            sampling_params, len(prompt_ids),
            await self._engine_context_length(), self.router_url, session_id,
        )
        return await _sglang_generate(
            self._http, prompt_ids, sampling_params, session_id,
            return_routed_experts=self.get_routed_experts_config()["effective"],
            routed_experts_prior=routed_experts_prior,
        )
