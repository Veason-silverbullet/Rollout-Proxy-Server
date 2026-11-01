"""vLLM /inference/v1/generate provider and protocol helpers.

Owns native token responses and routing-capture conversion.
The direct /v1/completions transport remains in rollout_provider.
"""

from __future__ import annotations
import asyncio
import base64
import io
import logging
from typing import Any, Callable
import httpx
import numpy as np
from .engines import ABORT, VLLM, EngineError, resolve_engine
from .rollout_base import (
    BaseRolloutProvider, _RouterTransport, _clamp_to_context_window,
    _engine_meta, _finite_logprob, engine_http_limits,
)

logger = logging.getLogger(__name__)


def _weight_version(output: dict, choice: dict, token_count: int) -> dict[str, Any]:
    """Validate native vLLM version spans before reducing to a turn version.

    Native vLLM supplies half-open ranges over output tokens; a synchronous training turn
    must have complete coverage from one version. Never infer a version from the request.
    """
    meta = _engine_meta(output)
    spans = choice.get("weight_versions")
    if spans is None:
        return meta
    if not isinstance(spans, list):
        raise EngineError("vLLM weight_versions must be a list of token spans")
    end, versions = 0, set()
    for span in spans:
        if (not isinstance(span, dict)
                or type(span.get("start")) is not int or type(span.get("end")) is not int
                or span["start"] != end or not end < span["end"] <= token_count
                or not isinstance(span.get("version"), str) or not span["version"]):
            raise EngineError("vLLM returned invalid weight-version token coverage")
        end = span["end"]
        versions.add(span["version"])
    if end != token_count or len(versions) > 1:
        raise EngineError("vLLM returned missing or mixed policy versions")
    if versions:
        version = next(iter(versions))
        if meta.get("weight_version", version) != version:
            raise EngineError("vLLM returned conflicting policy versions")
        meta["weight_version"] = version
    return meta


def _pack_vllm_routed_experts(
    choice: dict[str, Any],
    prompt_len: int,
    completion_len: int,
    session_id: str | None,
    prior: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Repack vLLM's base64 .npy capture into the same blob as SGLang.

    vLLM returns (tokens, layers, topk), with token coverage ending at
    prompt + completion - 1. A continuation may return only the rows after
    ``routed_experts_prompt_start``; older engines may return the full
    capture, which is sliced locally. Never load pickled numpy objects or
    narrow expert IDs that do not fit the recorder's uint8 format.
    """
    encoded = choice.get("routed_experts")
    if not isinstance(encoded, str) or not encoded:
        raise EngineError(
            f"Session {session_id}: return_routed_experts was requested but vLLM "
            "reported no routed_experts — enable --use-rollout-routing-replay on the engines"
        )
    try:
        values = np.load(io.BytesIO(base64.b64decode(encoded, validate=True)), allow_pickle=False)
    except Exception as e:
        raise EngineError(f"Session {session_id}: invalid vLLM routed_experts .npy payload") from e
    if not isinstance(values, np.ndarray):
        values.close()  # np.load also accepts .npz archives; they are not this protocol.
        raise EngineError(f"Session {session_id}: vLLM routed_experts must be a numpy array")
    if values.ndim not in (2, 3) or values.dtype.kind not in "iu" or not values.size:
        raise EngineError(f"Session {session_id}: vLLM routed_experts must be a nonempty integer token/layer array")
    values = values.reshape(values.shape[0], -1)
    full_rows = prompt_len + completion_len - 1
    start, cols_hint = prior or (0, None)
    rows = full_rows - start
    if start and values.shape[0] == full_rows:
        values = values[start:]
    if rows <= 0 or values.shape[0] != rows or (cols_hint and values.shape[1] != cols_hint):
        raise EngineError(
            f"Session {session_id}: vLLM routed_experts shape {values.shape} does not "
            f"line up with {rows} rows (start {start}, expected cols {cols_hint})"
        )
    if values.min() < 0 or values.max() > 255:
        raise EngineError(f"Session {session_id}: vLLM routed_experts ids outside [0, 256) do not fit uint8")
    return {
        "data": base64.b64encode(values.astype(np.uint8).tobytes()).decode("ascii"),
        "rows": rows,
        "cols": values.shape[1],
        "dtype": "uint8",
        "start": start,
    }


async def _vllm_generate(
    http: Any,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    session_id: str | None,
    *,
    return_routed_experts: bool = False,
    routed_experts_prior: tuple[int, int] | None = None,
) -> tuple[list[int], list[float], str | None, dict[str, Any]]:
    """Native vLLM token-ID protocol, parallel to :func:`_sglang_generate`.

    The router URL selects the policy model;
    no text rendering or model discovery is performed upstream. Cancellation
    closes this HTTP request, never broadcasts an abort to shared workers.
    """
    params = {k: v for k, v in sampling_params.items() if v is not None}
    params["logprobs"] = 1
    if params.get("stop"):
        params["include_stop_str_in_output"] = True
    if return_routed_experts:
        params["routed_experts_prompt_start"] = routed_experts_prior[0] if routed_experts_prior else 0
    resp = await http.post(
        "/inference/v1/generate",
        json={"token_ids": prompt_ids, "sampling_params": params},
        headers={"x-session-id": session_id} if session_id else None,
    )
    resp.raise_for_status()
    output = await asyncio.to_thread(resp.json) if return_routed_experts else resp.json()
    choices = output.get("choices") if isinstance(output, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise EngineError(f"Session {session_id}: vLLM must return exactly one generation choice")
    choice = choices[0]
    stop_reason = choice.get("finish_reason")
    if isinstance(stop_reason, dict):
        stop_reason = stop_reason.get("type")
    if stop_reason == "cancelled":
        stop_reason = ABORT
    if stop_reason == ABORT:
        return [], [], ABORT, _engine_meta(output)
    if stop_reason not in (None, "stop", "length"):
        raise EngineError(f"Session {session_id}: unknown vLLM finish_reason {stop_reason!r}")
    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list) or any(type(token) is not int or token < 0 for token in token_ids):
        raise EngineError(f"Session {session_id}: vLLM returned invalid token_ids")
    meta = _weight_version(output, choice, len(token_ids))
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list) or len(content) != len(token_ids):
        raise EngineError(f"Session {session_id}: vLLM logprobs do not line up with token_ids")
    log_probs: list[float] = []
    for token, entry in zip(token_ids, content):
        if not isinstance(entry, dict) or type(entry.get("logprob")) not in (int, float):
            raise EngineError(f"Session {session_id}: vLLM returned an invalid token/logprob pair")
        # Accept integer token_id and token="token_id:<id>" representations,
        # validating alignment without re-tokenizing text.
        if "token_id" in entry:
            aligned = type(entry["token_id"]) is int and entry["token_id"] == token
        else:
            aligned = entry.get("token") == f"token_id:{token}"
        if not aligned:
            raise EngineError(f"Session {session_id}: vLLM returned an invalid token/logprob pair")
        log_probs.append(_finite_logprob(entry["logprob"], session_id))
    usage = output.get("usage")
    if usage is not None:
        if not isinstance(usage, dict):
            raise EngineError(f"Session {session_id}: vLLM returned invalid usage")
        count = usage.get("completion_tokens")
        if count is not None and (type(count) is not int or count != len(token_ids)):
            raise EngineError(f"Session {session_id}: vLLM completion_tokens does not match token_ids")
    if return_routed_experts:
        meta["routed_experts"] = await asyncio.to_thread(
            _pack_vllm_routed_experts, choice, len(prompt_ids), len(token_ids), session_id, routed_experts_prior,
        )
    return token_ids, log_probs, stop_reason, meta


class VLLMRolloutProvider(_RouterTransport, BaseRolloutProvider):
    """Token HTTP ``/inference/v1/generate`` for Vime or compatible servers.

    The deployment owns the workers and load balancing; the proxy speaks
    their native HTTP dialect directly. ``x-session-id`` carries the agent
    stream for routers using ``consistent_hash``. Configure context_length
    to match ``--vllm-max-model-len``; router model discovery is not required.

    Routing replay is opt-in and requires compatible engine output. Weight
    updates must drain rollouts first; a record holds one policy version per turn.
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
        engine = resolve_engine(inference_engine or VLLM)
        if engine != VLLM:
            raise ValueError(
                f"VLLMRolloutProvider requires inference_engine='vllm' "
                f"for the native /inference/v1/generate protocol, got {engine!r}"
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
        self._context_window = context_length
        self._config_return_routed_experts = bool(return_routed_experts)
        self._runtime_return_routed_experts: bool | None = None
        super().__init__(engine, tokenizer_loader, sampling_overrides=sampling_overrides)
        logger.info("vLLM rollout provider routing via %s", self.router_url)
        if self._config_return_routed_experts:
            logger.info("%s: routed-experts capture on by config (R3)", self.router_url)
        if context_length is not None:
            logger.info("%s: engine context window pinned to %d tokens by config", self.router_url, context_length)
        else:
            logger.warning("%s: set context_length to the engine's context window in tokens; context clamping is disabled", self.router_url)

    async def _call_engine(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        session_id: str | None,
        routed_experts_prior: tuple[int, int] | None = None,
    ) -> tuple[list[int], list[float], str | None, dict[str, Any]]:
        _clamp_to_context_window(
            sampling_params, len(prompt_ids), self._context_window, self.router_url, session_id,
        )
        return await _vllm_generate(
            self._http, prompt_ids, sampling_params, session_id,
            return_routed_experts=self.get_routed_experts_config()["effective"],
            routed_experts_prior=routed_experts_prior,
        )
