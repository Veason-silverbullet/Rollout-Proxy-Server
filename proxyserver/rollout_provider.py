"""Rollout providers: route proxy requests to the rollout engines, strictly
token-in-token-out.

A provider owns the inference-side half of a completion: strict-TITO prompt
construction (:class:`~proxyserver.token_stream.TokenStreamManager`), the
transport to the rollout servers, engine-specific ``stop_reason``
normalization (:mod:`proxyserver.engines`), and tool-call parsing.  OpenAI
response formatting is *not* its job — the proxy server does that with
:func:`proxyserver.server.build_openai_response`.

Providers are selected by ``transport_mode`` in ``configs/{engine}-{transport}.yaml``:

* :class:`SGLangRolloutProvider` (``transport_mode: sglang_generate``) — HTTP
  through a router or native SGLang server, including Slime and VeRL.
  This provider speaks SGLang's native
  ``/generate`` dialect directly (``input_ids`` in, ``output_ids`` +
  ``meta_info`` out) and performs the vLLM-style→SGLang sampling-param
  translation itself.
* :class:`VLLMRolloutProvider` (``transport_mode: vllm_generate``) — HTTP
  through a router or native server, speaking ``/inference/v1/generate`` (``token_ids``
  in, ``choices`` with sampled token IDs, logprobs, and replay metadata out).
* :class:`DirectRolloutProvider` (``transport_mode: direct``) — plain HTTP
  straight to the vLLM / SGLang engines themselves, no training framework in
  the path.  Token-ID prompts go to vLLM's ``/v1/completions``
  (``return_tokens_as_token_ids``) or SGLang's ``/generate``; endpoints come
  from ``inference_engine_base_url`` / ``inference_engine_api_key`` and each
  session is sticky to one endpoint (prefix cache).

The backend is selected with ``inference_engine`` (``"vllm"`` or
``"sglang"``); see :mod:`proxyserver.engines` for what it changes. Generation
transports infer their engine when unspecified; direct supports both engines.

Every HTTP completion request must name a ``model``. The provider resolves
its tokenizer and tool parser through a
:class:`~proxyserver.model_registry.ModelRegistry` and **pins** the agent's
stream to that model on its first request. Direct callers of ``generate``
may omit ``model`` after the stream is pinned. A later request naming a
different model is rejected like any other strict-TITO violation, because
switching models mid-stream would mean switching tokenizers mid-stream.

Usage::

    from proxyserver.rollout_provider import DirectRolloutProvider, SGLangRolloutProvider, VLLMRolloutProvider

    # Slime or a VeRL-owned SGLang HTTP server:
    provider = SGLangRolloutProvider("http://<sglang-router-ip>:<port>")
    # or, for vLLM generation:
    provider = VLLMRolloutProvider("http://<vllm-router-ip>:<port>", context_length=32768)
    # or, for direct:
    provider = DirectRolloutProvider(["http://<engine-ip>:<port>/v1"],
                                     api_keys=["<key>"], inference_engine="vllm")

    prompt_ids, token_ids, log_probs, finish_reason, text, engine_meta = await provider.generate(
        messages=[...], session_id="trial_042", model="Qwen3.5-35B-A3B",
        max_tokens=2048,
    )
    parsed = await provider.parse_completion("trial_042", token_ids, text, finish_reason)
    # parsed.content, parsed.reasoning_content, parsed.tool_calls, parsed.finish_reason
"""

from __future__ import annotations
import asyncio
from copy import deepcopy
import base64
import io
import math
import numpy as np
import logging
from typing import Any, Callable
from uuid import uuid4
from .engines import ABORT, DEFAULT_MAX_TOKENS, LENGTH, SGLANG, VLLM, EngineAbort, EngineError, normalize_finish_reason, resolve_engine
from .model_registry import (SAMPLING_PARAM_KEYS, ModelRegistry, ResolvedModel,
                             UnknownModelError, resolve_profile, validate_sampling_overrides)
from .request_validation import validate_generation_parameters, validate_reasoning_effort
from .token_stream import ParsedCompletion, TokenStreamError, normalize_messages, validate_tools

logger = logging.getLogger(__name__)


async def _run_state_update(fn: Callable[..., Any], *args: Any) -> Any:
    """Finish a threaded state update before cancellation releases its session."""
    task = asyncio.create_task(asyncio.to_thread(fn, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Cancelling to_thread only cancels its awaiter. Let the thread finish
        # before session cleanup drops the model pin and token stream it owns.
        try:
            await task
        except Exception:
            pass
        raise


def _finite_logprob(value: Any, session_id: str | None) -> float:
    """Validate a sampled logprob before converting it for replay/recording."""
    if type(value) not in (int, float):
        raise EngineError(f"Session {session_id}: the rollout engine returned an invalid logprob")
    try:
        logprob = float(value)
    except OverflowError as e:
        raise EngineError(f"Session {session_id}: the rollout engine returned a non-finite logprob") from e
    if not math.isfinite(logprob):
        raise EngineError(f"Session {session_id}: the rollout engine returned a non-finite logprob")
    return logprob


def engine_http_limits() -> Any:
    """Connection-pool limits for engine HTTP clients (also used by the e2e harness).

    Idle connections expire before uvicorn's 5s keep-alive closes them, avoiding
    "Server disconnected" races on reuse. The caps restate httpx's defaults
    (100/20), which a bare ``Limits(keepalive_expiry=...)`` would lift.
    """
    import httpx

    return httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=2.0,
    )


#: Live generations an exactly repeated request may burn after a deterministic
#: engine 4xx before the stored rejection is re-raised without generating.
MAX_REJECTED_GENERATIONS = 2
#: 4xx statuses that are *not* deterministic verdicts on the request —
#: timeouts and throttling — and so never poison a turn.
TRANSIENT_4XX = (408, 425, 429)


class _RejectedTurn:
    """The last request of a session the engine rejected with a
    deterministic 4xx, and how many live generations it has burned."""

    __slots__ = ("messages", "error", "attempts", "reasoning_effort")

    def __init__(self, messages: list[dict[str, Any]], error: Exception, reasoning_effort: str | None = None) -> None:
        self.messages = messages  # normalized, as cached_turn compares them
        self.error = error        # re-raised verbatim on fail-fast
        self.attempts = 1
        self.reasoning_effort = reasoning_effort


class BaseRolloutProvider:
    """Strict-TITO completion flow, independent of the framework transport.

    Subclasses implement :meth:`_call_engine` — everything else (per-session
    model pinning and tokenizer resolution, per-session turn serialization,
    prompt construction, ``stop_reason`` normalization, abort/empty validation,
    decoding, committing the turn to the session's token stream, tool-call
    parsing) is shared, so every transport records identical training data.
    """

    def __init__(
        self,
        inference_engine: str | None = None,
        tokenizer_loader: Callable[[str], Any] | None = None,
        sampling_overrides: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            inference_engine: ``"vllm"`` or ``"sglang"``. Defaults to
                ``$INFERENCE_ENGINE``, then ``"vllm"``.
            tokenizer_loader: Optional ``(tokenizer_dir) -> tokenizer``
                override (the offline tests inject fake tokenizers here).
            sampling_overrides: Config-layer sampling policy that wins over
                request params; the runtime layer wins over it.
        """
        self.models = ModelRegistry(tokenizer_loader=tokenizer_loader)
        self._config_sampling_overrides = validate_sampling_overrides(sampling_overrides or {}, "sampling_overrides")
        # In-memory trainer layer (PUT /sampling_overrides), re-pushed every step.
        self._runtime_sampling_overrides: dict[str, Any] = {}
        self.engine = resolve_engine(inference_engine)
        # Per-stream state, dropped in release_session.
        self._session_models: dict[str, str] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._rejected_turns: dict[str, _RejectedTurn] = {}
        # Profiles already warned that bundled generation_config.json
        # defaults filled sampling knobs no request or override set.
        self._bundled_sampling_warned: set[str] = set()
        logger.info(
            "%s initialized (inference_engine=%s)",
            type(self).__name__, self.engine,
        )

    def release_session(self, session_id: str) -> None:
        """Free per-session state held by the provider.

        Drops the session's model pin and authoritative token stream.  Any
        router-side per-session state is the router's to evict. A transport
        that holds per-session state of its own (e.g. :class:`DirectRolloutProvider`'s
        sticky endpoint bindings) extends this method.
        """
        self._session_models.pop(session_id, None)
        self._session_locks.pop(session_id, None)
        self._rejected_turns.pop(session_id, None)
        self.models.drop_session(session_id)

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        """The session's turn-serialization lock (created on first use)."""
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        return lock

    # ------------------------------------------------------------------
    # Sampling overrides (training-side sampling policy)
    # ------------------------------------------------------------------

    def set_runtime_sampling_overrides(self, overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Replace the runtime sampling-override layer wholesale.

        The trainer's per-step push (``PUT /sampling_overrides``).  An empty
        mapping clears the layer, falling back to the config layer.  Returns
        :meth:`get_sampling_overrides` so the caller can verify the
        effective policy it just installed.
        """
        validated = validate_sampling_overrides(overrides, "sampling_overrides")
        self._runtime_sampling_overrides = validated
        logger.info("Runtime sampling overrides set: %s", validated or "{} (cleared)")
        return self.get_sampling_overrides()

    def get_sampling_overrides(self) -> dict[str, dict[str, Any]]:
        """The override layers: ``{"config": …, "runtime": …, "effective": …}``.

        ``effective`` — the runtime layer over the config layer — is what
        every request's sampling params receive on top of request-provided
        values.
        """
        return {
            "config": dict(self._config_sampling_overrides),
            "runtime": dict(self._runtime_sampling_overrides),
            "effective": {**self._config_sampling_overrides, **self._runtime_sampling_overrides},
        }

    # ------------------------------------------------------------------
    # Tokenizer identity (training–rollout consistency)
    # ------------------------------------------------------------------

    def tokenizer_fingerprint(self, model_name: str) -> dict[str, Any]:
        """The serving tokenizer's identity fingerprint for ``model_name``
        (``GET /tokenizer_fingerprint``): computed on the runtime tokenizer
        via the model registry, cached per profile.
        """
        return self.models.fingerprint_payload(model_name)

    # ------------------------------------------------------------------
    # Deterministic engine rejections (fail-fast on exact retries)
    # ------------------------------------------------------------------

    def _rejected_repeat(self, session_id: str, messages: list[dict[str, Any]],
                         reasoning_effort: str | None = None) -> Exception | None:
        """The stored rejection to re-raise, iff normalized ``messages`` match
        a request the engine already rejected with a deterministic 4xx
        :data:`MAX_REJECTED_GENERATIONS` times.

        A rejection slower than the caller's timeout would otherwise be retried
        forever; answering instantly gives the caller an error to stop on.
        """
        entry = self._rejected_turns.get(session_id)
        if entry is None or entry.attempts < MAX_REJECTED_GENERATIONS:
            return None  # no verdict yet, or still owed a live retry
        if normalize_messages(messages) != entry.messages or reasoning_effort != entry.reasoning_effort:
            return None
        status = getattr(getattr(entry.error, "response", None), "status_code", None)
        logger.error(
            "Session %s: the engine already rejected these normalized messages %d "
            "times (HTTP %s); failing fast without generating again so the "
            "caller receives the rejection instead of retrying forever",
            session_id, entry.attempts, status,
        )
        return entry.error

    def _note_engine_rejection(self, session_id: str, messages: list[dict[str, Any]], error: Exception,
                               reasoning_effort: str | None = None) -> None:
        """Record a deterministic engine rejection for :meth:`_rejected_repeat`.

        Only non-transient HTTP 4xx counts; 5xx and transport errors may succeed
        on retry.
        """
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status is None or not 400 <= status < 500 or status in TRANSIENT_4XX:
            return
        normalized = normalize_messages(messages)
        entry = self._rejected_turns.get(session_id)
        if entry is not None and entry.messages == normalized and entry.reasoning_effort == reasoning_effort:
            entry.attempts += 1
            entry.error = error
        else:
            self._rejected_turns[session_id] = _RejectedTurn(normalized, error, reasoning_effort)

    # ------------------------------------------------------------------
    # Per-session model resolution
    # ------------------------------------------------------------------

    def _resolve_for_session(self, session_id: str | None, model: str | None) -> ResolvedModel:
        """Resolve the session's model bundle, pinning on first use.

        The first request of a session pins it to the request's model; later
        requests may repeat that model or omit it, but never name another —
        switching models mid-session would switch tokenizers mid-stream and
        corrupt the recorded rollout.
        """
        if session_id is None:
            raise TokenStreamError("Strict token-in-token-out requires a session_id")
        pinned = self._session_models.get(session_id)
        if pinned is None:
            if not model:
                raise TokenStreamError(
                    f"Session {session_id}: the first request must name a model "
                    f"(the OpenAI 'model' field) so the proxy can pick its tokenizer"
                )
            resolved = self.models.resolve(model)
            self._session_models[session_id] = model
            return resolved
        if model and model != pinned:
            raise TokenStreamError(
                f"Session {session_id} is pinned to model {pinned!r} but the "
                f"request names {model!r}; a session cannot switch models "
                f"(and tokenizers) mid-stream"
            )
        return self.models.resolve(pinned)

    # ------------------------------------------------------------------
    # Transport hook
    # ------------------------------------------------------------------

    async def _call_engine(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        session_id: str | None,
        routed_experts_prior: tuple[int, int] | None = None,
    ) -> tuple[list[int], list[float], str | None, dict[str, Any]]:
        """Run one generation against the inference backend.

        Receives the vLLM-style ``sampling_params`` built by :meth:`generate`;
        transports speaking another dialect translate them.

        Returns:
            ``(token_ids, log_probs, stop_reason, engine_meta)`` as plain Python
            ``int``/``float``. ``stop_reason`` is in the engine's own vocabulary;
            ``engine_meta`` is :func:`_engine_meta` output (``{}`` if none).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict[str, Any]],
        session_id: str | None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        routed_experts_prior: tuple[int, int] | None = None,
        seed: int | None = None,
        stop_token_ids: list[int] | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[list[int], list[int], list[float], str, str, dict[str, Any]]:
        """Tokenize, call the rollout engine, return raw results.

        ``model`` selects the tokenizer and is pinned to the stream on its
        first request. Sampling precedence is::

            runtime overrides > config sampling_overrides > request
            > generation_config.json > engine

        A lost-response retry is re-served from the cached turn, and an exact
        repeat of a deterministically rejected request fails fast. Turns of one
        stream are serialized from model resolution through commit.

        Returns:
            ``(prompt_ids, token_ids, log_probs, finish_reason,
            completion_text, engine_meta)``; ``finish_reason`` is ``"stop"`` or
            ``"length"``, and ``engine_meta`` includes the resolved
            ``sampling_params``.
        """
        if session_id is None:
            raise TokenStreamError("Strict token-in-token-out requires a session_id")
        validate_tools(tools)
        try:
            validate_reasoning_effort(reasoning_effort)
        except ValueError as e:
            raise TokenStreamError(str(e)) from e
        tools = deepcopy(tools)
        async with self._session_lock(session_id):
            first_turn = session_id not in self._session_models
            resolved = await _run_state_update(self._resolve_for_session, session_id, model)
            try:
                validate_reasoning_effort(reasoning_effort, resolved.reasoning_efforts)
            except ValueError as e:
                raise TokenStreamError(str(e)) from e
            if resolved.reasoning_efforts:
                reasoning_effort = reasoning_effort or resolved.default_reasoning_effort
            else:
                reasoning_effort = None
            cached = await _run_state_update(resolved.token_streams.cached_turn, session_id, messages, reasoning_effort)
            if cached is not None:
                prompt_ids, token_ids, log_probs, finish_reason, engine_meta = cached
                completion_text = await asyncio.to_thread(resolved.tokenizer.decode, token_ids, skip_special_tokens=False)
                logger.warning(
                    "Session %s: normalized messages match the previous turn; "
                    "re-serving the cached completion (lost-response retry)",
                    session_id,
                )
                # Cached engine_meta keeps the weight version the tokens were sampled under.
                return prompt_ids, token_ids, log_probs, finish_reason, completion_text, engine_meta

            rejected = self._rejected_repeat(session_id, messages, reasoning_effort)
            if rejected is not None:
                raise rejected

            prompt_ids = await _run_state_update(resolved.token_streams.build_prompt, session_id, messages, tools, reasoning_effort)

            requested = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
            }
            sampling_params: dict[str, Any] = dict(resolved.sampling_defaults)
            sampling_params.update({k: v for k, v in requested.items() if v is not None})
            overrides = {**self._config_sampling_overrides, **self._runtime_sampling_overrides}
            sampling_params.update(overrides)
            # Bundled defaults silently shape the rollout distribution (e.g.
            # Qwen's top_k=20) unless the training side pins the policy.
            if resolved.profile not in self._bundled_sampling_warned:
                bundled = sorted(k for k in resolved.sampling_defaults if requested[k] is None and k not in overrides)
                if bundled:
                    self._bundled_sampling_warned.add(resolved.profile)
                    logger.warning(
                        "Profile %s: bundled generation_config.json supplies %s for requests that "
                        "omit them and no sampling override pins them; set sampling_overrides "
                        "when the trainer expects a fixed policy (logged once per profile)",
                        resolved.profile, {k: sampling_params[k] for k in bundled},
                    )
            sampling_params["max_tokens"] = max_tokens
            sampling_params["logprobs"] = True
            if stop is not None:
                sampling_params["stop"] = stop
            if seed is not None:
                sampling_params["seed"] = seed
            if stop_token_ids is not None:
                sampling_params["stop_token_ids"] = stop_token_ids
            try:
                validate_generation_parameters(sampling_params)
            except ValueError as e:
                raise TokenStreamError(str(e)) from e
            if resolved.stop_token_ids:
                # Harmony's message boundary is not an action boundary:
                # analysis must continue until a final answer or tool handoff.
                sampling_params["stop_token_ids"] = list(dict.fromkeys(
                    [*resolved.stop_token_ids, *(stop_token_ids or [])]
                ))
            logger.log(
                logging.INFO if first_turn else logging.DEBUG,
                "Session %s: sampling params %s (request keys: %s; override keys: %s)",
                session_id,
                {k: v for k, v in sampling_params.items() if k not in ("logprobs", "stop")},
                sorted(k for k, v in requested.items() if v is not None) or "none",
                sorted(overrides) or "none",
            )

            # Snapshot before awaiting: runtime controls may change mid-generation.
            actual_sampling = {k: sampling_params[k] for k in SAMPLING_PARAM_KEYS if k in sampling_params}
            try:
                # Pass the prior only when set: overrides without capture may omit the parameter.
                token_ids, log_probs, stop_reason, engine_meta = await self._call_engine(
                    prompt_ids, sampling_params, session_id,
                    **({"routed_experts_prior": routed_experts_prior} if routed_experts_prior else {}),
                )
            except Exception as e:
                self._note_engine_rejection(session_id, messages, e, reasoning_effort)
                raise
            self._rejected_turns.pop(session_id, None)
            engine_meta = {**engine_meta, "sampling_params": actual_sampling}

            if not isinstance(token_ids, list) or not isinstance(log_probs, list):
                raise EngineError(f"Session {session_id}: the rollout engine must return token/logprob lists")
            finish_reason = normalize_finish_reason(
                stop_reason,
                num_tokens=len(token_ids),
                max_tokens=sampling_params["max_tokens"],
            )

            # Validate before committing: an invalid completion must never
            # enter the authoritative token stream.
            if finish_reason == ABORT:
                raise EngineAbort(f"Session {session_id}: the rollout engine aborted the request")
            if not token_ids:
                # _sglang_generate also returns no tokens for misaligned logprobs.
                raise EngineError(f"Session {session_id}: the rollout engine returned no tokens")
            if len(log_probs) != len(token_ids):
                raise EngineError(
                    f"Session {session_id}: the rollout engine returned "
                    f"{len(log_probs)} logprobs for {len(token_ids)} tokens; "
                    f"refusing to record a misaligned turn"
                )
            if any(type(token) is not int or token < 0 for token in token_ids):
                raise EngineError(f"Session {session_id}: the rollout engine returned invalid token IDs")
            log_probs = [_finite_logprob(value, session_id) for value in log_probs]

            completion_text = await asyncio.to_thread(resolved.tokenizer.decode, token_ids, skip_special_tokens=False)

            if session_id not in self._session_models:
                # Released mid-generation by a direct caller; do not re-create
                # the stream. No await sits between this check and the return.
                logger.info(
                    "Session %s was released while this turn was generating; "
                    "returning the turn without re-creating its token stream",
                    session_id,
                )
            else:
                resolved.token_streams.commit(
                    session_id, messages, prompt_ids, token_ids,
                    log_probs=log_probs, finish_reason=finish_reason,
                    engine_meta=engine_meta,
                    tools=tools,
                    reasoning_effort=reasoning_effort,
                )

            return prompt_ids, token_ids, log_probs, finish_reason, completion_text, engine_meta

    # ------------------------------------------------------------------
    # Tool-call parsing
    # ------------------------------------------------------------------

    async def parse_completion(
        self,
        session_id: str | None,
        token_ids: list[int],
        completion_text: str,
        finish_reason: str,
        tools: list[dict] | None = None,
    ) -> ParsedCompletion:
        """Decode and cache the complete response for one committed turn.

        Content, reasoning, stable OpenAI tool-call IDs, and finish reason are
        published together. HTTP formatting, recording, and retries consume
        the same result. A parser failure leaves the cache empty so an exact
        retry can repair it without resampling. Complete tool calls upgrade
        the finish reason unless the engine reported length truncation.
        """
        pinned = self._session_models.get(session_id) if session_id is not None else None
        resolved = self.models.resolve(pinned) if pinned else None
        turn = resolved.token_streams.committed_turn(session_id, token_ids) if resolved else None
        if turn is not None and turn.parsed_completion is not None:
            return deepcopy(turn.parsed_completion)
        tool_parser = resolved.tool_parser if resolved else None
        if tool_parser is None:
            return ParsedCompletion(completion_text, None, finish_reason)

        # Parsing recovery must use the schema that produced the committed
        # prompt, even when a retry supplies different tool definitions.
        decoded = await tool_parser.extract_completion(
            token_ids, tools=turn.tools if turn is not None else tools, completion_text=completion_text,
        )
        parsed_calls = decoded.tool_calls
        openai_tool_calls = [
            {
                "id": f"call_{uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                },
            }
            for tc in parsed_calls or []
        ]
        if parsed_calls and finish_reason != LENGTH:
            finish_reason = "tool_calls"
        result = ParsedCompletion(
            content=decoded.content, tool_calls=openai_tool_calls or None,
            finish_reason=finish_reason, reasoning_content=decoded.reasoning_content,
        )
        if turn is not None:
            # A concurrent parser may have finished first. Publish once and
            # return copies so neither callers nor the recorder mutate replay
            # state. A failed parse never populates the cache and can retry.
            if turn.parsed_completion is None:
                turn.parsed_completion = result
            return deepcopy(turn.parsed_completion)
        return result


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
            _pack_routed_experts, meta_info, len(prompt_ids), len(token_ids), session_id,
            routed_experts_prior,
        )

    return token_ids, log_probs, stop_reason, meta


def _routing_headers(session_id: str | None) -> dict[str, str] | None:
    """Per-stream prefix-cache affinity for ``consistent_hashing`` routers;
    other router policies ignore the header."""
    return {"X-SMG-Routing-Key": session_id} if session_id else None


def _engine_meta(meta_info: dict[str, Any]) -> dict[str, Any]:
    """Engine-reported turn metadata: the policy weight version, if any."""
    meta: dict[str, Any] = {}
    weight_version = meta_info.get("weight_version")
    if weight_version is not None:
        meta["weight_version"] = str(weight_version)
    return meta


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


def _clamp_to_context_window(
    sampling_params: dict[str, Any],
    prompt_len: int,
    context_length: int | None,
    endpoint: str,
    session_id: str | None,
) -> None:
    """Clamp ``sampling_params["max_tokens"]`` (in place) to what
    ``endpoint``'s context window can still hold.

    Both engines reject ``prompt + max_tokens`` beyond their window up front;
    a clamped turn generates what fits instead. The in-place write is also the
    budget ``generate()`` falls back on when the engine reports no recognized
    finish reason. ``context_length=None`` disables the clamp.
    """
    if context_length is None:
        return
    budget = context_length - prompt_len
    if budget <= 0:
        raise EngineError(
            f"Session {session_id}: prompt is {prompt_len} tokens but "
            f"{endpoint} serves a {context_length}-token context "
            f"window; the session has no room left to generate"
        )
    max_tokens = sampling_params.get("max_tokens")
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    if max_tokens > budget:
        logger.warning(
            "Session %s: clamping max_tokens %d -> %d to fit the "
            "%d-token context window (prompt is %d tokens)",
            session_id, max_tokens, budget, context_length, prompt_len,
        )
        sampling_params["max_tokens"] = budget


class _RouterTransport:
    """Router HTTP client lifecycle and R3 capture layers of the generate transports."""

    _http: Any
    _config_return_routed_experts: bool
    _runtime_return_routed_experts: bool | None

    def set_runtime_routed_experts(self, enabled: bool | None) -> dict[str, Any]:
        """Set the trainer's capture layer (``PUT /routed_experts``); ``None`` restores config."""
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError(f"enabled must be true, false or null, got {enabled!r}")
        self._runtime_return_routed_experts = enabled
        logger.info("Runtime routed-experts capture set: %s", "cleared (config applies)" if enabled is None else enabled)
        return self.get_routed_experts_config()

    def get_routed_experts_config(self) -> dict[str, Any]:
        """The capture layers: ``{"config": …, "runtime": …, "effective": …}``."""
        runtime = self._runtime_return_routed_experts
        return {
            "config": self._config_return_routed_experts,
            "runtime": runtime,
            "effective": self._config_return_routed_experts if runtime is None else runtime,
        }

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()


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

        import httpx

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


def _truncates_support(params: dict[str, Any]) -> bool:
    """Whether the policy samples from a truncated support (top-p, top-k, or min-p).

    A trainer recomputing logprobs needs the support of every truncated
    distribution, not only the nucleus: bundled defaults such as Qwen's
    ``top_k=20`` truncate even when ``top_p`` is pinned to 1.
    """
    params = {k: v for k, v in params.items() if v is not None}
    # vLLM raises small positive temperatures to 0.01 before checking for
    # greedy sampling, so only zero disables the sampling cutoffs.
    if params.get("temperature", 1.0) == 0:
        return False
    return (
        params.get("top_p", 1.0) < 1.0
        or params.get("top_k", -1) > 0
        or params.get("min_p", 0.0) > 0.0
    )


class _MissingSamplingMask(EngineError):
    """The engine returned no sampling mask for a truncated policy."""


def _vllm_sampling_mask(choice: dict[str, Any], token_ids: list[int], session_id: str | None) -> dict[str, Any]:
    """Flatten vLLM's per-token sampling support for loss replay.

    Each sampled token must belong to its reported support. Empty/missing
    rows cannot be replaced by the sampled token alone: that would change
    the distribution the trainer uses to recompute logprobs.
    """
    mask = choice.get("sampling_mask")
    if mask is None:
        raise _MissingSamplingMask(
            f"Session {session_id}: vLLM requires a sampling mask for every generated token "
            "(top_p < 1, top_k > 0, or min_p > 0); enable return_sampling_mask at engine startup"
        )
    if not isinstance(mask, list) or len(mask) != len(token_ids):
        raise EngineError(f"Session {session_id}: vLLM returned a sampling mask that does not cover every generated token")
    ids: list[int] = []
    offsets = [0]
    for token, row in zip(token_ids, mask):
        if (
            not isinstance(row, list) or not row
            or any(type(value) is not int or value < 0 for value in row)
            or token not in row or len(set(row)) != len(row)
        ):
            raise EngineError(f"Session {session_id}: vLLM returned invalid sampling-mask token IDs")
        ids.extend(row)
        offsets.append(len(ids))
    return {"rollout_top_p_token_ids": ids, "rollout_top_p_token_offsets": offsets}


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
    capture_mask = _truncates_support(params)
    # Sampling masks are enabled at engine startup, in both Vime and
    # upstream vLLM. return_sampling_mask is not a SamplingParams field.
    if return_routed_experts:
        params["routed_experts_prompt_start"] = routed_experts_prior[0] if routed_experts_prior else 0
    resp = await http.post(
        "/inference/v1/generate",
        json={"token_ids": prompt_ids, "sampling_params": params},
        headers={"x-session-id": session_id} if session_id else None,
    )
    resp.raise_for_status()
    output = await asyncio.to_thread(resp.json) if capture_mask or return_routed_experts else resp.json()
    choices = output.get("choices") if isinstance(output, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise EngineError(f"Session {session_id}: vLLM must return exactly one generation choice")
    choice = choices[0]
    stop_reason = choice.get("finish_reason")
    if isinstance(stop_reason, dict):
        stop_reason = stop_reason.get("type")
    if stop_reason == "cancelled":
        stop_reason = ABORT
    meta = _engine_meta(output)
    if stop_reason == ABORT:
        return [], [], ABORT, meta
    if stop_reason not in (None, "stop", "length"):
        raise EngineError(f"Session {session_id}: unknown vLLM finish_reason {stop_reason!r}")
    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list) or any(type(token) is not int or token < 0 for token in token_ids):
        raise EngineError(f"Session {session_id}: vLLM returned invalid token_ids")
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
    if capture_mask:
        meta.update(await asyncio.to_thread(_vllm_sampling_mask, choice, token_ids, session_id))
    if return_routed_experts:
        meta["routed_experts"] = await asyncio.to_thread(
            _pack_vllm_routed_experts, choice, len(prompt_ids), len(token_ids), session_id,
            routed_experts_prior,
        )
    return token_ids, log_probs, stop_reason, meta


class VLLMRolloutProvider(_RouterTransport, BaseRolloutProvider):
    """Native vLLM HTTP ``/inference/v1/generate``, used by ``vllm_generate``.

    The deployment owns the workers and load balancing; the proxy speaks
    their native HTTP dialect directly. ``x-session-id`` carries the agent
    stream for routers using ``consistent_hash``. Configure context_length
    to match ``--vllm-max-model-len``; router model discovery is not required.

    Sampling support is captured automatically when the policy truncates it
    (top_p < 1, top_k > 0, or min_p > 0); enable return_sampling_mask at engine
    startup. Once the engine answers a truncated request without a mask, later
    truncated requests fail before generating; restart the proxy after enabling
    masks. Routing replay is opt-in and requires compatible engine output.
    Weight updates must drain rollouts first: a record holds one policy version per turn,
    so PipelineRL's mid-generation updates are not supported.
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

        import httpx

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
        # Set when the engine answers a truncated request without a sampling
        # mask: each later truncated request would burn a full generation only
        # to be rejected, so they fail before reaching the engine instead.
        self._engine_lacks_masks = False
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
        if self._engine_lacks_masks and _truncates_support(sampling_params):
            raise EngineError(
                f"Session {session_id}: the engine returned no sampling mask for an earlier "
                "request with top_p < 1, top_k > 0, or min_p > 0; refusing before generating. "
                "Enable return_sampling_mask at engine startup and restart the proxy, or pin "
                "top_p: 1.0, top_k: -1, and min_p: 0.0"
            )
        _clamp_to_context_window(
            sampling_params, len(prompt_ids), self._context_window, self.router_url, session_id,
        )
        try:
            return await _vllm_generate(
                self._http, prompt_ids, sampling_params, session_id,
                return_routed_experts=self.get_routed_experts_config()["effective"],
                routed_experts_prior=routed_experts_prior,
            )
        except _MissingSamplingMask:
            self._engine_lacks_masks = True
            raise


def normalize_engine_endpoints(
    base_urls: list[str] | str,
    api_keys: list[str] | str | None,
) -> tuple[list[str], list[str | None]]:
    """Normalize and validate the direct transport's endpoint wiring.

    Returns ``(roots, keys)``: server roots (whitespace and a trailing
    ``/v1`` stripped) and exactly one api_key per root — a single key is
    broadcast to every URL, ``None`` means unauthenticated.
    """
    if isinstance(base_urls, str):
        base_urls = [base_urls]
    roots = [str(url).strip().rstrip("/").removesuffix("/v1") for url in base_urls]
    if not roots:
        raise ValueError("direct transport needs at least one inference_engine_base_url")
    blank = [i for i, root in enumerate(roots) if not root]
    if blank:
        raise ValueError(
            f"direct transport got blank inference_engine_base_url "
            f"entr{'y' if len(blank) == 1 else 'ies'} at position(s) {blank}"
        )
    if api_keys is None or isinstance(api_keys, str):
        keys: list[str | None] = [api_keys] * len(roots)
    elif len(api_keys) == 1 and len(roots) > 1:
        keys = list(api_keys) * len(roots)
    elif len(api_keys) != len(roots):
        raise ValueError(
            f"direct transport got {len(api_keys)} api_keys for {len(roots)} "
            f"base URLs; provide one per URL or a single shared key"
        )
    else:
        keys = list(api_keys)
    return roots, keys


class DirectRolloutProvider(BaseRolloutProvider):
    """direct transport: the vLLM / SGLang engines over plain HTTP, strictly TITO.

    Token-ID prompts go to vLLM's ``/v1/completions`` (with
    ``return_tokens_as_token_ids``) or SGLang's native ``/generate``. Each
    stream is bound round-robin to one endpoint for prefix-cache reuse.
    Each endpoint's served model and context window are discovered lazily
    (vLLM ``/v1/models``; SGLang ``/get_model_info`` and ``/get_server_info``),
    and the served model must map to the request's tokenizer profile.
    """

    def __init__(
        self,
        base_urls: list[str] | str,
        inference_engine: str | None = None,
        api_keys: list[str] | str | None = None,
        request_timeout: float = 900.0,
        tokenizer_loader: Callable[[str], Any] | None = None,
        sampling_overrides: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            base_urls: OpenAI base URLs of the engines, one per running
                instance, e.g. ``["http://10.0.1.5:8000/v1"]`` (a trailing
                ``/v1`` is optional — routes are resolved from the root).
            inference_engine: ``"vllm"`` or ``"sglang"``. Defaults to
                ``$INFERENCE_ENGINE``, then ``"vllm"``.
            api_keys: Bearer tokens matching ``base_urls`` — one per URL, or
                a single key broadcast to all, or ``None`` for
                unauthenticated engines.
            request_timeout: Per-request HTTP timeout (seconds).
            tokenizer_loader: Optional tokenizer-loading override (see
                :class:`BaseRolloutProvider`).
        """
        roots, api_keys = normalize_engine_endpoints(base_urls, api_keys)

        super().__init__(inference_engine, tokenizer_loader, sampling_overrides=sampling_overrides)

        import httpx

        self.base_urls = roots
        self._clients = [
            httpx.AsyncClient(
                base_url=root,
                headers={"Authorization": f"Bearer {key}"} if key else None,
                timeout=request_timeout,
                limits=engine_http_limits(),
            )
            for root, key in zip(roots, api_keys)
        ]
        self._session_endpoints: dict[str, int] = {}
        self._rr_index = 0
        self._model_ids: list[str | None] = [None] * len(roots)
        self._model_lens: list[int | None] = [None] * len(roots)
        self._model_lock = asyncio.Lock()
        self._context_windows = [_SGLangContextWindow(self._model_lock) for _ in roots]
        logger.info(
            "Direct rollout provider routing to %d %s endpoint(s): %s",
            len(roots), self.engine, ", ".join(roots),
        )

    def release_session(self, session_id: str) -> None:
        """Free the session's model pin, token stream, and sticky endpoint binding."""
        self._session_endpoints.pop(session_id, None)
        super().release_session(session_id)

    def _acquire_endpoint(self, session_id: str | None) -> int:
        """Resolve the endpoint index for a session — sticky, round-robin on first use."""
        if session_id is not None and session_id in self._session_endpoints:
            return self._session_endpoints[session_id]
        index = self._rr_index % len(self._clients)
        self._rr_index = index + 1
        if session_id is not None:
            self._session_endpoints[session_id] = index
        return index

    async def _vllm_model_id(self, index: int) -> str:
        """The served model id of endpoint ``index``, from ``GET /v1/models``
        (fetched once and cached; the engine serves exactly one model).
        The same response carries ``max_model_len``, cached alongside for
        the context-window clamp in :meth:`_call_engine`."""
        if self._model_ids[index] is None:
            async with self._model_lock:
                if self._model_ids[index] is None:
                    resp = await self._clients[index].get("/v1/models")
                    resp.raise_for_status()
                    models = resp.json().get("data") or []
                    if not models:
                        raise EngineError(f"{self.base_urls[index]}: /v1/models reported no served model")
                    max_len = models[0].get("max_model_len")
                    self._model_lens[index] = int(max_len) if max_len else None
                    self._model_ids[index] = models[0]["id"]
        return self._model_ids[index]

    async def _sglang_model_id(self, index: int) -> str:
        """Discover and cache a direct SGLang endpoint's model identity.

        The served alias follows the same mapping contract as vLLM's model
        ID. Older SGLang versions expose only model_path, which the shared
        validator accepts as a mapped full path or checkpoint basename.
        Missing or malformed identity must fail before generation, even
        when context-window discovery could proceed without metadata.
        """
        if self._model_ids[index] is None:
            async with self._model_lock:
                if self._model_ids[index] is None:
                    resp = await self._clients[index].get("/get_model_info")
                    resp.raise_for_status()
                    info = resp.json()
                    model = (info.get("served_model_name") or info.get("model_path")) if isinstance(info, dict) else None
                    if not isinstance(model, str) or not model.strip():
                        raise EngineError(
                            f"{self.base_urls[index]}: /get_model_info reported no valid "
                            "served_model_name or model_path; refusing unverified token IDs"
                        )
                    self._model_ids[index] = model.strip()
        return self._model_ids[index]

    async def _sglang_context_length(self, index: int) -> int | None:
        """Endpoint ``index``'s discovered window; None disables clamping."""
        return await self._context_windows[index].get(self._clients[index], self.base_urls[index])

    def _validate_served_model(self, index: int, served_model: str, session_id: str | None) -> None:
        """Require an explicit profile mapping before sending token IDs.

        Model aliases and checkpoint paths are supported through mapping.json;
        a registered full name takes precedence over its basename.
        """
        claimed = self._session_models.get(session_id) if session_id is not None else None
        if claimed is None:
            raise EngineError(f"Session {session_id}: cannot verify the backend without a pinned model")
        for name in (served_model, served_model.rstrip("/").rsplit("/", 1)[-1]):
            try:
                served_profile = resolve_profile(name)
                break
            except UnknownModelError:
                continue
        else:
            raise EngineError(
                f"{self.base_urls[index]} serves unrecognized model {served_model!r}; "
                "add its name to tokenization/mapping.json with the verified tokenizer profile before sending token IDs"
            )
        claimed_profile = resolve_profile(claimed)
        if served_profile.name != claimed_profile.name:
            raise EngineError(
                f"Session {session_id} requests model {claimed!r} (profile {claimed_profile.name}) "
                f"but endpoint {self.base_urls[index]} serves {served_model!r} "
                f"(profile {served_profile.name}); refusing incompatible token IDs"
            )

    async def _call_engine(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        session_id: str | None,
        routed_experts_prior: tuple[int, int] | None = None,
    ) -> tuple[list[int], list[float], str | None, dict[str, Any]]:
        index = self._acquire_endpoint(session_id)
        http = self._clients[index]
        if self.engine == SGLANG:
            served_model = await self._sglang_model_id(index)
            self._validate_served_model(index, served_model, session_id)
            _clamp_to_context_window(
                sampling_params, len(prompt_ids),
                await self._sglang_context_length(index), self.base_urls[index], session_id,
            )
            return await _sglang_generate(http, prompt_ids, sampling_params, session_id)

        served_model = await self._vllm_model_id(index)
        self._validate_served_model(index, served_model, session_id)

        _clamp_to_context_window(
            sampling_params, len(prompt_ids),
            self._model_lens[index], self.base_urls[index], session_id,
        )
        max_tokens = sampling_params.get("max_tokens", DEFAULT_MAX_TOKENS)

        payload: dict[str, Any] = {
            "model": served_model,
            "prompt": prompt_ids,  # token IDs in
            "max_tokens": max_tokens,
            # vLLM's own dialect: logprobs is an int of extra top-logprobs
            # per position; 0 returns just the sampled token's logprob.
            "logprobs": 0 if sampling_params.get("logprobs") else None,
            "return_tokens_as_token_ids": True,
        }
        # top_k/min_p/repetition_penalty are vLLM extensions /v1/completions accepts.
        for key in (*SAMPLING_PARAM_KEYS, "stop", "seed", "stop_token_ids"):
            if sampling_params.get(key) is not None:
                payload[key] = sampling_params[key]

        resp = await http.post("/v1/completions", json=payload)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]

        logprobs_blob = choice.get("logprobs") or {}
        tokens = logprobs_blob.get("tokens") or []
        token_logprobs = logprobs_blob.get("token_logprobs") or []
        if (
            not isinstance(tokens, list) or not isinstance(token_logprobs, list)
            or len(tokens) != len(token_logprobs)
        ):
            raise EngineError(f"Session {session_id}: vLLM returned misaligned tokens and token_logprobs")
        token_ids = []
        log_probs = []
        # Validate each raw pair so filtering cannot shift positions.
        for index, (token, lp) in enumerate(zip(tokens, token_logprobs)):
            token_id = token.removeprefix("token_id:") if isinstance(token, str) else ""
            if (
                not isinstance(token, str) or not token.startswith("token_id:")
                or not token_id.isascii() or not token_id.isdecimal()
                or isinstance(lp, bool) or not isinstance(lp, (int, float))
            ):
                raise EngineError(f"Session {session_id}: vLLM returned an invalid token/logprob pair at position {index}")
            token_ids.append(int(token_id))
            log_probs.append(_finite_logprob(lp, session_id))
        # The OpenAI-compatible response reports no weight version.
        return token_ids, log_probs, choice.get("finish_reason"), {}

    async def aclose(self) -> None:
        """Close the underlying HTTP clients."""
        for client in self._clients:
            await client.aclose()
