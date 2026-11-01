"""Shared strict-TITO rollout flow and transport helpers.

Owns model pinning, sampling policy, retry state, token-stream commits, and
completion parsing. Concrete providers implement the engine HTTP protocol;
this module does not import any concrete provider.
"""

from __future__ import annotations
import asyncio
import logging
import math
from copy import deepcopy
from typing import Any, Callable
from uuid import uuid4
import httpx
from .engines import ABORT, DEFAULT_MAX_TOKENS, LENGTH, EngineAbort, EngineError, normalize_finish_reason, resolve_engine
from .model_registry import (
    SAMPLING_PARAM_KEYS, ModelRegistry, ResolvedModel,
    apply_temperature_floor, validate_sampling_overrides,
)
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
    return httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=2.0,
    )


#: Maximum engine calls for the same deterministically rejected request,
#: including the initial call, before exact retries re-raise its stored error.
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
        logger.info("%s initialized (inference_engine=%s)", type(self).__name__, self.engine)

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

        ``config`` and ``runtime`` retain requested values. ``effective`` is
        the runtime layer over config with the temperature floor applied.
        Request/model defaults are resolved separately during generation.
        """
        return {
            "config": dict(self._config_sampling_overrides),
            "runtime": dict(self._runtime_sampling_overrides),
            "effective": apply_temperature_floor({**self._config_sampling_overrides, **self._runtime_sampling_overrides}),
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

    def _resolve_for_session(self, session_id: str, model: str | None) -> ResolvedModel:
        """Resolve the session's model bundle, pinning on first use.

        The first request of a session pins it to the request's model; later
        requests may repeat that model or omit it, but never name another —
        switching models mid-session would switch tokenizers mid-stream and
        corrupt the recorded rollout.
        """
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
            ``int``/``float``. ``stop_reason`` is normalized by the transport
            where needed, then by :func:`normalize_finish_reason` in ``generate``.
            ``engine_meta`` may include the weight version and routing capture;
            absent metadata is ``{}``.
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

        A lost-response retry with unchanged effective ``reasoning_effort`` is
        re-served from the cached turn. A changed effort is rejected on retries
        and continuations. Deterministic engine rejections are re-raised without
        another engine call once ``MAX_REJECTED_GENERATIONS`` calls have failed
        for the same normalized messages and effective effort. Turns of one
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
            # Send an explicit temperature even without a bundled/request value,
            # so an engine's own defaults cannot silently select greedy decoding.
            sampling_params: dict[str, Any] = {"temperature": 1.0, **resolved.sampling_defaults}
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
            # Apply after validation and precedence, before transport and the
            # provenance snapshot. Exact retries retain their cached policy.
            sampling_params = apply_temperature_floor(sampling_params)
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


def _engine_meta(meta_info: dict[str, Any]) -> dict[str, Any]:
    """Engine-reported turn metadata: the policy weight version, if any."""
    meta: dict[str, Any] = {}
    weight_version = meta_info.get("weight_version")
    if weight_version is not None:
        meta["weight_version"] = str(weight_version)
    return meta


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
