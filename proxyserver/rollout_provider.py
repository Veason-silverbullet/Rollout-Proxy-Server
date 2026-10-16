"""Rollout providers: route proxy requests to the rollout engines, strictly
token-in-token-out.

A provider owns the inference-side half of a completion: strict-TITO prompt
construction (:class:`~proxyserver.token_stream.TokenStreamManager`), the
transport to the rollout servers, engine-specific ``stop_reason``
normalization (:mod:`proxyserver.engines`), and tool-call parsing.  OpenAI
response formatting is *not* its job — the proxy server does that with
:func:`proxyserver.server.build_openai_response`.

Providers are selected by ``transport_mode`` in ``configs/{engine}-{transport}.yaml``:

* :class:`SGLangRolloutProvider` (``transport_mode: slime``) — HTTP through
  slime's ``sgl-router``, which load-balances a pool of SGLang servers.  The
  router is a transparent HTTP proxy, so this provider speaks SGLang's native
  ``/generate`` dialect directly (``input_ids`` in, ``output_ids`` +
  ``meta_info`` out) and performs the vLLM-style→SGLang sampling-param
  translation itself.
* :class:`VLLMRolloutProvider` (``transport_mode: vime``) — HTTP through
  vime's ``vllm-router``, speaking ``/inference/v1/generate`` (``token_ids``
  in, ``choices`` with sampled token IDs, logprobs, and replay metadata out).
* ``transport_mode: verl`` — VeRL-owned HTTP servers, using
  :class:`SGLangRolloutProvider` for SGLang or :class:`VLLMRolloutProvider`
  for vLLM according to ``inference_engine``.
* :class:`DirectRolloutProvider` (``transport_mode: direct``) — plain HTTP
  straight to the vLLM / SGLang engines themselves, no training framework in
  the path.  Token-ID prompts go to vLLM's ``/v1/completions``
  (``return_tokens_as_token_ids``) or SGLang's ``/generate``; endpoints come
  from ``inference_engine_base_url`` / ``inference_engine_api_key`` and each
  session is sticky to one endpoint (prefix cache).

The backend is selected with ``inference_engine`` (``"vllm"`` or
``"sglang"``); see :mod:`proxyserver.engines` for what it changes.  slime is
SGLang-only; vime is vLLM-only; verl and direct support both engines.

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
    # or, for vime:
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
from .engines import ABORT, DEFAULT_MAX_TOKENS, LENGTH, SGLANG, VLLM, EngineAbort, EngineError, get_adapter
from .model_registry import (SAMPLING_PARAM_KEYS, ModelRegistry, ResolvedModel,
                             UnknownModelError, resolve_profile, validate_sampling_overrides)
from .token_stream import ParsedCompletion, TokenStreamError, normalize_messages

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
    """Connection-pool limits for HTTP clients that talk to the engines
    (the slime/vime/direct transports here; the e2e harness imports this too, so
    the tests exercise exactly the production connection behavior).

    The engines serve over uvicorn, whose default keep-alive timeout is 5s —
    the same 5s httpx keeps an idle connection reusable.  A connection idle
    for ~5s can therefore be reused at the exact moment the server closes it,
    failing the request with "Server disconnected without sending a
    response" and, through it, the whole trial.  Expiring idle connections
    well before the server does removes that race; requests after a longer
    idle gap (agents think for minutes between turns) simply open a fresh
    connection, as they always did.

    The pool caps restate httpx's ``DEFAULT_LIMITS`` (100/20) explicitly: a
    bare ``Limits(keepalive_expiry=...)`` would silently *lift* them to
    unbounded (its own field defaults are ``None``), letting N concurrent
    sessions open N sockets to one engine instead of queueing at the pool.
    """
    import httpx

    return httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=2.0,
    )


#: How many live generations one turn's request may burn after the engine
#: rejects it with a deterministic 4xx before an exact repeat of it is failed
#: fast (the stored rejection re-raised without generating again).  One live
#: retry tolerates an engine blip that mislabels itself 4xx; past that the
#: rejection is treated as the deterministic verdict it claims to be.
MAX_REJECTED_GENERATIONS = 2
#: 4xx statuses that are *not* deterministic verdicts on the request —
#: timeouts and throttling — and so never poison a turn.
TRANSIENT_4XX = (408, 425, 429)


class _RejectedTurn:
    """The last request of a session the engine rejected with a
    deterministic 4xx, and how many live generations it has burned."""

    __slots__ = ("messages", "error", "attempts")

    def __init__(self, messages: list[dict[str, Any]], error: Exception) -> None:
        self.messages = messages  # normalized, as cached_turn compares them
        self.error = error        # re-raised verbatim on fail-fast
        self.attempts = 1


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
        tool_parser_factory: Callable[[str, Any], Any] | None = None,
        inference_engine: str | None = None,
        tokenizer_loader: Callable[[str], Any] | None = None,
        sampling_overrides: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            tool_parser_factory: Optional callable
                ``(tool_call_parser, tokenizer) -> tool_parser`` supplied by
                the training framework (e.g. verl's
                ``ToolParser.get_tool_parser``); it receives each model's
                tool-call parser name declared by the mapped model profile.
                Without it the built-in parsers
                (:mod:`proxyserver.tokenization.tool_parser`) serve the name.  Built per
                tokenizer, since a parser is bound to the tokenizer it reads
                tokens with.
            inference_engine: ``"vllm"`` or ``"sglang"``. Defaults to
                ``$INFERENCE_ENGINE``, then ``"vllm"``.
            tokenizer_loader: Optional ``(tokenizer_dir) -> tokenizer``
                override for the registry's default
                ``AutoTokenizer.from_pretrained`` (the offline tests
                inject fake tokenizers here).
            sampling_overrides: Optional training-side sampling policy that
                wins *over* request-provided params (the
                ``sampling_overrides`` config field, restricted to
                :data:`~proxyserver.model_registry.SAMPLING_PARAM_KEYS`) —
                the lever for pinning the rollout distribution when the
                agent side cannot be trusted to.  A runtime layer pushed via
                :meth:`set_runtime_sampling_overrides` wins over this one.
        """
        #: Per-model (tokenizer, tool parser, token streams) bundles,
        #: resolved lazily from the model names requests carry.
        self.models = ModelRegistry(tool_parser_factory, tokenizer_loader)
        self._config_sampling_overrides = validate_sampling_overrides(sampling_overrides or {}, "sampling_overrides")
        # The trainer-pushed layer (PUT /sampling_overrides); wins over the
        # config layer.  In-memory only — a restart falls back to the config
        # layer, which is why the trainer re-pushes and verifies every step.
        self._runtime_sampling_overrides: dict[str, Any] = {}
        self.engine = get_adapter(inference_engine)
        # session_id -> the model name pinned on the session's first request.
        self._session_models: dict[str, str] = {}
        # session_id -> turn-serialization lock (see generate()); created on
        # first use, dropped in release_session.
        self._session_locks: dict[str, asyncio.Lock] = {}
        # session_id -> the request the engine last rejected with a
        # deterministic 4xx (see _rejected_repeat); cleared by the session's
        # next successful engine call, dropped in release_session.
        self._rejected_turns: dict[str, _RejectedTurn] = {}
        logger.info(
            "%s initialized (inference_engine=%s)",
            type(self).__name__, self.engine.name,
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
        via the model registry, cached per profile.  Raises
        :class:`~proxyserver.model_registry.UnknownModelError` for an
        unmapped name.  Synchronous and CPU-bound on first call (it hashes
        the full vocab) — the HTTP layer threads it off the event loop.
        """
        return self.models.fingerprint_payload(model_name)

    # ------------------------------------------------------------------
    # Deterministic engine rejections (fail-fast on exact retries)
    # ------------------------------------------------------------------

    def _rejected_repeat(self, session_id: str, messages: list[dict[str, Any]]) -> Exception | None:
        """The stored rejection to re-raise, iff normalized ``messages`` match
        a request the engine already rejected with a deterministic 4xx
        :data:`MAX_REJECTED_GENERATIONS` times.

        A 4xx is the engine's verdict on the request itself, so an exact
        repeat earns the exact same verdict — but only after burning another
        full generation to hear it.  When the rejection takes longer to
        arrive than the caller's own timeout, the caller never hears it at
        all: it retries the identical request, each retry generates for
        minutes and dies the same way, and the stream wedges forever while
        the engine burns (an engine emitting NaN logprobs on long-context
        turns — surfaced by vLLM as HTTP 400 "Out of range float values are
        not JSON compliant: nan" — wedged four rollouts exactly so).
        Answering the repeat instantly with the stored rejection puts the
        error in front of the caller while it is still connected, giving the
        loop something to break on — the fail-fast twin of
        :meth:`~proxyserver.token_stream.TokenStreamManager.cached_turn`'s
        replay cap.
        """
        entry = self._rejected_turns.get(session_id)
        if entry is None or entry.attempts < MAX_REJECTED_GENERATIONS:
            return None  # no verdict yet, or still owed a live retry
        if normalize_messages(messages) != entry.messages:
            return None
        status = getattr(getattr(entry.error, "response", None), "status_code", None)
        logger.error(
            "Session %s: the engine already rejected these normalized messages %d "
            "times (HTTP %s); failing fast without generating again so the "
            "caller receives the rejection instead of retrying forever",
            session_id, entry.attempts, status,
        )
        return entry.error

    def _note_engine_rejection(self, session_id: str, messages: list[dict[str, Any]], error: Exception) -> None:
        """Record a deterministic engine rejection of this request, for
        :meth:`_rejected_repeat` to fail an exact retry fast.

        Only HTTP 4xx counts (duck-typed via ``error.response.status_code``
        so every HTTP transport matches without this module importing its
        client library): 5xx and transport errors are the engine's problem
        and a retry may genuinely succeed, as may :data:`TRANSIENT_4XX`.
        """
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status is None or not 400 <= status < 500 or status in TRANSIENT_4XX:
            return
        normalized = normalize_messages(messages)
        entry = self._rejected_turns.get(session_id)
        if entry is not None and entry.messages == normalized:
            entry.attempts += 1
            entry.error = error
        else:
            self._rejected_turns[session_id] = _RejectedTurn(normalized, error)

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

        Receives the vLLM-style ``sampling_params`` built by :meth:`generate`
        (``max_tokens``, ``logprobs``, plus the sampling knobs resolved from
        the request and the model's bundled defaults — ``temperature``,
        ``top_p``, ``top_k``, ``min_p``, ``repetition_penalty``, ``presence_penalty``,
        ``frequency_penalty``, optional ``stop``); a transport whose backend
        speaks another dialect translates them itself.

        Returns:
            ``(token_ids, log_probs, stop_reason, engine_meta)`` — plain
            Python ``int`` and ``float``, never a framework's array scalars,
            so recording and response serialization cannot fail after commit.
            ``stop_reason`` is in the engine's own vocabulary (normalized by
            the caller).  ``engine_meta`` carries whatever the engine reported
            about the turn itself — see :func:`_engine_meta`; ``{}`` when the
            backend reports nothing.  It is a dict rather than more positional
            fields so adding one does not touch every transport again.
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
    ) -> tuple[list[int], list[int], list[float], str, str, dict[str, Any]]:
        """Tokenize, call the rollout engine, return raw results.

        ``model`` is the agent's OpenAI ``model`` field: it selects the
        tokenizer via the model registry and is pinned to the session on the
        first request (see :meth:`_resolve_for_session`); later turns may
        omit it.  The prompt is built strictly token-in-token-out per
        session: after the first turn, previously sampled completion tokens
        are reused verbatim and only the new inter-turn messages are
        tokenized (see :class:`~proxyserver.token_stream.TokenStreamManager`).

        Sampling params the request leaves at ``None`` fall back to the
        model's bundled ``generation_config.json`` (its publisher-recommended
        decoding settings — see :func:`~proxyserver.model_registry.load_sampling_defaults`),
        then to the engine's own defaults.  Rollout quality depends on this:
        raw full-distribution sampling (``top_p=1`` with unbounded ``top_k``)
        is what tips long agentic completions into repetition loops.  The
        override layers then win over whatever the request *did* set (see
        :meth:`get_sampling_overrides`), so the full precedence is::

            runtime overrides > config sampling_overrides > request
            > generation_config.json > engine

        Returns:
            ``(prompt_ids, token_ids, log_probs, finish_reason,
            completion_text, engine_meta)``, where ``finish_reason`` is
            ``"stop"`` or ``"length"`` — already normalized across engines —
            and ``engine_meta`` is what the engine reported about the turn
            (see :func:`_engine_meta`), ``{}`` when it reported nothing.

        A request whose normalized messages match the session's previous
        normalized messages is treated as a lost-response retry and re-served
        from the cached turn: same tokens, same logprobs, no second generation.
        Retries are idempotent while the cache and replay allowance remain. Top-level
        tools and sampling parameters are excluded from the match, so changes
        to those fields alone still replay the cached turn. The model pin is
        checked first. The mirror holds for failure: a request the engine already
        **rejected** with a deterministic 4xx :data:`MAX_REJECTED_GENERATIONS`
        times is failed fast with the stored rejection instead of burning another 
        generation to hear the same verdict (see :meth:`_rejected_repeat`).

        Turns of one session are serialized here on a per-session lock held
        from the retry lookup through the commit: a retry racing its original
        — e.g. one whose caller disconnected while the generation is still
        running here — parks until the original commits and is then re-served
        from the cache instead of triggering a second generation of the same turn.
        """
        # Resolve and tokenize off the event loop: the first resolution of a
        # tokenizer directory loads the tokenizer from disk, and
        # apply_chat_template is a CPU-bound template render of the history
        # (an injected tokenizer_loader may even make it an HTTP round-trip)
        # — it must not stall other in-flight proxy requests (same reason
        # decode below is threaded).
        first_turn = session_id is not None and session_id not in self._session_models
        resolved = await _run_state_update(self._resolve_for_session, session_id, model)

        # Serialize generation through commit, including for callers using
        # the provider directly. A retry waits for the original's commit and
        # is then re-served by cached_turn. The HTTP completion pipeline also
        # serializes parsing and recording after this lock is released.
        async with self._session_lock(session_id):
            # Matching normalized messages are treated as a lost-response retry:
            # the turn is already committed to the stream, so re-serve it rather
            # than reject (which would strand the rollout) or re-sample (which
            # would fork the stream).  Threaded because the comparison
            # normalizes the full message list.
            cached = await _run_state_update(resolved.token_streams.cached_turn, session_id, messages)
            if cached is not None:
                prompt_ids, token_ids, log_probs, finish_reason, engine_meta = cached
                completion_text = await asyncio.to_thread(resolved.tokenizer.decode, token_ids, skip_special_tokens=False)
                logger.warning(
                    "Session %s: normalized messages match the previous turn; "
                    "re-serving the cached completion (lost-response retry)",
                    session_id,
                )
                # engine_meta comes from the cache too, so a retry arriving
                # after a weight update still reports the version the tokens
                # were actually sampled under — reporting the current one would
                # make an off-policy turn look on-policy.
                return prompt_ids, token_ids, log_probs, finish_reason, completion_text, engine_meta

            # The mirror case: an exact repeat of a request the engine
            # *rejected* (deterministic 4xx) is failed fast with the stored
            # rejection rather than re-generated — otherwise a rejection that
            # outlasts the caller's timeout is retried forever and the stream
            # wedges (see _rejected_repeat).
            rejected = self._rejected_repeat(session_id, messages)
            if rejected is not None:
                raise rejected

            prompt_ids = await _run_state_update(resolved.token_streams.build_prompt, session_id, messages, tools)

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
            # The override layers win over the request: the training side's
            # sampling policy is not the agent's to relax (config layer from
            # the YAML, runtime layer pushed by the trainer per step).
            overrides = {**self._config_sampling_overrides, **self._runtime_sampling_overrides}
            sampling_params.update(overrides)
            sampling_params["max_tokens"] = max_tokens
            sampling_params["logprobs"] = True
            if stop:
                sampling_params["stop"] = stop
            if seed is not None:
                if type(seed) is not int:
                    raise TokenStreamError("seed must be an integer")
                sampling_params["seed"] = seed
            if stop_token_ids is not None:
                if not isinstance(stop_token_ids, list) or any(type(token) is not int or token < 0 for token in stop_token_ids):
                    raise TokenStreamError("stop_token_ids must be a list of nonnegative integers")
                sampling_params["stop_token_ids"] = stop_token_ids
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

            try:
                # The prior rides along only when there is one, so transport
                # hooks that predate it (tests, third-party providers) keep
                # their three-argument signature.
                token_ids, log_probs, stop_reason, engine_meta = await self._call_engine(
                    prompt_ids, sampling_params, session_id,
                    **({"routed_experts_prior": routed_experts_prior} if routed_experts_prior else {}),
                )
            except Exception as e:
                self._note_engine_rejection(session_id, messages, e)
                raise
            self._rejected_turns.pop(session_id, None)

            if not isinstance(token_ids, list) or not isinstance(log_probs, list):
                raise EngineError(f"Session {session_id}: the rollout engine must return token/logprob lists")
            finish_reason = self.engine.finish_reason(
                stop_reason,
                num_tokens=len(token_ids),
                max_tokens=sampling_params["max_tokens"],
            )

            # Validate before decoding and committing: an invalid
            # completion must never enter the session's authoritative token
            # stream, or every later turn of the trial silently trains on it.
            if finish_reason == ABORT:
                raise EngineAbort(f"Session {session_id}: the rollout engine aborted the request")
            if not token_ids:
                # _sglang_generate returns empty token_ids when the engine's
                # logprob payload does not line up with its output IDs.
                raise EngineError(f"Session {session_id}: the rollout engine returned no tokens")
            if len(log_probs) != len(token_ids):
                # The trainer pairs logprobs with tokens positionally, so a
                # misaligned payload recorded as-is is silent training-data
                # corruption.  logprobs are always requested (sampling_params
                # above), so every transport must return one per token.
                raise EngineError(
                    f"Session {session_id}: the rollout engine returned "
                    f"{len(log_probs)} logprobs for {len(token_ids)} tokens; "
                    f"refusing to record a misaligned turn"
                )
            if any(type(token) is not int or token < 0 for token in token_ids):
                raise EngineError(f"Session {session_id}: the rollout engine returned invalid token IDs")
            log_probs = [_finite_logprob(value, session_id) for value in log_probs]

            # Decode off the event loop: it is CPU-bound on long completions and
            # must not stall other in-flight proxy requests.
            completion_text = await asyncio.to_thread(resolved.tokenizer.decode, token_ids, skip_special_tokens=False)

            if session_id not in self._session_models:
                # A direct provider caller can release a session while its
                # turn is generating. Return the result without re-creating
                # the released token stream, which would otherwise occupy
                # an LRU slot until eviction. HTTP DELETE instead cancels
                # and drains active work before releasing its streams.
                # No await sits between this check and the return, so a
                # release cannot interleave after it.
                logger.info(
                    "Session %s was released while this turn was generating; "
                    "returning the turn without re-creating its token stream",
                    session_id,
                )
            else:
                # logprobs/finish_reason/engine_meta ride along so an exact
                # retry of this request (a lost-response recovery) can be
                # re-served — see cached_turn above.
                resolved.token_streams.commit(
                    session_id, messages, prompt_ids, token_ids,
                    log_probs=log_probs, finish_reason=finish_reason,
                    engine_meta=engine_meta,
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

        decoded = await tool_parser.extract_completion(token_ids, tools=tools, completion_text=completion_text)
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

    Shared by :class:`SGLangRolloutProvider` (through slime's sgl-router,
    which is transparent) and :class:`DirectRolloutProvider` (straight to an
    SGLang server) — the wire protocol is identical.  Translates the
    vLLM-style ``max_tokens``/``logprobs`` into ``max_new_tokens``/``return_logprob``
    and reads the sampled ids + ``meta_info.finish_reason.type``
    (``"stop"``/``"length"``/``"abort"`` — the sglang adapter's vocabulary)
    from the response.  A logprob payload that does not line up with the
    sampled ids is refused (empty result, which ``generate()`` turns into
    an ``EngineError``) rather than recorded.

    ``return_routed_experts`` additionally asks the engine for the per-token
    MoE expert selections (the same request key slime's own rollout sends for
    ``--use-rollout-routing-replay``); the repacked payload rides the returned
    meta as ``routed_experts`` — see :func:`_pack_routed_experts`.
    ``routed_experts_prior`` is ``(rows, cols)`` of the blob the recorder
    already holds for this stream: when set, the request carries
    ``routed_experts_start_len=rows`` so the engine returns only the rows
    added since — a turn's delta instead of the whole prefix again.

    ``session_id`` is the composed stream id (one per agent of a rollout); it
    rides along as the router's routing key so every turn of a stream lands on
    the replica that already holds its prefix.
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
                raise EngineError(
                    f"Session {session_id}: SGLang returned an invalid logprob entry at position {index}"
                )
            logprob_token_ids.append(entry[1])
            if return_logprob:
                log_probs.append(_finite_logprob(entry[0], session_id))
    if not token_ids and output_token_logprobs:
        # `output_ids` comes back only from an engine started tokens-only
        # (``--skip-tokenizer-init``).  With the tokenizer initialized SGLang
        # returns decoded ``text`` instead, and the sampled ids survive only
        # inside the logprob triples ``(logprob, token_id, text)`` — which is
        # exactly where slime's own rollout reads them
        # (``slime/rollout/sglang_rollout.py``).  Taking them from the same
        # place makes the proxy correct against an engine launched either way,
        # which matters because under the slime transport the *training
        # framework* owns the launch and does not pass that flag.
        token_ids = logprob_token_ids

    if return_logprob and logprob_token_ids != token_ids:
        # Equal lengths alone cannot prove alignment: every logprob triple
        # identifies its token, and must agree with output_ids positionally.
        # Return an empty result so generate() refuses the turn before commit.
        logger.error(
            "Session %s: SGLang logprob/token ID mismatch (%d vs %d tokens)",
            session_id, len(logprob_token_ids), len(token_ids),
        )
        return [], [], stop_reason, meta

    if return_routed_experts:
        # Repacked off the event loop too: the decode + uint8 cast walks the
        # same >100 MB payload the parse above did.
        meta["routed_experts"] = await asyncio.to_thread(
            _pack_routed_experts, meta_info, len(prompt_ids), len(token_ids), session_id,
            routed_experts_prior,
        )

    return token_ids, log_probs, stop_reason, meta


def _routing_headers(session_id: str | None) -> dict[str, str] | None:
    """Session-affinity header for slime's ``sgl-router`` / SGLang Model Gateway.

    Under ``--router-policy consistent_hashing`` the router pins one routing
    key to one worker, so an agent's turns all reuse the replica that already
    holds its prefix.  Without it a 100-turn agentic stream can bounce across
    replicas and re-prefill its whole context every turn — the dominant cost in
    long multi-turn rollouts.  Routers on other policies ignore the header, so
    it is always safe to send.

    Keyed on the *stream* (one agent of one rollout), not the rollout: agents
    of one trial hold independent token streams with no shared prefix, so
    pinning them together would only unbalance the pool.
    """
    return {"X-SMG-Routing-Key": session_id} if session_id else None


def _engine_meta(meta_info: dict[str, Any]) -> dict[str, Any]:
    """Engine-reported facts about a turn that outlive the response.

    Currently just the policy weight version, which an RL trainer needs to
    prove a rollout was on-policy: a session whose turns span two versions
    straddled a weight update, and the samples it yields are silently mixed.
    Returned as a dict so a later field (cached-token counts, speculative
    decoding stats) does not change every transport's signature again.
    """
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
    """SGLang's ``meta_info["routed_experts"]``, repacked for the record.

    The engine (``enable_return_routed_experts``) reports the per-token MoE
    expert selections as base64 of a little-endian int32 C-order array of
    logical shape ``(rows, num_layers * topk)``.  Without a start offset the
    rows cover the **whole** stream up to and including this turn, prompt
    included (``stream_so_far - 1``: a token's routing exists once it has
    been fed as input).  With ``prior = (rows, cols)`` of the blob the
    recorder already holds, the request asked for ``routed_experts_start_len=rows``
    and the engine returns only ``[rows, stream - 1)`` — the turn's delta,
    which the recorder appends (see ``SessionRecorder.record_completion``);
    ``start`` in the returned blob says which one this is.  An engine that
    predates the request key, or a router that strips it, still answers with
    the full range: that case is recognised from ``prior``'s column count and
    sliced down here, so capture keeps working and only the bandwidth saving
    is lost.  Expert ids are tiny (< num_experts, 256 on current models), so
    int32 is 4x air: the record keeps uint8.  ``cols`` stays a single number
    because the proxy holds no model config — the trainer-side consumer, which
    knows ``num_layers`` × ``topk``, validates and reshapes.

    A missing or misaligned payload is refused (``EngineError``) rather than
    recorded, like the logprob guard in :func:`_sglang_generate`: a silently
    short or corrupt tensor would crash the trainer's replay pass hours
    later (slime asserts ``rows == tokens - 1`` deep inside Megatron), and
    a missing one means the engines were launched without the capture flag
    — fail the first turn, loudly, instead.
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
        # The engine ignored routed_experts_start_len and sent the whole
        # stream: keep the delta the recorder expects.
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

    A bare SGLang server reports its ServerArgs flattened at the top level; a
    router may hand back one worker's payload verbatim or wrap the per-worker
    payloads in a list (every worker serves the same model, so the first entry
    speaks for the pool).  Anything unrecognized is ``None`` — no clamp.

    ``context_length`` is ServerArgs' CLI override and stays null unless the
    operator launched with ``--context-length`` — i.e. on a default launch it
    says nothing.  The scheduler's actual per-request cap rides along in the
    same payload as ``max_req_input_len`` (min of the model's window and KV
    capacity, less a small buffer), so it serves as the fallback: a few tokens
    conservative as a total-window bound, which a clamp can afford.
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

    Both engines hard-reject a request that does not fit, up-front: vLLM 400s
    when ``prompt + max_tokens > max_model_len``, and SGLang admission-checks
    ``prompt + max_new_tokens <= context_len``.  Near the end of a long
    rollout that turns every remaining turn into an unrecoverable error storm
    even while room to generate remains; a clamped turn instead generates
    what fits. Explicit engine finish reasons remain authoritative, including
    an EOS on the last allowed token. The in-place write also updates the
    token-budget fallback used when the engine omits a recognized reason.

    ``context_length=None`` means the window is unknown (the engine would
    not say): no clamp, the engine polices its own limits.
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


class SGLangRolloutProvider(BaseRolloutProvider):
    """Native SGLang HTTP ``/generate``, used by the slime and verl modes.

    Sends token-ID prompts to a SGLang server, including one hosted by VeRL,
    or through a transparent HTTP router such as Slime's ``sgl-router``.
    When a router is used, it owns placement and balancing.

    Because the router is transparent, the provider speaks SGLang's native
    dialect itself: ``max_tokens``→``max_new_tokens`` and ``logprobs``→``return_logprob``,
    and reads ``output_ids`` + ``meta_info.finish_reason.type`` (``"stop"``/``"length"``/``"abort"`` —
    the sglang adapter's vocabulary) from the response.  A logprob payload
    that does not line up with the output ids is refused (empty result →
    ``EngineError``) rather than recorded.

    The same provider also works pointed straight at a single SGLang server —
    the router adds balancing, not protocol.
    """

    def __init__(
        self,
        router_url: str,
        tool_parser_factory: Callable[[str, Any], Any] | None = None,
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
            router_url: Base URL of slime's ``sgl-router`` (or of a bare
                SGLang server), e.g. ``http://10.0.1.5:30000``.
            tool_parser_factory: Optional ``(tool_call_parser, tokenizer) -> tool_parser``.
            inference_engine: Must resolve to ``"sglang"`` — slime's router
                fronts SGLang servers only.  Defaults to ``"sglang"``
                directly (``$INFERENCE_ENGINE`` is *not* consulted, so a
                vLLM-flavored environment cannot silently misconfigure it).
            api_key: Optional bearer token sent to the router.
            request_timeout: Per-request HTTP timeout (seconds).
            tokenizer_loader: Optional tokenizer-loading override (see
                :class:`BaseRolloutProvider`).
            context_length: The engines' context window in tokens, stated by
                the operator instead of discovered — what the engines were
                launched with (SGLang's ``--context-length``, i.e. slime's
                ``--sglang-context-length``).  Needed because slime's
                ``sgl-router`` answers ``GET /get_server_info`` from its own
                ``RouterManager`` stub (``{"router_manager": true,
                "routers_count": ..., "workers_count": ...}``) instead of
                forwarding a worker's payload, so against that router
                discovery can never find a window and the clamp in
                :meth:`_call_engine` would stay off for the whole run.
                ``None`` keeps discovery, which is correct against a bare
                SGLang server (it reports its own ``ServerArgs``).
            return_routed_experts: Ask the engines for the per-token MoE
                expert selections on every ``/generate`` and record the
                repacked payload per agent stream — what slime's
                ``--use-rollout-routing-replay`` (R3) trains on.  Requires
                engines launched with ``enable_return_routed_experts``
                (slime sets that server arg from the same flag); with it
                off, the first captured turn fails loudly.  This is the
                config baseline; the trainer may layer a runtime toggle
                over it per step via ``PUT /routed_experts``
                (:meth:`set_runtime_routed_experts`).
        """
        engine = get_adapter(inference_engine or SGLANG)
        if engine.name != SGLANG:
            raise ValueError(
                f"SGLangRolloutProvider requires inference_engine='sglang' "
                f"for the native /generate protocol, got {engine.name!r}"
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
        # Routed-experts capture layers, shaped like the sampling overrides:
        # the config baseline survives a proxy restart, the runtime layer
        # (PUT /routed_experts) wins while set and None means "cleared".
        self._config_return_routed_experts = bool(return_routed_experts)
        self._runtime_return_routed_experts: bool | None = None
        super().__init__(tool_parser_factory, engine.name, tokenizer_loader, sampling_overrides=sampling_overrides)
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

    def set_runtime_routed_experts(self, enabled: bool | None) -> dict[str, Any]:
        """Set (or clear) the runtime routed-experts capture layer.

        The trainer's per-step push (``PUT /routed_experts``): capture on
        for train steps, off for eval steps — whose streams nothing trains
        on, so their payloads would be pure waste.  ``None`` clears the
        layer, falling back to the config baseline.  Returns
        :meth:`get_routed_experts_config` so the caller can verify the
        effective state it just installed.
        """
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError(f"enabled must be true, false or null, got {enabled!r}")
        self._runtime_return_routed_experts = enabled
        logger.info(
            "Runtime routed-experts capture set: %s",
            "cleared (config applies)" if enabled is None else enabled,
        )
        return self.get_routed_experts_config()

    def get_routed_experts_config(self) -> dict[str, Any]:
        """The capture layers: ``{"config": …, "runtime": …, "effective": …}``.

        ``effective`` — the runtime layer when set, else the config baseline
        — is what every ``/generate`` request obeys.
        """
        runtime = self._runtime_return_routed_experts
        return {
            "config": self._config_return_routed_experts,
            "runtime": runtime,
            "effective": self._config_return_routed_experts if runtime is None else runtime,
        }

    async def _call_engine(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        session_id: str | None,
        routed_experts_prior: tuple[int, int] | None = None,
    ) -> tuple[list[int], list[float], str | None, dict[str, Any]]:
        # Window learned lazily from the router (_engine_context_length); the
        # clamp itself is shared with the direct transport's paths.
        _clamp_to_context_window(
            sampling_params, len(prompt_ids),
            await self._engine_context_length(), self.router_url, session_id,
        )
        return await _sglang_generate(
            self._http, prompt_ids, sampling_params, session_id,
            return_routed_experts=self.get_routed_experts_config()["effective"],
            routed_experts_prior=routed_experts_prior,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()


def _vime_sampling_mask(choice: dict[str, Any], token_ids: list[int], session_id: str | None) -> dict[str, Any]:
    """Flatten vime's per-token nucleus support for loss replay.

    Each sampled token must belong to its reported support. Empty/missing
    rows cannot be replaced by the sampled token alone: that would change
    the distribution the trainer uses to recompute logprobs.
    """
    mask = choice.get("sampling_mask")
    if not isinstance(mask, list) or len(mask) != len(token_ids):
        raise EngineError(
            f"Session {session_id}: vime requires a sampling mask for every generated token "
            "(top_p < 1); enable return_sampling_mask at engine startup"
        )
    ids: list[int] = []
    offsets = [0]
    for token, row in zip(token_ids, mask):
        if (
            not isinstance(row, list) or not row
            or any(type(value) is not int or value < 0 for value in row)
            or token not in row or len(set(row)) != len(row)
        ):
            raise EngineError(f"Session {session_id}: vime returned invalid sampling-mask token IDs")
        ids.extend(row)
        offsets.append(len(ids))
    return {"rollout_top_p_token_ids": ids, "rollout_top_p_token_offsets": offsets}


def _pack_vime_routed_experts(
    choice: dict[str, Any],
    prompt_len: int,
    completion_len: int,
    session_id: str | None,
    prior: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Repack vime's base64 .npy capture into the same blob as slime.

    vLLM returns (tokens, layers, topk), with token coverage ending at
    prompt + completion - 1. A continuation may return only the rows after
    ``routed_experts_prompt_start``; older engines may return the full
    capture, which is sliced locally. Never load pickled numpy objects or
    narrow expert IDs that do not fit the recorder's uint8 format.
    """
    encoded = choice.get("routed_experts")
    if not isinstance(encoded, str) or not encoded:
        raise EngineError(
            f"Session {session_id}: return_routed_experts was requested but vime "
            "reported no routed_experts — enable --use-rollout-routing-replay on the engines"
        )
    try:
        values = np.load(io.BytesIO(base64.b64decode(encoded, validate=True)), allow_pickle=False)
    except Exception as e:
        raise EngineError(f"Session {session_id}: invalid vime routed_experts .npy payload") from e
    if not isinstance(values, np.ndarray):
        values.close()  # np.load also accepts .npz archives; they are not this protocol.
        raise EngineError(f"Session {session_id}: vime routed_experts must be a numpy array")
    if values.ndim not in (2, 3) or values.dtype.kind not in "iu" or not values.size:
        raise EngineError(f"Session {session_id}: vime routed_experts must be a nonempty integer token/layer array")
    values = values.reshape(values.shape[0], -1)
    full_rows = prompt_len + completion_len - 1
    start, cols_hint = prior or (0, None)
    rows = full_rows - start
    if start and values.shape[0] == full_rows:
        values = values[start:]
    if rows <= 0 or values.shape[0] != rows or (cols_hint and values.shape[1] != cols_hint):
        raise EngineError(
            f"Session {session_id}: vime routed_experts shape {values.shape} does not "
            f"line up with {rows} rows (start {start}, expected cols {cols_hint})"
        )
    if values.min() < 0 or values.max() > 255:
        raise EngineError(f"Session {session_id}: vime routed_experts ids outside [0, 256) do not fit uint8")
    return {
        "data": base64.b64encode(values.astype(np.uint8).tobytes()).decode("ascii"),
        "rows": rows,
        "cols": values.shape[1],
        "dtype": "uint8",
        "start": start,
    }


async def _vime_generate(
    http: Any,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    session_id: str | None,
    *,
    return_routed_experts: bool = False,
    routed_experts_prior: tuple[int, int] | None = None,
) -> tuple[list[int], list[float], str | None, dict[str, Any]]:
    """Vime/upstream vLLM token-ID protocol, parallel to :func:`_sglang_generate`.

    The router URL selects the policy model, as in vime's agent adapter;
    no text rendering or model discovery is performed upstream. Cancellation
    closes this HTTP request, never broadcasts an abort to shared workers.
    """
    params = {k: v for k, v in sampling_params.items() if v is not None}
    params["logprobs"] = 1
    if params.get("stop"):
        params["include_stop_str_in_output"] = True
    capture_mask = params.get("top_p", 1.0) < 1.0
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
        raise EngineError(f"Session {session_id}: vime must return exactly one generation choice")
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
        raise EngineError(f"Session {session_id}: unknown vime finish_reason {stop_reason!r}")
    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list) or any(type(token) is not int or token < 0 for token in token_ids):
        raise EngineError(f"Session {session_id}: vime returned invalid token_ids")
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list) or len(content) != len(token_ids):
        raise EngineError(f"Session {session_id}: vime logprobs do not line up with token_ids")
    log_probs: list[float] = []
    for token, entry in zip(token_ids, content):
        if not isinstance(entry, dict) or type(entry.get("logprob")) not in (int, float):
            raise EngineError(f"Session {session_id}: vime returned an invalid token/logprob pair")
        # Vime deployments return an integer token_id; upstream vLLM's
        # token endpoint encodes it in token="token_id:<id>". Validate
        # whichever representation is supplied, never re-tokenize text.
        if "token_id" in entry:
            aligned = type(entry["token_id"]) is int and entry["token_id"] == token
        else:
            aligned = entry.get("token") == f"token_id:{token}"
        if not aligned:
            raise EngineError(f"Session {session_id}: vime returned an invalid token/logprob pair")
        log_probs.append(_finite_logprob(entry["logprob"], session_id))
    usage = output.get("usage")
    if usage is not None:
        if not isinstance(usage, dict):
            raise EngineError(f"Session {session_id}: vime returned invalid usage")
        count = usage.get("completion_tokens")
        if count is not None and (type(count) is not int or count != len(token_ids)):
            raise EngineError(f"Session {session_id}: vime completion_tokens does not match token_ids")
    if capture_mask:
        meta.update(await asyncio.to_thread(_vime_sampling_mask, choice, token_ids, session_id))
    if return_routed_experts:
        meta["routed_experts"] = await asyncio.to_thread(
            _pack_vime_routed_experts, choice, len(prompt_ids), len(token_ids), session_id,
            routed_experts_prior,
        )
    return token_ids, log_probs, stop_reason, meta


class VLLMRolloutProvider(BaseRolloutProvider):
    """Native vLLM HTTP ``/inference/v1/generate``, used by vime and verl.

    The deployment owns the workers and load balancing; the proxy speaks
    their native HTTP dialect directly. ``x-session-id`` carries the agent
    stream for routers using ``consistent_hash``. Configure context_length
    to match ``--vllm-max-model-len``; router model discovery is not required.

    Top-p support is captured automatically when top_p < 1; enable return_sampling_mask
    at engine startup. Routing replay is opt-in and requires compatible engine output.
    Weight updates must drain rollouts first: a record holds one policy version per turn,
    so PipelineRL's mid-generation updates are not supported.
    """

    def __init__(
        self,
        router_url: str,
        tool_parser_factory: Callable[[str, Any], Any] | None = None,
        inference_engine: str | None = None,
        api_key: str | None = None,
        request_timeout: float = 900.0,
        tokenizer_loader: Callable[[str], Any] | None = None,
        sampling_overrides: dict[str, Any] | None = None,
        context_length: int | None = None,
        return_routed_experts: bool = False,
    ) -> None:
        engine = get_adapter(inference_engine or VLLM)
        if engine.name != VLLM:
            raise ValueError(
                f"VLLMRolloutProvider requires inference_engine='vllm' "
                f"for the native /inference/v1/generate protocol, got {engine.name!r}"
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
        super().__init__(tool_parser_factory, engine.name, tokenizer_loader, sampling_overrides=sampling_overrides)
        logger.info("vLLM rollout provider routing via %s", self.router_url)
        if self._config_return_routed_experts:
            logger.info("%s: routed-experts capture on by config (R3)", self.router_url)
        if context_length is not None:
            logger.info("%s: engine context window pinned to %d tokens by config", self.router_url, context_length)
        else:
            logger.warning("%s: set context_length to the engine's context window in tokens; context clamping is disabled", self.router_url)

    def set_runtime_routed_experts(self, enabled: bool | None) -> dict[str, Any]:
        """Set the trainer's capture override; None restores the config baseline."""
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError("enabled must be true, false or null")
        self._runtime_return_routed_experts = enabled
        logger.info("Runtime routed-experts capture set: %s", "cleared (config applies)" if enabled is None else enabled)
        return self.get_routed_experts_config()

    def get_routed_experts_config(self) -> dict[str, Any]:
        """The capture layers, matching the slime transport's control API."""
        runtime = self._runtime_return_routed_experts
        return {
            "config": self._config_return_routed_experts,
            "runtime": runtime,
            "effective": self._config_return_routed_experts if runtime is None else runtime,
        }

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
        return await _vime_generate(
            self._http, prompt_ids, sampling_params, session_id,
            return_routed_experts=self.get_routed_experts_config()["effective"],
            routed_experts_prior=routed_experts_prior,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()


def normalize_engine_endpoints(
    base_urls: list[str] | str,
    api_keys: list[str] | str | None,
) -> tuple[list[str], list[str | None]]:
    """Normalize and validate the direct transport's endpoint wiring.

    Returns ``(roots, keys)``: server roots (whitespace and a trailing
    ``/v1`` stripped) and exactly one api_key per root — a single key is
    broadcast to every URL, ``None`` means unauthenticated.

    This is the single source of truth for the endpoint-list shape:
    :class:`DirectRolloutProvider` applies it at construction, so a
    misconfigured endpoint list fails when the provider is created,
    not at the first inference request minutes into a rollout.
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

    No training framework sits in the path: the proxy POSTs token-ID prompts
    straight to the engines' own APIs — vLLM's ``/v1/completions`` with
    ``prompt`` as token IDs and ``return_tokens_as_token_ids`` (the sampled
    IDs come back as ``"token_id:<id>"`` strings, EOS included), or SGLang's
    native ``/generate`` (shared with the slime transport, whose router is
    transparent).

    Endpoints come from ``inference_engine_base_url`` / ``inference_engine_api_key``
    in ``configs/{engine}-direct.yaml`` — lists, one entry per running engine
    instance; a single api_key is broadcast to every base URL.  A session is
    bound to one endpoint round-robin on its first request and stays **sticky**
    for its lifetime: multi-turn
    prompts of one trial reuse the same engine replica and its prefix cache.

    ``stop_reason`` is reported in each engine's **own** vocabulary, which
    the adapters in :mod:`proxyserver.engines` already accept: SGLang's
    ``meta_info.finish_reason.type`` is authoritative, and vLLM's raw
    ``finish_reason`` (``"stop"``/``"length"``/``"abort"``) passes through
    verbatim, including a natural stop on the final permitted token.

    vLLM's ``/v1/completions`` requires a model name; it is discovered once
    per endpoint from ``GET /v1/models`` (the engine serves exactly one
    model) rather than configured.  Each endpoint's context window is
    discovered the same lazy way — vLLM's ``max_model_len`` rides along with
    ``/v1/models``, SGLang's ``context_length`` comes from ``GET /get_server_info``
    — and caps each turn's ``max_tokens`` to the room the window has left
    (:func:`_clamp_to_context_window`).

    Direct SGLang discovers its served name from ``GET /get_model_info``
    (falling back to ``model_path`` on older servers). Both engines require
    that identity to map to the request's tokenizer profile before sending
    token IDs. Discovery is cached per endpoint; failed discovery is retried.
    """

    def __init__(
        self,
        base_urls: list[str] | str,
        tool_parser_factory: Callable[[str, Any], Any] | None = None,
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
            tool_parser_factory: Optional ``(tool_call_parser, tokenizer) -> tool_parser``.
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

        super().__init__(tool_parser_factory, inference_engine, tokenizer_loader, sampling_overrides=sampling_overrides)

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
        # Sticky session -> endpoint bindings (prefix cache), assigned
        # round-robin on a session's first request.
        self._session_endpoints: dict[str, int] = {}
        self._rr_index = 0
        # Per-endpoint served-model id and context window, discovered lazily
        # from the engine itself: vLLM's ride along with /v1/models, SGLang's
        # model identity comes from /get_model_info and its context window
        # comes from /get_server_info (see
        # _SGLangContextWindow for the fetch-failure semantics).
        self._model_ids: list[str | None] = [None] * len(roots)
        self._model_lens: list[int | None] = [None] * len(roots)
        self._model_lock = asyncio.Lock()
        self._context_windows = [_SGLangContextWindow(self._model_lock) for _ in roots]
        logger.info(
            "Direct rollout provider routing to %d %s endpoint(s): %s",
            len(roots), self.engine.name, ", ".join(roots),
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
        if self.engine.name == SGLANG:
            served_model = await self._sglang_model_id(index)
            self._validate_served_model(index, served_model, session_id)
            # Window learned lazily from the endpoint (_sglang_context_length);
            # Clamping changes the token budget, not the native finish reason:
            # an explicit stop remains a stop, including at the budget boundary.
            _clamp_to_context_window(
                sampling_params, len(prompt_ids),
                await self._sglang_context_length(index), self.base_urls[index], session_id,
            )
            return await _sglang_generate(http, prompt_ids, sampling_params, session_id)

        served_model = await self._vllm_model_id(index)
        self._validate_served_model(index, served_model, session_id)

        # Window learned lazily alongside the model id (_vllm_model_id); the
        # clamp itself is shared with the SGLang paths.
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
        # Sampling knobs ride through as generate() resolved them:
        # runtime overrides > config overrides > request > bundled
        # generation_config > engine defaults. An absent knob stays absent.
        # top_k/min_p/repetition_penalty are
        # vLLM extensions its OpenAI-compatible /v1/completions accepts.
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
        # Validate each raw pair. Independently filtering malformed tokens and
        # missing logprobs can shift positions while leaving equal lengths.
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
        # vLLM's raw finish_reason — generate() validates token/logprob
        # alignment and non-emptiness before anything is recorded.  vLLM's
        # OpenAI-compatible response reports nothing about the weights it is
        # serving, so there is no engine metadata to carry here.
        return token_ids, log_probs, choice.get("finish_reason"), {}

    async def aclose(self) -> None:
        """Close the underlying HTTP clients."""
        for client in self._clients:
            await client.aclose()
