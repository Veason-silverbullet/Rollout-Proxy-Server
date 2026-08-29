"""Rollout providers: route proxy requests to the rollout engines, strictly
token-in-token-out.

A provider owns the inference-side half of a completion: strict-TITO prompt
construction (:class:`~proxyserver.token_stream.TokenStreamManager`), the
transport to the rollout servers, engine-specific ``stop_reason``
normalization (:mod:`proxyserver.engines`), and tool-call parsing.  OpenAI
response formatting is *not* its job — the proxy server does that with
:func:`proxyserver.server.build_openai_response`, identically for both
operating modes.

One transport subclass exists per supported transport mode
(``transport_mode`` in ``configs/{engine}-{transport}.yaml``):

* :class:`RayRolloutProvider` (``transport_mode: verl``) — Ray RPC through
  verl's request load balancer (``acquire_server`` / ``release_server``,
  servers exposing ``generate(prompt_ids, sampling_params, request_id) ->
  TokenOutput``), as verl provides.
* :class:`SlimeRolloutProvider` (``transport_mode: slime``) — HTTP through
  slime's ``sgl-router``, which load-balances a pool of SGLang servers.  The
  router is a transparent HTTP proxy, so this provider speaks SGLang's native
  ``/generate`` dialect directly (``input_ids`` in, ``output_ids`` +
  ``meta_info`` out) and performs the vLLM-style→SGLang sampling-param
  translation itself — the translation verl does inside its rollout server.
* :class:`DirectRolloutProvider` (``transport_mode: direct``) — plain HTTP
  straight to the vLLM / SGLang engines themselves, no training framework in
  the path.  Token-ID prompts go to vLLM's ``/v1/completions``
  (``return_tokens_as_token_ids``) or SGLang's ``/generate``; endpoints come
  from ``inference_engine_base_url`` / ``inference_engine_api_key`` and each
  session is sticky to one endpoint (prefix cache).

The backend is selected with ``inference_engine`` (``"vllm"`` or
``"sglang"``); see :mod:`proxyserver.engines` for what it changes.  slime is
SGLang-only; verl and direct support both engines.

The tokenizer is not a constructor argument: every ``generate`` call names a
``model`` (the agent's OpenAI ``model`` field, validated by the proxy), the
provider resolves its tokenizer and tool parser through a
:class:`~proxyserver.model_registry.ModelRegistry`, and the session is
**pinned** to that model on its first request — a later request naming a
different model is rejected like any other strict-TITO violation, because
switching models mid-session would mean switching tokenizers mid-stream.

Usage::

    from proxyserver.rollout_provider import DirectRolloutProvider, RayRolloutProvider, SlimeRolloutProvider

    provider = RayRolloutProvider(load_balancer, inference_engine="sglang")
    # or, for slime:
    provider = SlimeRolloutProvider("http://<sglang-router-ip>:<port>")
    # or, for direct:
    provider = DirectRolloutProvider(["http://<engine-ip>:<port>/v1"],
                                     api_keys=["<key>"], inference_engine="vllm")

    prompt_ids, token_ids, log_probs, finish_reason, text, engine_meta = await provider.generate(
        messages=[...], session_id="trial_042", model="Qwen3.5-35B-A3B",
        max_tokens=2048,
    )
    content, tool_calls, finish_reason = await provider.parse_tool_calls("trial_042", token_ids, text, finish_reason)
"""

from __future__ import annotations
import asyncio
import base64
import numpy as np
import logging
from typing import Any, Callable
from uuid import uuid4
from .engines import ABORT, DEFAULT_MAX_TOKENS, LENGTH, SGLANG, EngineAbort, EngineError, get_adapter
from .model_registry import ModelRegistry, ResolvedModel, validate_sampling_overrides
from .token_stream import TokenStreamError, normalize_messages

logger = logging.getLogger(__name__)


def _plain_scalars(values: Any, cast: Any) -> list:
    """A framework's token/logprob container as plain Python scalars.

    ``tolist()`` is the fast path every numpy array and torch tensor offers
    and it already yields built-in ``int``/``float`` — at C speed, several
    times faster than casting element by element, which matters because
    completions run to tens of thousands of tokens and this runs on the
    event loop.  Anything else is cast explicitly, since a plain list may
    still hold array *scalars* that ``json.dumps`` cannot encode.
    """
    tolist = getattr(values, "tolist", None)
    return tolist() if callable(tolist) else list(map(cast, values))


def engine_http_limits() -> Any:
    """Connection-pool limits for HTTP clients that talk to the engines
    (the slime/direct transports here; the e2e harness imports this too, so
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
    prompt construction, ``stop_reason`` normalization, abort/empty
    validation, decoding, committing the turn to the session's token stream,
    tool-call parsing) is shared, so every transport records identical
    training data.
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
                ``tool_call_parser`` name from ``tokenization/mapping.json``.
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
        framework-side per-session state (e.g. sticky-session bindings inside
        verl's ``GlobalRequestLoadBalancer``, or slime's router-side routing
        state) is the framework's to evict.  A transport that holds
        per-session state of its own (e.g. :class:`DirectRolloutProvider`'s
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
    # Deterministic engine rejections (fail-fast on exact retries)
    # ------------------------------------------------------------------

    def _rejected_repeat(self, session_id: str, messages: list[dict[str, Any]]) -> Exception | None:
        """The stored rejection to re-raise, iff ``messages`` exactly repeats
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
            "Session %s: the engine already rejected this exact request %d "
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
    ) -> tuple[list[int], list[float], str | None, dict[str, Any]]:
        """Run one generation on the framework's rollout servers.

        Receives the vLLM-style ``sampling_params`` built by :meth:`generate`
        (``max_tokens``, ``logprobs``, plus the sampling knobs resolved from
        the request and the model's bundled defaults — ``temperature``,
        ``top_p``, ``top_k``, ``min_p``, ``repetition_penalty``,
        ``presence_penalty``, ``frequency_penalty``, optional ``stop``); a
        transport whose backend speaks another dialect translates them
        itself.

        Returns:
            ``(token_ids, log_probs, stop_reason, engine_meta)`` — plain
            Python ``int`` and ``float``, never a framework's array scalars,
            since these travel on to ``json.dumps`` at the relay long after
            the turn is committed and could not be regenerated there.
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
        decoding settings — see
        :func:`~proxyserver.model_registry.load_sampling_defaults`), then to
        the engine's own defaults.  Rollout quality depends on this: raw
        full-distribution sampling (``top_p=1`` with unbounded ``top_k``) is
        what tips long agentic completions into repetition loops.  The
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

        A request that repeats the session's previous request **verbatim** is
        a lost-response retry (the turn was generated and committed but the
        reply never reached the agent); it is re-served from the cached turn
        — same tokens, same logprobs, no second generation — so retries are
        idempotent and the expensive completion is never wasted.  The mirror
        holds for failure: a request the engine already **rejected** with a
        deterministic 4xx :data:`MAX_REJECTED_GENERATIONS` times is failed
        fast with the stored rejection instead of burning another generation
        to hear the same verdict (see :meth:`_rejected_repeat`).

        Turns of one session are serialized here on a per-session lock held
        from the retry lookup through the commit: a retry racing its original
        — e.g. one whose first attempt timed out at the proxy while the
        generation is still running here — parks until the original commits
        and is then re-served from the cache instead of triggering a second
        generation of the same turn.
        """
        # Resolve and tokenize off the event loop: the first resolution of a
        # tokenizer directory loads the tokenizer from disk, and
        # apply_chat_template is a CPU-bound template render of the history
        # (an injected tokenizer_loader may even make it an HTTP round-trip)
        # — it must not stall other in-flight proxy requests (same reason
        # decode below is threaded).
        first_turn = session_id is not None and session_id not in self._session_models
        resolved = await asyncio.to_thread(self._resolve_for_session, session_id, model)

        # Serialize turns of one session.  The proxy's own per-session lock
        # covers only requests still alive at the proxy: a request that
        # timed out there (503) or whose caller was torn down releases it
        # while this generation is still running, and the agent SDK's
        # automatic retry would race it into a second generation of the
        # same turn — forking the token stream.  Parked here instead, the
        # retry proceeds once the original commits and is re-served by
        # cached_turn below.  (Concurrent sessions are unaffected; see the
        # token_stream concurrency note.)
        async with self._session_lock(session_id):
            # An exact repeat of the previous request is a lost-response retry:
            # the turn is already committed to the stream, so re-serve it rather
            # than reject (which would strand the rollout) or re-sample (which
            # would fork the stream).  Threaded because the comparison
            # normalizes the full message list.
            cached = await asyncio.to_thread(resolved.token_streams.cached_turn, session_id, messages)
            if cached is not None:
                prompt_ids, token_ids, log_probs, finish_reason, engine_meta = cached
                completion_text = await asyncio.to_thread(resolved.tokenizer.decode, token_ids, skip_special_tokens=False)
                logger.warning(
                    "Session %s: request repeats the previous turn verbatim; "
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

            prompt_ids = await asyncio.to_thread(resolved.token_streams.build_prompt, session_id, messages, tools)

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
            logger.log(
                logging.INFO if first_turn else logging.DEBUG,
                "Session %s: sampling params %s (request keys: %s; override keys: %s)",
                session_id,
                {k: v for k, v in sampling_params.items() if k not in ("logprobs", "stop")},
                sorted(k for k, v in requested.items() if v is not None) or "none",
                sorted(overrides) or "none",
            )

            try:
                token_ids, log_probs, stop_reason, engine_meta = await self._call_engine(
                    prompt_ids, sampling_params, session_id,
                )
            except Exception as e:
                self._note_engine_rejection(session_id, messages, e)
                raise
            self._rejected_turns.pop(session_id, None)

            finish_reason = self.engine.finish_reason(
                stop_reason,
                num_tokens=len(token_ids),
                max_tokens=sampling_params["max_tokens"],
            )

            # Validate before decoding and committing: an aborted or empty
            # completion must never enter the session's authoritative token
            # stream, or every later turn of the trial silently trains on it.
            if finish_reason == ABORT:
                raise EngineAbort(f"Session {session_id}: the rollout engine aborted the request")
            if not token_ids:
                # An SGLang rollout server (verl's, or SGLang behind slime's
                # router) yields empty token_ids when the engine's logprob payload
                # does not line up with its output ids; vLLM can do the same on an
                # aborted request.
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

            # Decode off the event loop: it is CPU-bound on long completions and
            # must not stall other in-flight proxy requests.
            completion_text = await asyncio.to_thread(resolved.tokenizer.decode, token_ids, skip_special_tokens=False)

            if session_id not in self._session_models:
                # The session was released (deleted) while this turn was
                # generating — its model pin is gone.  Committing would
                # re-create the token stream release_session just dropped,
                # and with the session deleted nothing ever drops it again:
                # it would squat an LRU slot until eviction, which at scale
                # evicts live sessions instead.  The turn still returns so
                # the caller's pipeline finishes (the recorder drops it by
                # tombstone).  No await sits between this check and the
                # return, so a release cannot interleave after it.
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

    async def parse_tool_calls(
        self,
        session_id: str | None,
        token_ids: list[int],
        completion_text: str,
        finish_reason: str,
        tools: list[dict] | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, str]:
        """Extract tool calls and fold them into the final ``finish_reason``.

        Uses the tool parser of the session's pinned model (parsers are
        tokenizer-bound), so call this after :meth:`generate` of the same
        session.  Pass the request's ``tools`` so the parser can type each
        argument by its declared schema instead of guessing from the text
        (see :meth:`~proxyserver.tokenization.tool_parser.Qwen3CoderToolParser._coerce`).

        A parsed tool call upgrades ``finish_reason`` to ``"tool_calls"``
        **unless the response was truncated**, where ``"length"`` stands:
        the turn really was cut off, and an agent that treated it as a
        clean tool-call turn would never learn the rest of the completion
        is missing.  Any *complete* call parsed before the cut is still
        returned — it is one the model finished making — while a cut-off
        trailing call never parses at all and stays in the content.

        The returned content is agent-clean whether tool parsing is enabled
        or not: when the model's mapping entry names no ``tool_call_parser``
        the registry hands out a
        :class:`~proxyserver.tokenization.tool_parser.NullToolParser`, so special tokens
        (e.g. ``<|im_end|>``) are dropped and a leading reasoning block is
        stripped either way.  ``completion_text`` — :meth:`generate`'s raw
        decode, kept for the session record — is returned verbatim only
        when the session has no pinned model (:meth:`generate` never served
        it) or a framework factory explicitly built no parser.

        Args:
            session_id: The session :meth:`generate` just served.
            finish_reason: The engine-normalized reason from :meth:`generate`
                (``"stop"`` or ``"length"``).

        Returns:
            ``(content_text, openai_tool_calls_or_none, finish_reason)``
        """
        pinned = self._session_models.get(session_id) if session_id is not None else None
        tool_parser = self.models.resolve(pinned).tool_parser if pinned else None
        if tool_parser is None:
            return completion_text, None, finish_reason

        content_text, parsed_calls = await tool_parser.extract_tool_calls(token_ids, tools=tools)
        if not parsed_calls:
            return content_text, None, finish_reason

        openai_tool_calls = [
            {
                "id": f"call_{uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                },
            }
            for tc in parsed_calls
        ]
        if finish_reason != LENGTH:
            finish_reason = "tool_calls"
        return content_text, openai_tool_calls, finish_reason


class RayRolloutProvider(BaseRolloutProvider):
    """verl transport: the framework's rollout servers via Ray, strictly TITO.

    The vLLM and SGLang rollout servers share one ``generate`` signature
    and ``TokenOutput`` result (see :mod:`proxyserver.engines`; verl's
    servers are the reference implementation), so the only engine-dependent
    step is normalizing ``stop_reason`` — delegated to :mod:`proxyserver.engines`.
    """

    def __init__(
        self,
        load_balancer: Any,
        tool_parser_factory: Callable[[str, Any], Any] | None = None,
        inference_engine: str | None = None,
        tokenizer_loader: Callable[[str], Any] | None = None,
        sampling_overrides: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            load_balancer: Ray actor handle of the framework's request load
                balancer (e.g. verl's ``GlobalRequestLoadBalancer``); it must
                expose ``acquire_server`` / ``release_server``.
            tool_parser_factory: Optional ``(tool_call_parser, tokenizer) -> tool_parser``
                (e.g. verl's ``ToolParser.get_tool_parser``).
            inference_engine: ``"vllm"`` or ``"sglang"``. Defaults to
                ``$INFERENCE_ENGINE``, then ``"vllm"``.
            tokenizer_loader: Optional tokenizer-loading override (see
                :class:`BaseRolloutProvider`).
        """
        # Ray actor handle of the framework's request load balancer. We acquire
        # a rollout server handle per request via ``acquire_server`` and release
        # it via ``release_server`` so the recipe side carries no routing
        # state and stays consistent with the framework's main rollout path.
        self._lb = load_balancer
        super().__init__(tool_parser_factory, inference_engine, tokenizer_loader, sampling_overrides=sampling_overrides)

    async def _call_engine(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        session_id: str | None,
    ) -> tuple[list[int], list[float], str | None, dict[str, Any]]:
        # Sticky on session_id so multi-turn requests in the same trial reuse
        # the same engine replica (preserves prefix cache). The per-call
        # ``request_id`` passed to ``server.generate`` is a fresh uuid so the
        # engine treats every turn as a new request.
        sticky_key = session_id or "default"

        server_id, server = await self._lb.acquire_server.remote(request_id=sticky_key)
        try:
            # Await the ObjectRef directly (as acquire_server above already
            # does): ray.get in asyncio.to_thread would park a
            # default-executor thread for the whole generation, capping
            # concurrent rollouts at the pool size (min(32, cpus + 4)) and
            # starving the tokenize/decode/record threads every other request shares.
            output = await server.generate.remote(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=uuid4().hex,
            )
        finally:
            # Fire-and-forget: mirrors the framework's own release path
            # (e.g. verl's LLMServerClient._release_server).
            self._lb.release_server.remote(server_id=server_id)

        # Coerce to plain ints/floats.  A framework's TokenOutput may carry
        # torch/numpy scalars, which the other transports already normalize
        # (they parse them out of JSON) but which would travel from here all
        # the way to the relay's json.dumps and fail there — after the turn
        # is committed, so it could not be regenerated either.
        # `is not None`, never truthiness: a numpy array or torch tensor of
        # more than one element raises ValueError ("truth value ... is
        # ambiguous") on `if arr`, which would abort the turn *after* the
        # engine generated it.
        token_ids = _plain_scalars(output.token_ids, int)
        log_probs = _plain_scalars(output.log_probs, float) if output.log_probs is not None else []
        # The framework's TokenOutput exposes engine metadata as attributes
        # rather than SGLang's meta_info dict; only weight_version is defined
        # across both rollout servers, and older ones omit it entirely.
        weight_version = getattr(output, "weight_version", None)
        meta = {"weight_version": str(weight_version)} if weight_version is not None else {}
        return token_ids, log_probs, getattr(output, "stop_reason", None), meta


async def _sglang_generate(
    http: Any,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    session_id: str | None,
    return_routed_experts: bool = False,
) -> tuple[list[int], list[float], str | None, dict[str, Any]]:
    """One generation in SGLang's native ``/generate`` dialect.

    Shared by :class:`SlimeRolloutProvider` (through slime's sgl-router,
    which is transparent) and :class:`DirectRolloutProvider` (straight to an
    SGLang server) — the wire protocol is identical.  Translates the
    vLLM-style ``max_tokens``/``logprobs`` into
    ``max_new_tokens``/``return_logprob`` (the translation verl performs
    inside its ``AsyncSGLangServer``) and reads the sampled ids +
    ``meta_info.finish_reason.type`` (``"stop"``/``"length"``/``"abort"`` —
    the sglang adapter's vocabulary) from the response.  A logprob payload
    that does not line up with the sampled ids is refused (empty result, which
    ``generate()`` turns into an ``EngineError``) rather than recorded.

    ``return_routed_experts`` additionally asks the engine for the per-token
    MoE expert selections (the same request key slime's own rollout sends for
    ``--use-rollout-routing-replay``); the repacked payload rides the returned
    meta as ``routed_experts`` — see :func:`_pack_routed_experts`.

    ``session_id`` is the composed stream id (one per agent of a rollout); it
    rides along as the router's routing key so every turn of a stream lands on
    the replica that already holds its prefix.
    """
    params = dict(sampling_params)
    params["max_new_tokens"] = params.pop("max_tokens", None)
    return_logprob = bool(params.pop("logprobs", False))
    params = {k: v for k, v in params.items() if v is not None}

    body: dict[str, Any] = {
        "input_ids": prompt_ids,
        "sampling_params": params,
        "return_logprob": return_logprob,
    }
    if return_routed_experts:
        body["return_routed_experts"] = True
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
    token_ids = [int(token_id) for token_id in output.get("output_ids") or []]
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
        token_ids = [int(entry[1]) for entry in output_token_logprobs]

    if return_routed_experts:
        # Repacked off the event loop too: the decode + uint8 cast walks the
        # same >100 MB payload the parse above did.
        meta["routed_experts"] = await asyncio.to_thread(
            _pack_routed_experts, meta_info, len(prompt_ids), len(token_ids), session_id,
        )

    if not return_logprob:
        return token_ids, [], stop_reason, meta

    if len(output_token_logprobs) != len(token_ids):
        # Refuse a misaligned logprob payload the same way the verl
        # SGLang server does: empty result, which generate() turns into
        # an EngineError before anything reaches the token stream.
        logger.error(
            "Session %s: SGLang logprob/token length mismatch (%d vs %d)",
            session_id, len(output_token_logprobs), len(token_ids),
        )
        return [], [], stop_reason, meta
    log_probs = [float(entry[0]) for entry in output_token_logprobs]
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
) -> dict[str, Any]:
    """SGLang's ``meta_info["routed_experts"]``, repacked for the record.

    The engine (``enable_return_routed_experts``) reports the per-token MoE
    expert selections as base64 of a little-endian int32 C-order array of
    logical shape ``(stream_so_far - 1, num_layers * topk)`` — the **whole**
    stream up to and including this turn, prompt included, so each turn's
    payload supersedes the previous one and the recorder keeps only the
    latest per agent.  Expert ids are tiny (< num_experts, 256 on current
    models), so int32 is 4x air: the record keeps uint8.  ``cols`` stays a
    single number because the proxy holds no model config — the trainer-side
    consumer, which knows ``num_layers`` × ``topk``, validates and reshapes.

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
    rows = prompt_len + completion_len - 1
    if rows <= 0 or values.size == 0 or values.size % rows != 0:
        raise EngineError(
            f"Session {session_id}: routed_experts payload does not line up with "
            f"the stream ({values.size} values for {rows} rows = prompt "
            f"{prompt_len} + completion {completion_len} - 1)"
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
    what fits and ends as an honest ``finish_reason='length'``, which agents
    already understand.  The in-place write is load-bearing: ``generate()``
    measures truncation against ``sampling_params["max_tokens"]``, so on
    vLLM (which cannot report truncation) a turn cut at the clamp still
    reads ``length`` rather than a natural ``stop``.

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


class SlimeRolloutProvider(BaseRolloutProvider):
    """slime transport: SGLang servers via slime's ``sgl-router``, strictly TITO.

    slime holds no Ray-side load-balancer actor — it fronts its pool of
    SGLang servers with ``sgl-router``, a transparent HTTP proxy (default
    policy: least-inflight), and its own rollout code simply POSTs to
    ``http://{sglang_router_ip}:{sglang_router_port}/generate``.  This
    provider does the same with the proxy's token-ID prompts, so the router
    keeps owning placement and balancing.

    Because the router is transparent, the provider speaks SGLang's native
    dialect itself: ``max_tokens``→``max_new_tokens`` and
    ``logprobs``→``return_logprob`` (the translation verl performs inside its
    ``AsyncSGLangServer``), and reads ``output_ids`` +
    ``meta_info.finish_reason.type`` (``"stop"``/``"length"``/``"abort"`` —
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
                f"SlimeRolloutProvider requires inference_engine='sglang' "
                f"(slime's sgl-router fronts SGLang servers), got {engine.name!r}"
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
        # The engines' context window.  A configured value is authoritative and
        # starts out already "known", so the router is never asked; otherwise it
        # is learned lazily from GET /get_server_info (see
        # _engine_context_length), where _known also covers the "asked, and the
        # answer names no context_length" case, which is cached like a value.
        self._context_length: int | None = context_length
        self._context_length_known = context_length is not None
        self._context_length_lock = asyncio.Lock()
        self._context_length_fetch_warned = False
        # Routed-experts capture layers, shaped like the sampling overrides:
        # the config baseline survives a proxy restart, the runtime layer
        # (PUT /routed_experts) wins while set and None means "cleared".
        self._config_return_routed_experts = bool(return_routed_experts)
        self._runtime_return_routed_experts: bool | None = None
        super().__init__(tool_parser_factory, engine.name, tokenizer_loader, sampling_overrides=sampling_overrides)
        logger.info("Slime rollout provider routing via %s", self.router_url)
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
        """The engines' context window, from ``GET /get_server_info`` (fetched
        once and cached) — the SGLang counterpart of the vLLM path's
        ``max_model_len`` from ``/v1/models``, for the context-window clamp in
        :meth:`_call_engine`.  ``None`` disables the clamp: a router that
        cannot answer (older build, endpoint down) must not break generation,
        so a failed fetch warns once and is retried on a later turn, while an
        answer that names no usable window is cached as unknown.

        A ``context_length`` passed to the constructor is already cached as
        known, so this returns it without asking the router at all — the way
        the window is supplied when the router cannot report it.
        """
        if self._context_length_known:
            return self._context_length
        async with self._context_length_lock:
            if self._context_length_known:
                return self._context_length
            try:
                resp = await self._http.get("/get_server_info")
                resp.raise_for_status()
                info = resp.json()
            except Exception as e:
                if not self._context_length_fetch_warned:
                    self._context_length_fetch_warned = True
                    logger.warning(
                        "%s: could not fetch /get_server_info (%s); running without "
                        "the context-window clamp until it answers — meanwhile the "
                        "engine itself rejects prompt+max_tokens overruns",
                        self.router_url, e,
                    )
                return None
            self._context_length = _served_context_length(info)
            self._context_length_known = True
            if self._context_length is None:
                logger.warning(
                    "%s: /get_server_info names neither context_length nor max_req_input_len; "
                    "the context-window clamp stays off — set context_length in the proxy config "
                    "to pin the window (slime's sgl-router answers this endpoint from its own "
                    "RouterManager stub instead of forwarding a worker's payload)",
                    self.router_url,
                )
            else:
                logger.info("%s: engine context window is %d tokens (from /get_server_info)", self.router_url, self._context_length)
            return self._context_length

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
    misconfigured endpoint list fails when the provider (or the relay worker
    that builds one) is created, not at the first inference request minutes
    into a rollout.
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

    Endpoints come from ``inference_engine_base_url`` /
    ``inference_engine_api_key`` in ``configs/{engine}-direct.yaml`` — lists,
    one entry per running engine instance; a single api_key is broadcast to
    every base URL.  A session is bound to one endpoint round-robin on its
    first request and stays **sticky** for its lifetime, mirroring verl's
    ``acquire_server`` routing: multi-turn prompts of one trial reuse the
    same engine replica and its prefix cache.

    ``stop_reason`` is reported in each engine's **own** vocabulary, which
    the adapters in :mod:`proxyserver.engines` already accept: SGLang's
    ``meta_info.finish_reason.type`` is authoritative, and vLLM's raw
    ``finish_reason`` (``"stop"``/``"length"``/``"abort"``) passes through
    verbatim — a raw ``"length"`` is trusted, while a raw ``"stop"`` is still
    double-checked against the token count (the vllm adapter cannot assume
    truncation reporting, since through verl it never gets any).

    vLLM's ``/v1/completions`` requires a model name; it is discovered once
    per endpoint from ``GET /v1/models`` (the engine serves exactly one
    model) rather than configured.  Each endpoint's context window is
    discovered the same lazy way — vLLM's ``max_model_len`` rides along with
    ``/v1/models``, SGLang's ``context_length`` comes from
    ``GET /get_server_info`` — and caps each turn's ``max_tokens`` to the
    room the window has left (:func:`_clamp_to_context_window`).
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
        # context window comes from /get_server_info (see
        # _sglang_context_length for the fetch-failure semantics).
        self._model_ids: list[str | None] = [None] * len(roots)
        self._model_lens: list[int | None] = [None] * len(roots)
        self._model_lock = asyncio.Lock()
        self._context_lengths: list[int | None] = [None] * len(roots)
        self._context_known: list[bool] = [False] * len(roots)
        self._context_fetch_warned: set[int] = set()
        # (endpoint index, claimed model) pairs already warned about, so a
        # claimed-vs-served mismatch is reported once, not every turn.
        self._model_mismatch_warned: set[tuple[int, str]] = set()
        logger.info(
            "Direct rollout provider routing to %d %s endpoint(s): %s",
            len(roots), self.engine.name, ", ".join(roots),
        )

    def release_session(self, session_id: str) -> None:
        """Free the session's model pin, token stream, and sticky endpoint binding."""
        self._session_endpoints.pop(session_id, None)
        super().release_session(session_id)

    def _acquire_endpoint(self, session_id: str | None) -> int:
        """Resolve the endpoint index for a session — sticky, round-robin
        on first use (mirrors verl's ``acquire_server`` placement)."""
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
                        raise EngineError(
                            f"{self.base_urls[index]}: /v1/models reported no served model"
                        )
                    max_len = models[0].get("max_model_len")
                    self._model_lens[index] = int(max_len) if max_len else None
                    self._model_ids[index] = models[0]["id"]
        return self._model_ids[index]

    async def _sglang_context_length(self, index: int) -> int | None:
        """Endpoint ``index``'s context window, from ``GET /get_server_info``
        (fetched once and cached) — the SGLang counterpart of the vLLM path's
        ``max_model_len`` from ``/v1/models``, for the context-window clamp
        in :meth:`_call_engine`.  ``None`` disables the clamp: an engine that
        cannot answer (older build, endpoint down) must not break generation,
        so a failed fetch warns once and is retried on a later turn, while an
        answer that names no usable window is cached as unknown.
        """
        if self._context_known[index]:
            return self._context_lengths[index]
        async with self._model_lock:
            if self._context_known[index]:
                return self._context_lengths[index]
            try:
                resp = await self._clients[index].get("/get_server_info")
                resp.raise_for_status()
                info = resp.json()
            except Exception as e:
                if index not in self._context_fetch_warned:
                    self._context_fetch_warned.add(index)
                    logger.warning(
                        "%s: could not fetch /get_server_info (%s); running without "
                        "the context-window clamp until it answers — meanwhile the "
                        "engine itself rejects prompt+max_tokens overruns",
                        self.base_urls[index], e,
                    )
                return None
            self._context_lengths[index] = _served_context_length(info)
            self._context_known[index] = True
            if self._context_lengths[index] is None:
                logger.warning("%s: /get_server_info names neither context_length nor max_req_input_len; the context-window clamp stays off", self.base_urls[index])
            else:
                logger.info("%s: engine context window is %d tokens (from /get_server_info)", self.base_urls[index], self._context_lengths[index])
            return self._context_lengths[index]

    async def _call_engine(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        session_id: str | None,
    ) -> tuple[list[int], list[float], str | None, dict[str, Any]]:
        index = self._acquire_endpoint(session_id)
        http = self._clients[index]
        if self.engine.name == SGLANG:
            # Window learned lazily from the endpoint (_sglang_context_length);
            # SGLang itself then reports a clamped turn as finish_reason='length'
            # (reports_truncation=True).
            _clamp_to_context_window(
                sampling_params, len(prompt_ids),
                await self._sglang_context_length(index), self.base_urls[index], session_id,
            )
            return await _sglang_generate(http, prompt_ids, sampling_params, session_id)

        served_model = await self._vllm_model_id(index)
        # Cross-check the agent's claim against what the engine actually
        # serves (vLLM reports its id here anyway): a mismatch means the
        # prompt was tokenized with a vocabulary the engine may not speak —
        # silent training-data corruption, not a crash.  The served id is
        # often a filesystem path, so its basename counts as a match.
        claimed = self._session_models.get(session_id) if session_id is not None else None
        if (
            claimed
            and claimed not in (served_model, served_model.rsplit("/", 1)[-1])
            and (index, claimed) not in self._model_mismatch_warned
        ):
            self._model_mismatch_warned.add((index, claimed))
            logger.warning(
                "Session %s claims model %r but endpoint %s serves %r; the "
                "claimed name selects the tokenizer, so make sure they agree",
                session_id, claimed, self.base_urls[index], served_model,
            )

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
        # Sampling knobs ride through as generate() resolved them (request >
        # bundled generation_config); an absent knob stays absent so the
        # engine's own defaults apply.  top_k/min_p/repetition_penalty are
        # vLLM extensions its OpenAI-compatible /v1/completions accepts.
        for key in (
            "temperature", "top_p", "top_k", "min_p", "stop",
            "repetition_penalty", "presence_penalty", "frequency_penalty",
        ):
            if sampling_params.get(key) is not None:
                payload[key] = sampling_params[key]

        resp = await http.post("/v1/completions", json=payload)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]

        logprobs_blob = choice.get("logprobs") or {}
        token_ids = [
            int(token.split(":", 1)[1])
            for token in logprobs_blob.get("tokens") or []
            if token.startswith("token_id:")
        ]
        log_probs = [float(lp) for lp in logprobs_blob.get("token_logprobs") or [] if lp is not None]
        # vLLM's raw finish_reason — generate() validates token/logprob
        # alignment and non-emptiness before anything is recorded.  vLLM's
        # OpenAI-compatible response reports nothing about the weights it is
        # serving, so there is no engine metadata to carry here.
        return token_ids, log_probs, choice.get("finish_reason"), {}

    async def aclose(self) -> None:
        """Close the underlying HTTP clients."""
        for client in self._clients:
            await client.aclose()
