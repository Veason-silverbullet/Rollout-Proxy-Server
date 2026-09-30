"""OpenAI-compatible recording proxy backed by an in-process rollout provider.

The provider tokenizes prompts, preserves exact sampled token histories, and
calls remote engines through HTTP or VeRL actors through Ray. Sessions are opened lazily using API keys
of the form ``{REAL_API_KEY}{delimiter}{session_id}{delimiter}{agent_id}``.
Each agent owns an independent token stream; recording and deletion operate
on the whole rollout session. Every request must name a mapped model.
"""

from __future__ import annotations
import asyncio
import json
import logging
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable, Coroutine
from uuid import uuid4
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from .engines import DEFAULT_MAX_TOKENS
from .model_registry import UnknownModelError, resolve_profile
from .recorder import DEFAULT_SESSION_ROOT, SessionRecorder, SessionStore

logger = logging.getLogger(__name__)

# Default delimiter between the shared real API key, the session_id, and the
# agent_id inside a keyed api_key
# ("{REAL_API_KEY}{delimiter}{session_id}{delimiter}{agent_id}").  The
# real key may contain this substring; session and agent ids are guaranteed
# by convention never to contain it, so the key is split at the delimiter's
# *last* occurrences. Internal stream ids always use this fixed delimiter,
# even when API-key parsing uses a custom key_delimiter (see stream_id below).
PROXY_KEY_DELIMITER = "-"

# Session and agent ids ride inside the api_key header and in
# /sessions/{id} URL paths, so restrict them to URL/header-safe characters.
# (This charset also excludes the default PROXY_KEY_DELIMITER, which
# keyed-api-key parsing and stream-id composition rely on; a custom
# delimiter must likewise stay outside it.)
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._]{1,256}$")


def stream_id(session_id: str, agent_id: str) -> str:
    """The id of one agent's strict-TITO token stream.

    This is what the inference side keys *all* per-stream state on — token
    streams, model pins, per-stream turn locks, sticky endpoint
    bindings; every layer below the HTTP endpoint treats it as an opaque
    session-like id.
    """
    return f"{session_id}{PROXY_KEY_DELIMITER}{agent_id}"


# Default root for per-session error logs: ./proxyserver/logs
_DEFAULT_LOG_ROOT = Path(__file__).resolve().parent / "logs"


class SessionErrorLogger:
    """Per-session error log files for debugging failed rollouts.

    All sessions of one proxy run share a directory stamped with the
    server start time; each failing session gets its own file:
    ``{root}/{yyyy}-{mm}-{dd}-{hh}-{mm}-{ss}/{session_id}.log``.
    Every line is prefixed with the error occurrence time.  The directory
    is created lazily, so an error-free run leaves nothing behind.
    """

    def __init__(self, root: Path):
        self._run_dir = root / time.strftime("%Y-%m-%d-%H-%M-%S")
        self._lock = threading.Lock()

    def log(self, session_id: str, message: str) -> None:
        """Append one error line.  Never raises — logging failures must not
        break request serving."""
        try:
            with self._lock:
                self._run_dir.mkdir(parents=True, exist_ok=True)
                path = self._run_dir / f"{session_id}.log"
                with path.open("a", encoding="utf-8") as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        except OSError as e:
            logger.warning("Cannot write error log for session %s: %s", session_id, e)

# Type alias for the completion handler that frameworks can inject.
# Signature:
#   async (proxy, session_id, agent_id, model, messages, body, is_streaming) -> Response
# agent_id is always present; handlers key their per-stream inference state on
# stream_id(session_id, agent_id).
CompletionHandler = Callable[
    ["LLMProxyServer", str, str, str, list, dict, bool],
    Coroutine[Any, Any, Any],
]


# ---------------------------------------------------------------------------
# FastAPI application builder
# ---------------------------------------------------------------------------


def _build_app(proxy: "LLMProxyServer") -> FastAPI:
    """Build the FastAPI application that serves as the OpenAI proxy."""

    app = FastAPI(title="LLM Proxy", version="0.5.0")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ---- sampling overrides ------------------------------------------------
    # Training-side sampling policy that wins over request-provided params.
    # Provider callbacks own the config and runtime policy layers.

    @app.get("/sampling_overrides")
    async def get_sampling_overrides():
        if proxy._get_sampling_overrides is None:
            raise HTTPException(status_code=501, detail="Sampling overrides are not configured")
        return proxy._get_sampling_overrides()

    @app.put("/sampling_overrides")
    async def put_sampling_overrides(request: Request):
        """Replace the runtime override layer (the trainer's per-step push).

        Requires the **bare** real API key: this port is published to the
        agents' network, and a policy-mutating endpoint must not be open.
        The keyed per-session form agents hold does not pass.
        """
        if proxy._set_sampling_overrides is None:
            raise HTTPException(status_code=501, detail="Sampling overrides are not configured")
        token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not proxy._api_key or token != proxy._api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Body must be a JSON object of sampling params")
        try:
            return proxy._set_sampling_overrides(body)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ---- routed-experts capture ------------------------------------------
    # Whether /generate requests ask the engines for per-token MoE expert
    # selections (R3, slime's --use-rollout-routing-replay).  Local slime
    # transport only: the provider owns the capture layers.

    @app.get("/routed_experts")
    async def get_routed_experts():
        if proxy._get_routed_experts_config is None:
            raise HTTPException(status_code=501, detail="Routed-experts capture requires the slime transport")
        return proxy._get_routed_experts_config()

    @app.put("/routed_experts")
    async def put_routed_experts(request: Request):
        """Set the runtime capture layer (the trainer's per-step toggle).

        Body: ``{"enabled": true|false|null}`` — ``null`` clears the layer,
        falling back to the config baseline.  Requires the **bare** real API
        key, exactly as ``PUT /sampling_overrides``: this port is published
        to the agents' network, and an endpoint that mutates what training
        data is captured must not be open.
        """
        if proxy._set_routed_experts_config is None:
            raise HTTPException(status_code=501, detail="Routed-experts capture requires the slime transport")
        token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not proxy._api_key or token != proxy._api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict) or "enabled" not in body:
            raise HTTPException(status_code=400, detail='Body must be a JSON object {"enabled": true|false|null}')
        try:
            return proxy._set_routed_experts_config(body["enabled"])
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ---- tokenizer identity ----------------------------------------------
    # Training–rollout consistency: the trainer compares the serving
    # tokenizer's fingerprint against its --hf-checkpoint's before any trial
    # runs (harbor.slime_bridge.rollout._sync_tokenizer_identity).  Local
    # tokenizers belong to the provider on this process.

    @app.get("/tokenizer_fingerprint")
    async def get_tokenizer_fingerprint(model: str):
        """Identity fingerprint of the tokenizer serving ``model``.

        Open like ``GET /sampling_overrides`` — it reveals only hashes.
        Computed on the runtime tokenizer and cached per profile; the first
        call hashes the full vocab, so it runs off the event loop.
        """
        if proxy._get_tokenizer_fingerprint is None:
            raise HTTPException(status_code=501, detail="Tokenizer fingerprints are not configured")
        try:
            return await asyncio.to_thread(proxy._get_tokenizer_fingerprint, model)
        except UnknownModelError as e:
            raise HTTPException(status_code=404, detail=str(e))

    # ---- session management ----------------------------------------------
    # Sessions are opened lazily by the first keyed completion request;
    # these endpoints let the rollout driver fetch, complete, and delete
    # the recorded sessions.

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        # Serialized under the recorder lock and off the event loop: the
        # record grows with the session, and a turn recording concurrently
        # in a worker thread must not mutate it mid-dump.
        try:
            data = await asyncio.to_thread(proxy.recorder.dump_session, session_id)
        except RuntimeError as e:
            # An unresolved recording failure may leave the record short:
            # refuse to serve it so the driver's collection fails the trial
            # into a whole-task retry instead of training on a record that
            # omits a committed turn.
            raise HTTPException(status_code=500, detail=str(e)) from e
        if data is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return data

    @app.post("/sessions/{session_id}/complete")
    async def complete_session(session_id: str):
        # Off the event loop: this rewrites the full session JSON on disk.
        await asyncio.to_thread(proxy.recorder.mark_completed, session_id)
        return {"session_id": session_id, "status": "completed"}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        try:
            await proxy.delete_session(session_id)
        except Exception as e:
            proxy.error_log.log(session_id, f"session cleanup failed: {e}")
            raise HTTPException(status_code=503, detail=f"Session cleanup failed: {e}") from e
        return {"session_id": session_id, "status": "deleted"}

    # ---- OpenAI-compatible chat completions ------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Generate a chat completion and record data.

        Keyed routing: the session_id and agent_id are extracted from the
        ``Authorization: Bearer {REAL_API_KEY}{delimiter}{session_id}{delimiter}{agent_id}``
        header and the session is created lazily on the first request.
        The request's ``model`` claim is decided here — required, and
        rejected unless the proxy's tokenizer mapping serves it — then
        processed by the injected provider handler.
        """
        token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        parsed = proxy.parse_session_key(token)
        if parsed is None:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        session_id, agent_id = parsed

        body = await request.json()
        messages: list[dict[str, Any]] = body.get("messages", [])
        is_streaming: bool = body.get("stream", False)

        # Error-log lines share the session's file; name the agent so a
        # multi-agent rollout's log stays attributable.
        who = f"agent {agent_id}: "

        def _reject(status: int, message: str) -> JSONResponse:
            proxy.error_log.log(session_id, f"{who}HTTP {status}: {message}")
            return JSONResponse(status_code=status, content={"error": message})

        # A deleted session's stragglers are refused outright.  The driver
        # deletes a session when its rollout is over — for a cancelled trial
        # that happens while the agent side is still running, and each turn
        # it keeps requesting would otherwise occupy an engine with a
        # completion nobody will train on and re-create per-session state
        # (record, lock, stream tracking) that nothing ever cleans up again.
        # 410 identifies a deleted rollout while its tombstone is retained;
        # retention is bounded by the recorder's TTL and capacity.
        if proxy.recorder.is_deleted(session_id):
            return _reject(
                410,
                f"session {session_id} was deleted; its rollout is over and "
                f"no further turns will be served",
            )

        # The agent's model claim, decided by the proxy: the request names
        # the model, the tokenizer mapping says whether the proxy serves it.
        model = body.get("model")
        if not model:
            return _reject(
                400,
                "request names no model; set the OpenAI 'model' field so the "
                "proxy can resolve its tokenizer",
            )
        try:
            resolve_profile(model)
        except UnknownModelError as e:
            return _reject(404, str(e))
        except (FileNotFoundError, ValueError) as e:
            # Mapped but unservable: missing tokenizer directory on this
            # node, a broken profile module, or a missing/malformed mapping
            # file — server-side misconfiguration, not the agent's claim, so
            # 500 rather than 404, but still a clean logged JSON error instead
            # of a traceback.
            return _reject(500, str(e))

        proxy.recorder.ensure_session(session_id, model_name=model)
        stream = stream_id(session_id, agent_id)
        proxy._track_stream(session_id, stream)
        untrack = proxy._track_session_task(session_id, asyncio.current_task())

        # Handlers report failures as error responses (or raise); mirror
        # them into the session's error log for post-mortem debugging.
        # Turns of one *stream* are serialized: each agent's strict-TITO
        # stream is stateful, and an agent SDK's automatic retry may race
        # the request it retries — the lock parks the retry until the
        # original commits, so the retry is then recognized as an exact
        # repeat and re-served instead of interleaving with (or re-sampling
        # over) the original.  Locks are per stream, not per session, so a
        # rollout's agents generate concurrently.
        try:
            async with proxy._session_lock(stream):
                if proxy.recorder.is_deleted(session_id):
                    return _reject(410, f"session {session_id} was deleted")
                response = await proxy._completion_handler(proxy, session_id, agent_id, model, messages, body, is_streaming)
        except asyncio.CancelledError:
            if proxy.recorder.is_deleted(session_id):
                return _reject(410, f"session {session_id} was deleted")
            raise
        except Exception as e:
            proxy.error_log.log(session_id, f"{who}unhandled {type(e).__name__}: {e}")
            raise
        finally:
            untrack()
        status = getattr(response, "status_code", 200)
        if status >= 400:
            detail = getattr(response, "body", b"").decode("utf-8", "replace")
            proxy.error_log.log(session_id, f"{who}HTTP {status}: {detail}")
        return response


    return app


# ---------------------------------------------------------------------------
# Request parameter helpers
# ---------------------------------------------------------------------------


def _sampling_param(body: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read a sampling parameter, treating an explicit JSON ``null`` like an
    omitted field (OpenAI semantics).  Omitted sampling knobs stay ``None``
    all the way into the provider, which fills them from the model's bundled
    ``generation_config.json`` defaults, then leaves the rest to the engine —
    the proxy never invents a hardcoded value of its own."""
    value = body.get(key)
    return default if value is None else value


def _max_tokens_param(body: dict[str, Any], default: int = DEFAULT_MAX_TOKENS) -> Any:
    """The request's completion-token budget.

    ``max_completion_tokens`` is OpenAI's current name for it and wins when
    present; ``max_tokens`` is the deprecated spelling agents still commonly
    send.  Honoring only the old name silently capped modern SDK users at
    the default.  An explicit JSON ``null`` counts as omitted (OpenAI
    semantics), falling through to the next source."""
    value = body.get("max_completion_tokens")
    if value is None:
        value = body.get("max_tokens")
    return default if value is None else value


# ---------------------------------------------------------------------------
# OpenAI response builders
# ---------------------------------------------------------------------------


def _reasoning_of(model: str, completion_text: str) -> str | None:
    """The think-block reasoning of ``completion_text`` per ``model``'s
    profile reasoning parser, or ``None`` (no block, or a no-reasoning
    model).  Never raises — a response must build even if the model's
    mapping entry changed mid-flight."""
    if not completion_text:
        return None
    try:
        return resolve_profile(model).reasoning.extract(completion_text)
    except Exception:
        return None


def _message_fields(
    model: str,
    completion_text: str,
    content_text: str,
    tool_calls: list[dict[str, Any]] | None,
) -> tuple[str | None, str | None]:
    """The assistant message's ``(content, reasoning_content)`` pair.

    One definition of the agent-visible fields extracted from a completion,
    shared by :func:`build_openai_response` and the session-record call
    sites so the recorded turn always matches the message the agent saw:
    ``content`` is the tool parser's agent-clean text (``None`` on a
    tool-call turn with no surrounding text), ``reasoning_content`` the
    think block extracted from the raw ``completion_text`` per ``model``'s
    profile (``None`` when the model emitted none).
    """
    if tool_calls:
        content = content_text.strip() if content_text and content_text.strip() else None
    else:
        content = content_text
    return content, _reasoning_of(model, completion_text)


def build_openai_response(
    *,
    token_ids: list[int],
    content_text: str,
    tool_calls: list[dict[str, Any]] | None,
    finish_reason: str,
    model: str,
    prompt_tokens: int = 0,
    completion_text: str = "",
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat completion response dict.

    ``model`` is the agent's validated model claim — the model the session
    is pinned to.  ``completion_text`` is the raw decode of ``token_ids``;
    when the model emitted a thinking block, the assistant message carries
    it as ``reasoning_content`` (DeepSeek-style) alongside the reasoning-
    stripped ``content``.
    """
    content, reasoning = _message_fields(model, completion_text, content_text, tool_calls)
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning is not None:
        message["reasoning_content"] = reasoning

    return {
        "id": f"chatcmpl-{uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(token_ids),
            "total_tokens": prompt_tokens + len(token_ids),
        },
    }


def build_openai_stream_response(response_body: dict[str, Any]) -> StreamingResponse:
    """Replay a completed chat completion as an OpenAI SSE stream.

    The completion pipeline finishes generation, validation, token-stream
    commit, and recording before building a response. With that policy,
    ``stream: true`` is honored in *format only*: the finished response from
    :func:`build_openai_response` is emitted as ``chat.completion.chunk``
    events followed by ``[DONE]``. All transport modes use this response
    format, and the turn is already recorded before the first byte is sent.
    """
    choice = response_body["choices"][0]
    message = choice["message"]

    delta: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    if message.get("reasoning_content") is not None:
        delta["reasoning_content"] = message["reasoning_content"]
    if message.get("tool_calls"):
        # Streaming tool calls carry a per-entry index for client-side merging.
        delta["tool_calls"] = [
            {"index": i, **tc} for i, tc in enumerate(message["tool_calls"])
        ]

    base = {
        "id": response_body["id"],
        "object": "chat.completion.chunk",
        "created": response_body.get("created", int(time.time())),
        "model": response_body["model"],
    }
    chunks = [
        {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
            "usage": response_body["usage"],
        },
    ]

    async def _sse():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Completion handler (framework-neutral, built around a provider)
# ---------------------------------------------------------------------------


def make_completion_handler(provider: Any):
    """Build the completion handler around a rollout provider.

    ``provider`` is any :class:`~proxyserver.rollout_provider.BaseRolloutProvider`
    (slime's router transport, direct's engine transport, ...) — the handler itself
    is framework-neutral: generate strictly
    token-in-token-out, parse tool calls, record the turn, format with
    :func:`build_openai_response`. A strict-TITO violation (including a
    mid-session model switch) maps to 400; an unknown model maps to 404.
    Upstream 4xx errors are preserved, upstream 5xx errors map to 502, and
    other generation failures map to 500.

    A caller that gives up mid-turn (an agent-side timeout or disconnect)
    does not abort the turn: depending on the installed Starlette, a client
    disconnect may cancel the endpoint task, and unshielded that
    cancellation would propagate into the provider's inference call and abort
    the engine's generation — losing the expensive turn, and making a turn
    longer than the agent's timeout impossible to ever complete (each retry
    would re-abort at the same point).  The generate→record pipeline
    therefore runs shielded and holds its own per-stream lock through
    parsing and recording. An exact retry waits until the detached pipeline
    has recorded its turn before re-serving the committed result.
    """
    from .token_stream import TokenStreamError

    async def _run_completion(
        proxy: "LLMProxyServer",
        session_id: str,
        agent_id: str,
        model: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
        is_streaming: bool,
    ):
        # The provider keys all per-stream state (token stream, model pin,
        # sticky endpoint, turn lock) on what it calls a session id; hand it
        # the composed stream id so each agent owns an independent stream.
        stream = stream_id(session_id, agent_id)
        # Rows/cols of the routed-experts blob already recorded for this
        # stream, so the engine returns only this turn's delta (R3).  Only a
        # continuation extends that blob: a request with no assistant message
        # opens a new conversation whose stream starts from scratch, so the
        # stored rows describe a different stream and must not offset it.
        session_record = proxy.recorder.get_session(session_id)
        continues = any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages)
        prior_blob = session_record.routed_experts.get(agent_id) if session_record and continues else None
        try:
            prompt_ids, token_ids, log_probs, finish_reason, completion_text, engine_meta = (
                await provider.generate(
                    messages=messages,
                    session_id=stream,
                    routed_experts_prior=(prior_blob.rows, prior_blob.cols) if prior_blob else None,
                    model=model,
                    temperature=_sampling_param(body, "temperature"),
                    top_p=_sampling_param(body, "top_p"),
                    top_k=_sampling_param(body, "top_k"),
                    min_p=_sampling_param(body, "min_p"),
                    max_tokens=_max_tokens_param(body),
                    stop=body.get("stop"),
                    tools=body.get("tools"),
                    repetition_penalty=body.get("repetition_penalty"),
                    presence_penalty=_sampling_param(body, "presence_penalty"),
                    frequency_penalty=_sampling_param(body, "frequency_penalty"),
                )
            )
            content_text, tool_calls, finish_reason = await provider.parse_tool_calls(
                stream, token_ids, completion_text, finish_reason,
                tools=body.get("tools"),
            )
        except UnknownModelError as e:
            # The endpoint validates the model against the mapping before
            # dispatch, so reaching this means the provider's own resolution
            # disagrees (e.g. a mapping edit mid-flight). Return the same
            # HTTP 404 and error-message field as endpoint validation.
            logger.error("Unknown model for stream %s: %s", stream, e)
            return JSONResponse(status_code=404, content={"error": str(e)})
        except TokenStreamError as e:
            logger.error("Strict TITO violation for stream %s: %s", stream, e)
            return JSONResponse(status_code=400, content={"error": str(e)})
        except httpx.HTTPStatusError as e:
            # Mirror an upstream 4xx instead of flattening it to 500.
            status = e.response.status_code
            detail = (e.response.text or "").strip()[:1000]
            logger.error("Upstream HTTP %s for stream %s: %s", status, stream, detail or e)
            if status < 500:
                return JSONResponse(
                    status_code=status,
                    content={"error": f"engine rejected the request (HTTP {status}): {detail}"},
                )
            return JSONResponse(
                status_code=502,
                content={"error": f"engine error (HTTP {status}): {detail}"},
            )
        except Exception as e:
            logger.error("Generate failed for stream %s: %s", stream, e)
            return JSONResponse(status_code=500, content={"error": str(e)})

        # Record off the event loop: the session JSON
        # rewrite grows with the session.
        content, reasoning_content = _message_fields(model, completion_text, content_text, tool_calls)
        try:
            await asyncio.to_thread(
                proxy.recorder.record_completion,
                session_id=session_id,
                agent_id=agent_id,
                messages=messages,
                completion_text=completion_text,
                token_ids=token_ids,
                logprobs=log_probs,
                finish_reason=finish_reason,
                prompt_token_ids=prompt_ids,
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
                weight_version=engine_meta.get("weight_version"),
                routed_experts=engine_meta.get("routed_experts"),
            )
        except Exception as e:
            # The turn is already committed to the engine's token stream, so
            # this cannot be undone: an exact retry re-serves it, and a
            # session record missing it would silently omit the turn's
            # sampled tokens from training. Block collection until an exact
            # retry records this turn (or verifies it was already appended
            # before a snapshot/save error). Other successful turns cannot
            # clear the failure. Answer 500 so the agent retries recording.
            logger.error(
                "Recording failed for session %s (agent %s) after the turn "
                "was committed: %s", session_id, agent_id, e,
            )
            await asyncio.to_thread(
                proxy.recorder.mark_recording_failed,
                session_id,
                agent_id=agent_id,
                messages=messages,
                prompt_token_ids=prompt_ids,
                token_ids=token_ids,
            )
            return JSONResponse(
                status_code=500,
                content={"error": f"failed to record the committed turn: {e}"},
            )

        response_body = build_openai_response(
            token_ids=token_ids,
            content_text=content_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=len(prompt_ids),
            model=model,
            completion_text=completion_text,
        )
        if is_streaming:
            return build_openai_stream_response(response_body)
        return JSONResponse(content=response_body)

    async def _handle_completion(
        proxy: "LLMProxyServer",
        session_id: str,
        agent_id: str,
        model: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
        is_streaming: bool,
    ):
        async def run_serialized():
            stream = stream_id(session_id, agent_id)
            lock = proxy._completion_locks.setdefault(stream, asyncio.Lock())
            # The HTTP request lock is released when its caller disconnects.
            # This lock belongs to the shielded pipeline and survives that
            # cancellation, preventing retries from overtaking parsing or
            # recording after the provider has committed its token stream.
            async with lock:
                if proxy.recorder.is_deleted(session_id):
                    return JSONResponse(status_code=410, content={"error": f"session {session_id} was deleted"})
                return await _run_completion(proxy, session_id, agent_id, model, messages, body, is_streaming)

        task = asyncio.create_task(run_serialized())
        proxy._track_session_task(session_id, task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled() or proxy.recorder.is_deleted(session_id):
                raise  # the pipeline itself was torn down; nothing to preserve
            # The caller was torn down (agent timeout/disconnect) but the
            # turn may be mid-generation and is expensive: let it finish
            # detached so it still commits and is recorded — the agent's
            # exact retry is then re-served the same turn (the recorder
            # deduplicates the double delivery).
            proxy.error_log.log(
                session_id,
                f"agent {agent_id}: "
                "caller gave up mid-turn (timeout or disconnect); finishing "
                "the turn detached and recording it (an exact retry of the "
                "request is re-served this turn)",
            )
            proxy._orphan_tasks.add(task)
            task.add_done_callback(proxy._reap_orphan_task)
            raise

    return _handle_completion


# ---------------------------------------------------------------------------
# LLMProxyServer
# ---------------------------------------------------------------------------


class LLMProxyServer:
    """Serve and record completions using a required in-process handler.

    Build the handler with ``make_completion_handler(provider)`` and
    pass ``provider.release_session`` as ``on_session_deleted``. The CLI
    wires provider callbacks for sampling, tokenizer identity, and capture.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 0,
        completion_handler: CompletionHandler | None = None,
        on_session_deleted: Callable[[str], None] | None = None,
        get_sampling_overrides: Callable[[], dict[str, Any]] | None = None,
        set_sampling_overrides: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        get_routed_experts_config: Callable[[], dict[str, Any]] | None = None,
        set_routed_experts_config: Callable[[bool | None], dict[str, Any]] | None = None,
        get_tokenizer_fingerprint: Callable[[str], dict[str, Any]] | None = None,
        api_key: str | None = None,
        error_log_dir: str | Path | None = None,
        save_rollout_sessions: bool = True,
        save_rollout_logprobs: bool = True,
        save_rollout_routed_experts: bool = False,
        session_dir: str | Path | None = None,
        key_delimiter: str | None = None,
    ):
        """
        Args:
            host: Bind address for the HTTP server.
            port: Port number. ``0`` means auto-select a free port.
            api_key: Shared secret for keyed routing — agents call
                ``POST /v1/chat/completions`` with
                ``api_key = f"{api_key}{delimiter}{session_id}{delimiter}{agent_id}"``.
                When ``None`` every completion request is rejected with 401,
                so any proxy that serves agents needs a key.
            error_log_dir: Root directory for per-session error logs
                (default ``proxyserver/logs``).  Each proxy run logs to
                ``{server-start-time}/{session_id}.log`` under this root.
            save_rollout_sessions: Persist every recorded session to disk as
                ``{session_dir}/{server-start-time}/{session_id}.json``,
                rewritten after each turn.  On by default; set ``False`` to
                keep records in memory only (fetchable via
                ``GET /sessions/{id}`` until deleted).
            save_rollout_logprobs: Include ``completion_logprobs`` in the
                persisted session JSON.  On by default; set ``False`` to
                trim the on-disk records.  The in-memory record always
                carries the logprobs, so ``GET /sessions/{id}`` is
                unaffected.  Ignored when ``save_rollout_sessions`` is
                ``False``.
            save_rollout_routed_experts: Include the ``routed_experts``
                blobs in the persisted session JSON.  **Off** by default —
                a session file is rewritten in full after every turn, and
                the blobs are large — and, like the logprobs, the
                in-memory record (``GET /sessions/{id}``) always carries
                them.  Ignored when ``save_rollout_sessions`` is ``False``.
            session_dir: Root directory for the persisted session records
                (default ``proxyserver/sessions``).  Ignored when
                ``save_rollout_sessions`` is ``False``.
            key_delimiter: Delimiter separating the real API key, session_id,
                and agent_id inside a keyed api_key (default :data:`PROXY_KEY_DELIMITER`,
                ``"-"``; configurable as ``proxy_api_delimiter`` in 
                ``configs/{engine}-{transport}.yaml``).
            completion_handler: Required async callable for inference.  Signature:
                ``async (proxy, session_id, agent_id, model, messages, body, is_streaming) -> Response``
                (``agent_id`` is always present; per-stream inference state
                belongs under ``stream_id(session_id, agent_id)``).
            on_session_deleted: Optional callback invoked when a session is
                deleted — once per stream the session's agents used, with
                the stream id (see :meth:`delete_session`).  Useful for
                framework-specific cleanup (e.g. releasing sticky-session
                bindings; the CLI passes the provider's
                ``release_session`` here).
            get_sampling_overrides: Optional callable backing
                ``GET /sampling_overrides`` (the CLI passes the
                provider's ``get_sampling_overrides``).  Without it the
                endpoint answers 501.
            set_sampling_overrides: Optional callable backing
                ``PUT /sampling_overrides`` (the CLI passes the
                provider's ``set_runtime_sampling_overrides``); may raise
                ``ValueError``, surfaced as HTTP 400.
            get_routed_experts_config: Optional callable backing
                ``GET /routed_experts`` (the Slime transport passes the
                provider's ``get_routed_experts_config``).  Without it the
                endpoint answers 501 — only the slime transport captures
                routed experts.
            set_routed_experts_config: Optional callable backing
                ``PUT /routed_experts`` (the Slime transport passes the
                provider's ``set_runtime_routed_experts``); may raise
                ``ValueError``, surfaced as HTTP 400.
            get_tokenizer_fingerprint: Optional callable backing
                ``GET /tokenizer_fingerprint`` (the CLI passes the
                provider's ``tokenizer_fingerprint``).  Without it the
                endpoint answers 501. May raise ``UnknownModelError``,
                surfaced as HTTP 404.
        """
        self.host = host
        self.port = port
        self._on_session_deleted = on_session_deleted
        self._get_sampling_overrides = get_sampling_overrides
        self._set_sampling_overrides = set_sampling_overrides
        self._get_routed_experts_config = get_routed_experts_config
        self._set_routed_experts_config = set_routed_experts_config
        self._get_tokenizer_fingerprint = get_tokenizer_fingerprint
        self.key_delimiter = key_delimiter or PROXY_KEY_DELIMITER
        self._api_key = api_key

        if not callable(completion_handler):
            raise ValueError("completion_handler is required; build it with make_completion_handler(provider)")
        self._completion_handler = completion_handler
        self.error_log = SessionErrorLogger(
            Path(error_log_dir) if error_log_dir is not None else _DEFAULT_LOG_ROOT,
        )
        self.session_store = (
            SessionStore(
                Path(session_dir) if session_dir is not None else DEFAULT_SESSION_ROOT,
            )
            if save_rollout_sessions
            else None
        )
        self.recorder = SessionRecorder(
            store=self.session_store,
            save_logprobs=save_rollout_logprobs,
            save_routed_experts=save_rollout_routed_experts,
        )
        # Keep detached completion pipelines alive until they finish recording.
        self._orphan_tasks: set[asyncio.Task] = set()

        # Per-stream turn serialization: turns of one agent's stream must be
        # sequential (the strict-TITO stream is stateful), but an agent
        # SDK's automatic retry can race the request it retries.  The lock
        # parks the retry until the original commits, which also makes it
        # recognizable as an exact repeat to re-serve.  Keyed by stream id
        # (see stream_id), so a rollout's agents run concurrently.
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Completion pipelines keep serialization through recording
        # even after cancellation releases the caller's HTTP request lock.
        self._completion_locks: dict[str, asyncio.Lock] = {}
        # session_id -> stream ids its agents used, so DELETE /sessions/{id}
        # can release every stream of a multi-agent rollout.
        self._session_streams: dict[str, set[str]] = {}
        # Includes requests queued on turn locks and shielded completion pipelines.
        self._session_tasks: dict[str, set[asyncio.Task]] = {}
        self._deletion_tasks: dict[str, asyncio.Task] = {}

        self.app = _build_app(self)

        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task | None = None
        self._actual_port: int | None = None

    # -- session management ------------------------------------------------

    def parse_session_key(self, token: str) -> tuple[str, str] | None:
        """Extract ``(session_id, agent_id)`` from a keyed api_key:
        ``{REAL_API_KEY}{delimiter}{session_id}{delimiter}{agent_id}``.

        The real key may itself contain the delimiter; session and agent
        ids never do, so the token is split at the delimiter's *last*
        occurrences.  Returns ``None`` (never raises) when keyed routing is
        disabled, the real-key part does not match, or an id is missing or
        contains unsafe characters.
        """
        if not self._api_key or not token:
            return None
        head, sep, agent_id = token.rpartition(self.key_delimiter)
        if not sep:
            return None
        real_key, sep2, session_id = head.rpartition(self.key_delimiter)
        if (
            not sep2
            or real_key != self._api_key
            or not _SESSION_ID_RE.match(session_id)
            or not _SESSION_ID_RE.match(agent_id)
        ):
            return None
        return session_id, agent_id

    def _session_lock(self, stream: str) -> asyncio.Lock:
        """The stream's turn-serialization lock (created on first use).

        Keyed by stream id: each agent's turns serialize independently, so
        a rollout's agents generate concurrently.
        """
        lock = self._session_locks.get(stream)
        if lock is None:
            lock = self._session_locks.setdefault(stream, asyncio.Lock())
        return lock

    def _track_stream(self, session_id: str, stream: str) -> None:
        """Remember that ``session_id`` used stream ``stream``, so deleting
        the session can release every stream its agents opened."""
        self._session_streams.setdefault(session_id, set()).add(stream)

    def _track_session_task(self, session_id: str, task: asyncio.Task | None) -> Callable[[], None]:
        if task is None:
            raise RuntimeError("Session work must run in an asyncio task")
        tasks = self._session_tasks.setdefault(session_id, set())
        tasks.add(task)

        def done(finished: asyncio.Task) -> None:
            tasks.discard(finished)
            if not tasks and self._session_tasks.get(session_id) is tasks:
                self._session_tasks.pop(session_id, None)

        task.add_done_callback(done)

        def untrack() -> None:
            task.remove_done_callback(done)
            done(task)

        return untrack


    def _reap_orphan_task(self, task: asyncio.Task) -> None:
        """Done-callback for detached turn-preserving tasks: drop the strong
        reference and consume the outcome, so a failure surfaces as one
        error line instead of a garbage-collection-time "exception was never
        retrieved" warning."""
        self._orphan_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Detached turn-preserving task failed: %s", task.exception())

    async def delete_session(self, session_id: str) -> None:
        """Refuse new turns, cancel active work, then release every stream.

        Concurrent DELETEs share cleanup, which survives a caller disconnect.
        A failed cleanup retains stream bindings so another DELETE can retry.
        """
        task = self._deletion_tasks.get(session_id)
        if task is None:
            self.recorder.delete_session(session_id)
            task = asyncio.create_task(self._delete_session(session_id))
            self._deletion_tasks[session_id] = task

            def done(finished: asyncio.Task) -> None:
                self._deletion_tasks.pop(session_id, None)
                if not finished.cancelled() and finished.exception() is not None:
                    logger.error("Session %s cleanup failed: %s", session_id, finished.exception())

            task.add_done_callback(done)
        await asyncio.shield(task)

    async def _delete_session(self, session_id: str) -> None:
        tasks = set(self._session_tasks.get(session_id, ()))
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.wait(tasks)
            for task in tasks:
                if not task.cancelled():
                    task.exception()
        streams = self._session_streams.get(session_id) or {session_id}
        errors = []
        for stream in sorted(streams):
            try:
                if self._on_session_deleted is not None:
                    self._on_session_deleted(stream)
                self._session_locks.pop(stream, None)
                self._completion_locks.pop(stream, None)
            except Exception as e:
                errors.append((stream, e))
        if errors:
            raise RuntimeError(f"Failed to release streams: {errors}") from errors[0][1]
        self._session_streams.pop(session_id, None)

    # -- HTTP server lifecycle ---------------------------------------------

    @property
    def url(self) -> str | None:
        if self._actual_port is None:
            return None
        return f"http://{self.host}:{self._actual_port}"

    async def start(self) -> str:
        """Start the HTTP server.  Returns the base URL."""
        if self.port == 0:
            self._actual_port = _find_free_port()
        else:
            self._actual_port = self.port

        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self._actual_port,
            log_level="info",
            ws="none",
        )
        server = uvicorn.Server(config)

        async def _serve() -> None:
            try:
                await server.serve()
            except SystemExit as e:
                raise RuntimeError(f"uvicorn startup failed (exit code {e.code})") from e

        self._server = server
        self._serve_task = asyncio.create_task(_serve())

        for _ in range(100):
            if server.started:
                break
            if self._serve_task.done():
                break
            await asyncio.sleep(0.1)
        if not server.started:
            task, self._serve_task, self._server = self._serve_task, None, None
            if task.done():
                task.result()
            else:
                task.cancel()
            raise RuntimeError(f"Proxy HTTP server failed to start on {self.host}:{self._actual_port}")

        logger.info("LLM Proxy started at %s", self.url)
        return self.url

    async def stop(self) -> None:
        """Gracefully shut down the HTTP server."""
        if self._server is not None:
            self._server.should_exit = True
            if self._serve_task is not None:
                try:
                    await self._serve_task
                except asyncio.CancelledError:
                    logger.debug("Proxy serve task was cancelled during shutdown")
                self._serve_task = None
            self._server = None
            # Let detached completion pipelines whose callers gave up finish before
            # returning. Also finish deletions whose callers disconnected,
            # so provider resources are not closed beneath their cleanup.
            pending = self._orphan_tasks | set(self._deletion_tasks.values())
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            logger.info("LLM Proxy stopped")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
