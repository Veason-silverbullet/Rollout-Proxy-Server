"""OpenAI-compatible recording proxy backed by an in-process rollout provider.

The provider tokenizes prompts, preserves exact sampled token histories, and calls
native engine HTTP endpoints, directly or through routers. Sessions are opened
lazily using API keys of the form ``{REAL_API_KEY}{delimiter}{session_id}{delimiter}{agent_id}``.
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
from .protocol import PROXY_INSTANCE_HEADER, PROXY_KEY_DELIMITER, PROXY_RESTART_MESSAGE
from .recorder import SessionRecorder, SessionStore
from .resources import storage_path
from .request_validation import validate_chat_request, validate_reasoning_effort
from .token_stream import ParsedCompletion, TokenStreamError

logger = logging.getLogger(__name__)

# URL/header-safe ids, excluding the key delimiter and the internal "-" separator;
# custom key delimiters must stay outside this charset too.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9._]{1,256}$")


def stream_id(session_id: str, agent_id: str) -> str:
    """The id all per-stream inference state is keyed on."""
    return f"{session_id}-{agent_id}"


class SessionErrorLogger:
    """Per-session error log files for debugging failed rollouts.

    All sessions of one proxy run share a directory stamped with the
    server start time; each failing session gets its own file:
    ``{root}/{yyyy}-{mm}-{dd}-{hh}-{mm}-{ss}/{session_id}.log``.
    Every line is prefixed with the error occurrence time.  The directory
    is created lazily, so an error-free run leaves nothing behind.
    """

    def __init__(self, root: Path):
        self._run_dir = root.expanduser().resolve() / time.strftime("%Y-%m-%d-%H-%M-%S")
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

# async (proxy, session_id, agent_id, model, messages, body, is_streaming) -> Response
# Handlers key per-stream inference state on stream_id(session_id, agent_id).
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

    @app.middleware("http")
    async def instance_binding(request: Request, call_next):
        expected = request.headers.get(PROXY_INSTANCE_HEADER)
        if expected is not None and expected != proxy.instance_id:
            response = JSONResponse(status_code=412, content={"detail": PROXY_RESTART_MESSAGE})
        else:
            response = await call_next(request)
        response.headers[PROXY_INSTANCE_HEADER] = proxy.instance_id
        return response

    @app.get("/health")
    async def health():
        return {"status": "ok", "instance_id": proxy.instance_id}

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
    # The generate-protocol providers own the per-token MoE capture layers (R3).

    @app.get("/routed_experts")
    async def get_routed_experts():
        if proxy._get_routed_experts_config is None:
            raise HTTPException(status_code=501, detail="Routed-experts capture requires the sglang_generate or vllm_generate transport")
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
            raise HTTPException(status_code=501, detail="Routed-experts capture requires the sglang_generate or vllm_generate transport")
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
    # Collectors compare the serving tokenizer against the training checkpoint
    # before starting rollouts.

    @app.get("/tokenizer_fingerprint")
    async def get_tokenizer_fingerprint(model: str):
        """Identity fingerprint of the tokenizer serving ``model``.

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

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        try:
            data = await asyncio.to_thread(proxy.recorder.dump_session, session_id)
        except RuntimeError as e:
            # An unresolved recording failure may leave the record short.
            raise HTTPException(status_code=500, detail=str(e)) from e
        if data is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return data

    @app.post("/sessions/{session_id}/finalize")
    async def finalize_session(session_id: str):
        await proxy.finalize_session(session_id)
        return {"session_id": session_id, "status": "finalized"}

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

        Session and agent ids come from the keyed API key; the session is
        created lazily. The model must be served by the tokenizer mapping.
        """
        token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        parsed = proxy.parse_session_key(token)
        if parsed is None:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        session_id, agent_id = parsed

        who = f"agent {agent_id}: "

        def _reject(status: int, message: str) -> JSONResponse:
            proxy.error_log.log(session_id, f"{who}HTTP {status}: {message}")
            return JSONResponse(status_code=status, content={"error": message})

        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            return _reject(400, "Body must be valid JSON")
        try:
            await asyncio.to_thread(validate_chat_request, body)
        except (ValueError, TokenStreamError) as e:
            return _reject(400, str(e))
        messages = body["messages"]
        is_streaming = body.get("stream", False)

        # Refuse stragglers of a deleted session while its tombstone lasts.
        if proxy.recorder.is_deleted(session_id):
            return _reject(
                410,
                f"session {session_id} was deleted; its rollout is over and "
                f"no further turns will be served",
            )

        model = body["model"]
        try:
            profile = resolve_profile(model)
        except UnknownModelError as e:
            return _reject(404, str(e))
        except (FileNotFoundError, ValueError) as e:
            # Mapped but unservable: server-side misconfiguration.
            return _reject(500, str(e))

        try:
            validate_reasoning_effort(body.get("reasoning_effort"), profile.reasoning_efforts)
        except ValueError as e:
            return _reject(400, str(e))

        # Admission and task registration contain no await: finalization either
        # includes this request (even while queued), or refuses it outright.
        if session_id in proxy._finalization_tasks:
            return _reject(409, f"session {session_id} is finalizing or finalized")
        proxy.recorder.ensure_session(session_id, model_name=model)
        stream = stream_id(session_id, agent_id)
        proxy._track_stream(session_id, stream)
        untrack = proxy._track_session_task(session_id, asyncio.current_task())

        try:
            response = await proxy._run_turn(session_id, agent_id, model, messages, body, is_streaming)
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


def _max_tokens_param(body: dict[str, Any], default: int = DEFAULT_MAX_TOKENS) -> Any:
    """The completion-token budget; ``max_completion_tokens`` wins over
    ``max_tokens``, and JSON ``null`` counts as omitted."""
    value = body.get("max_completion_tokens")
    if value is None:
        value = body.get("max_tokens")
    return default if value is None else value


# ---------------------------------------------------------------------------
# OpenAI response builders
# ---------------------------------------------------------------------------


def build_openai_response(
    *,
    token_ids: list[int],
    parsed: ParsedCompletion,
    model: str,
    prompt_tokens: int = 0,
) -> dict[str, Any]:
    """Format the cached completion without decoding or resolving a model."""
    message: dict[str, Any] = {"role": "assistant", "content": parsed.message_content}
    if parsed.tool_calls:
        message["tool_calls"] = parsed.tool_calls
    if parsed.reasoning_content is not None:
        message["reasoning_content"] = parsed.reasoning_content

    return {
        "id": f"chatcmpl-{uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": parsed.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(token_ids),
            "total_tokens": prompt_tokens + len(token_ids),
        },
    }


def build_openai_stream_response(response_body: dict[str, Any]) -> StreamingResponse:
    """Replay a completed, already recorded chat completion as an OpenAI SSE
    stream: ``stream: true`` is honored in format only."""
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
    """Build a server completion callback around a rollout provider.

    Generates strictly token-in-token-out, parses tool calls, records the turn,
    and formats the response. TITO violations map to 400, unknown models to
    404, upstream 4xx are preserved, upstream 5xx map to 502, and other
    failures to 500. Pass it as ``LLMProxyServer(completion_handler=...)``;
    calling it directly bypasses the server's lifecycle protections.
    """
    async def _run_completion(
        proxy: "LLMProxyServer",
        session_id: str,
        agent_id: str,
        model: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
        is_streaming: bool,
    ):
        stream = stream_id(session_id, agent_id)
        # Only a continuation extends the recorded routing blob; a new
        # conversation starts its capture from scratch.
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
                    # Omitted/null sampling values let the provider apply defaults.
                    temperature=body.get("temperature"),
                    top_p=body.get("top_p"),
                    top_k=body.get("top_k"),
                    min_p=body.get("min_p"),
                    max_tokens=_max_tokens_param(body),
                    stop=body.get("stop"),
                    tools=body.get("tools"),
                    repetition_penalty=body.get("repetition_penalty"),
                    presence_penalty=body.get("presence_penalty"),
                    frequency_penalty=body.get("frequency_penalty"),
                    **({"reasoning_effort": body["reasoning_effort"]} if body.get("reasoning_effort") is not None else {}),
                    **{key: body[key] for key in ("seed", "stop_token_ids") if body.get(key) is not None},
                )
            )
        except UnknownModelError as e:
            logger.error("Unknown model for stream %s: %s", stream, e)
            return JSONResponse(status_code=404, content={"error": str(e)})
        except TokenStreamError as e:
            logger.error("Strict TITO violation for stream %s: %s", stream, e)
            return JSONResponse(status_code=400, content={"error": str(e)})
        except httpx.HTTPStatusError as e:
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

        # Generation has committed the turn. Parsing and recording must share
        # the failure guard: either can leave the record missing sampled tokens.
        try:
            parsed = await provider.parse_completion(
                stream, token_ids, completion_text, finish_reason,
                tools=body.get("tools"),
            )
            await asyncio.to_thread(
                proxy.recorder.record_completion,
                session_id=session_id,
                agent_id=agent_id,
                messages=messages,
                completion_text=completion_text,
                token_ids=token_ids,
                logprobs=log_probs,
                finish_reason=parsed.finish_reason,
                prompt_token_ids=prompt_ids,
                content=parsed.message_content,
                reasoning_content=parsed.reasoning_content,
                tool_calls=parsed.tool_calls,
                weight_version=engine_meta.get("weight_version"),
                sampling_params=engine_meta.get("sampling_params"),
                routed_experts=engine_meta.get("routed_experts"),
                rollout_top_p_token_ids=engine_meta.get("rollout_top_p_token_ids"),
                rollout_top_p_token_offsets=engine_meta.get("rollout_top_p_token_offsets"),
            )
        except Exception as e:
            # The turn is committed; block collection until an exact retry
            # records (or verifies) it.
            logger.error(
                "Parsing or recording failed for session %s (agent %s) after the turn "
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
            parsed=parsed,
            prompt_tokens=len(prompt_ids),
            model=model,
        )
        if is_streaming:
            return build_openai_stream_response(response_body)
        return JSONResponse(content=response_body)

    return _run_completion


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
        save_rollout_sampling_masks: bool = False,
    ):
        """
        Args:
            host: Bind address.
            port: Port number; ``0`` auto-selects a free port.
            completion_handler: Required; build it with :func:`make_completion_handler`.
            on_session_deleted: Called once per stream id when a session is deleted.
            get_sampling_overrides: Backs ``GET /sampling_overrides``.
            set_sampling_overrides: Backs ``PUT /sampling_overrides``; ``ValueError`` becomes 400.
            get_routed_experts_config: Backs ``GET /routed_experts``.
            set_routed_experts_config: Backs ``PUT /routed_experts``; ``ValueError`` becomes 400.
            get_tokenizer_fingerprint: Backs ``GET /tokenizer_fingerprint``;
                ``UnknownModelError`` becomes 404.
            api_key: Real key for keyed routing; without it every completion is 401.
            error_log_dir: Root for per-run error logs (default under
                ``PROXY_DATA_DIR``).
            save_rollout_sessions: Persist session snapshots to disk.
            save_rollout_logprobs: Include logprobs in disk snapshots.
            save_rollout_routed_experts: Include routing blobs in disk snapshots.
            session_dir: Root for session snapshots (default under
                ``PROXY_DATA_DIR``).
            key_delimiter: Separator inside keyed API keys (default ``"|"``).
            save_rollout_sampling_masks: Include sampling support in disk snapshots.

        Control endpoints answer 501 when their callback is omitted. The
        in-memory record served by ``GET /sessions/{id}`` always carries every
        field, whatever the disk snapshot flags say.
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
        self.instance_id = uuid4().hex

        if not callable(completion_handler):
            raise ValueError("completion_handler is required; build it with make_completion_handler(provider)")
        self._completion_handler = completion_handler
        self.error_log = SessionErrorLogger(
            Path(error_log_dir).expanduser().resolve() if error_log_dir is not None else storage_path("logs"),
        )
        self.session_store = (
            SessionStore(
                Path(session_dir).expanduser().resolve() if session_dir is not None else storage_path("sessions"),
            )
            if save_rollout_sessions
            else None
        )
        self.recorder = SessionRecorder(
            store=self.session_store,
            save_logprobs=save_rollout_logprobs,
            save_routed_experts=save_rollout_routed_experts,
            save_sampling_masks=save_rollout_sampling_masks,
        )
        # Keep detached completion pipelines alive until they finish recording.
        self._orphan_tasks: set[asyncio.Task] = set()

        self._turn_locks: dict[str, asyncio.Lock] = {}  # per stream, see _run_turn
        self._session_streams: dict[str, set[str]] = {}
        # Includes requests queued on turn locks and shielded completion pipelines.
        self._session_tasks: dict[str, set[asyncio.Task]] = {}
        self._deletion_tasks: dict[str, asyncio.Task] = {}
        # Retain outcomes until deletion: success and failure both seal admission.
        self._finalization_tasks: dict[str, asyncio.Task] = {}
        self._session_work_changed: dict[str, asyncio.Event] = {}

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
        occurrences. An optional instance ID follows the real key; stale
        bound keys raise HTTP 412 before opening a session. Returns ``None``
        when keyed routing is disabled, the real-key part does not match,
        or an id is missing or contains unsafe characters.
        """
        if not self._api_key or not token:
            return None
        head, sep, agent_id = token.rpartition(self.key_delimiter)
        if not sep:
            return None
        real_key, sep2, session_id = head.rpartition(self.key_delimiter)
        if real_key != self._api_key:
            base_key, bound, instance_id = real_key.rpartition(self.key_delimiter)
            if not bound or base_key != self._api_key:
                return None
            if instance_id != self.instance_id:
                raise HTTPException(status_code=412, detail=PROXY_RESTART_MESSAGE)
        if (
            not sep2
            or not _SESSION_ID_RE.match(session_id)
            or not _SESSION_ID_RE.match(agent_id)
        ):
            return None
        return session_id, agent_id

    def _turn_lock(self, stream: str) -> asyncio.Lock:
        """The stream's turn-serialization lock (created on first use)."""
        lock = self._turn_locks.get(stream)
        if lock is None:
            lock = self._turn_locks.setdefault(stream, asyncio.Lock())
        return lock

    def _track_stream(self, session_id: str, stream: str) -> None:
        """Remember a session's streams so deletion can release them all."""
        self._session_streams.setdefault(session_id, set()).add(stream)

    def _track_session_task(self, session_id: str, task: asyncio.Task | None) -> Callable[[], None]:
        if task is None:
            raise RuntimeError("Session work must run in an asyncio task")
        tasks = self._session_tasks.setdefault(session_id, set())
        tasks.add(task)

        def notify() -> None:
            changed = self._session_work_changed.get(session_id)
            if changed is not None:
                changed.set()

        notify()

        def done(finished: asyncio.Task) -> None:
            tasks.discard(finished)
            notify()
            if not tasks and self._session_tasks.get(session_id) is tasks:
                self._session_tasks.pop(session_id, None)

        task.add_done_callback(done)

        def untrack() -> None:
            task.remove_done_callback(done)
            done(task)

        return untrack

    async def _run_turn(
        self,
        session_id: str,
        agent_id: str,
        model: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
        is_streaming: bool,
    ) -> Any:
        """Run one turn through the completion handler, one turn per stream at a time.

        A racing SDK retry waits for the original to commit and record, then
        is re-served as an exact repeat. A caller that gives up mid-turn does
        not abort it: the shielded turn finishes detached and is recorded, so
        turns longer than the agent's timeout can still complete.
        """
        lock = self._turn_lock(stream_id(session_id, agent_id))
        await lock.acquire()
        task = asyncio.create_task(self._call_handler(session_id, agent_id, model, messages, body, is_streaming))
        # A task cancelled before its first step never runs a ``finally``
        # block; a done-callback releases the lock however the task ends.
        task.add_done_callback(lambda _: lock.release())
        self._track_session_task(session_id, task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled() or self.recorder.is_deleted(session_id):
                raise  # the turn itself was torn down; nothing to preserve
            self.error_log.log(
                session_id,
                f"agent {agent_id}: "
                "caller gave up mid-turn (timeout or disconnect); finishing "
                "the turn detached and recording it (an exact retry of the "
                "request is re-served this turn)",
            )
            self._orphan_tasks.add(task)
            task.add_done_callback(self._reap_orphan_task)
            raise

    async def _call_handler(
        self,
        session_id: str,
        agent_id: str,
        model: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
        is_streaming: bool,
    ) -> Any:
        # Deleted while this request was queued on the turn lock.
        if self.recorder.is_deleted(session_id):
            return JSONResponse(status_code=410, content={"error": f"session {session_id} was deleted"})
        return await self._completion_handler(self, session_id, agent_id, model, messages, body, is_streaming)

    def _reap_orphan_task(self, task: asyncio.Task) -> None:
        """Drop a detached turn's reference and log its failure, if any."""
        self._orphan_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Detached turn-preserving task failed: %s", task.exception())

    async def finalize_session(self, session_id: str) -> None:
        """Seal admission and drain accepted work through recording.

        A disconnected collector does not cancel the barrier. Failed barriers
        stay sealed; recover using a fresh trial/session, or delete to abandon.
        This guarantees an in-memory recording, not durable storage or a
        successful/training-valid trial.
        """
        if self.recorder.is_deleted(session_id):
            raise HTTPException(410, "Session was deleted")
        task = self._finalization_tasks.get(session_id)
        if task is None:
            if self.recorder.get_session(session_id) is None:
                raise HTTPException(404, "Session not found")
            task = asyncio.create_task(self._finalize_session(session_id))
            self._finalization_tasks[session_id] = task
            # Retrieve failures even if every waiting HTTP caller disconnects.
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if self.recorder.is_deleted(session_id):
                raise HTTPException(410, "Session was deleted") from None
            raise
        except RuntimeError as exc:
            if self.recorder.is_deleted(session_id):
                raise HTTPException(410, "Session was deleted") from exc
            raise HTTPException(500, str(exc)) from exc
        if self.recorder.is_deleted(session_id):
            raise HTTPException(410, "Session was deleted")

    async def _finalize_session(self, session_id: str) -> None:
        changed = asyncio.Event()
        self._session_work_changed[session_id] = changed
        try:
            while True:
                changed.clear()
                if not any(not task.done() for task in self._session_tasks.get(session_id, ())):
                    break
                # Wait on tracking changes, not ASGI tasks: endpoints untrack before
                # sending responses to slow clients. Re-read to include work queued since.
                await changed.wait()
            await asyncio.to_thread(self.recorder.finalize_session, session_id)
        finally:
            self._session_work_changed.pop(session_id, None)

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
        finalization = self._finalization_tasks.get(session_id)
        if finalization is not None:
            tasks.add(finalization)
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
                self._turn_locks.pop(stream, None)
            except Exception as e:
                errors.append((stream, e))
        if errors:
            raise RuntimeError(f"Failed to release streams: {errors}") from errors[0][1]
        self._session_streams.pop(session_id, None)
        self._finalization_tasks.pop(session_id, None)

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
            # Let detached completion pipelines whose callers gave up finish
            # before returning. Also finish deletions whose callers disconnected,
            # so provider resources are not closed beneath their cleanup.
            pending = self._orphan_tasks | set(self._deletion_tasks.values()) | set(self._finalization_tasks.values())
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            logger.info("LLM Proxy stopped")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
