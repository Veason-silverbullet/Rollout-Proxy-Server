"""WebSocket-based inference relay for standalone proxy deployment.

When the proxy is deployed as a standalone service, it uses an
``InferenceRelay`` to forward completion requests to connected
framework-side workers over WebSocket.

Protocol
~~~~~~~~

All messages are JSON objects with a ``type`` field.

**Worker → Proxy** (once, on connect) and **Proxy → Worker** (the reply)::

    {"type": "worker_hello", "worker_id": "<stable id>"}
    {"type": "worker_hello_ack", "worker_id": "<stable id>", "status": "connected"}

``worker_id`` must be a non-empty string, and must be *stable across
reconnects*: sessions are bound to it and their token streams live only on
that worker (see the dispatch note below).  A blank or non-string id is
rejected with close code 4001; an absent one is assigned by the proxy,
which forfeits reconnect stickiness.

**Proxy → Worker** (pushed when an agent calls the proxy)::

    {
        "type": "completion_request",
        "request_id": "<uuid>",
        "session_id": "<trial-id>",
        "model": "<model name>",
        "messages": [...],
        "temperature": null,
        "top_p": null,
        "top_k": null,
        "min_p": null,
        "max_tokens": 4096,
        "stop": null,
        "tools": null,
        "repetition_penalty": null,
        "presence_penalty": null,
        "frequency_penalty": null
    }

``model`` is the request's model claim, already validated by the proxy
against its tokenizer mapping; the worker resolves the tokenizer from it
and pins it to the session.  Sampling knobs are ``null`` when the agent's
request omitted them — the worker-side provider fills those from the
model's bundled ``generation_config.json`` defaults.

The relay has no incremental reply: the worker always answers with the
finished turn, and for ``stream: true`` requests the proxy replays that
completed response as SSE chunks (see ``server.build_openai_stream_response``).

**Worker → Proxy** (after inference completes)::

    {
        "type": "completion_response",
        "request_id": "<uuid>",
        "prompt_token_ids": [...],
        "token_ids": [...],
        "log_probs": [...],
        "completion_text": "...",
        "content_text": "...",
        "tool_calls": [...] | null,
        "finish_reason": "stop" | "tool_calls" | "length",
        "weight_version": "<engine-reported policy version>" | null,
        "error": null,
        "error_type": null
    }

``error_type`` is the failure discriminator: a worker sets it — to
``"invalid_request"`` (the request's own fault; the proxy answers 400) or
``"internal"`` (the proxy answers 503) — if and only if the turn failed.
It is *not* inferred from ``error``, because several failures a worker hits
routinely stringify to ``""`` (every httpx timeout among them), and a blank
message read as "no error" would deliver the failure payload as a
successful empty turn.  A failed response must still carry a non-empty
``error`` describing the fault; a third-party worker that sets only
``error`` is still classified as failed, for compatibility.

**Proxy → Worker** (when a session is deleted; lets workers drop
per-session state such as strict-TITO token streams)::

    {"type": "session_closed", "session_id": "<trial-id>"}

Keep-alive uses WebSocket protocol-level pings (sent by the worker's
``websockets`` library, answered automatically by the server) — there are
no application-level heartbeat messages.

Dispatch is sticky per ``session_id``: all turns of a session go to the
same worker (required for strict token-in-token-out prompts, whose state
is worker-local).  A session is never re-bound — if its worker does not
reconnect within a grace period, its requests fail rather than silently
re-tokenizing history on another worker.  The ``session_id`` on the wire
is really a **stream id**: for one agent of a multi-agent rollout the
proxy composes ``{session_id}{PROXY_KEY_DELIMITER}{agent_id}`` (see ``server.stream_id``),
so each agent's stream binds, dispatches, and is released independently —
the relay and the workers need no other knowledge of agents.

A completed turn outliving its caller is **not** dropped: rollout turns are
expensive, and once the worker commits one to the session's token stream it
must reach the session record.  Requests whose caller timed out (or was
cancelled) stay recordable for a grace window, and a late
``completion_response`` is handed to :attr:`InferenceRelay.on_orphan_response`
(the proxy records the turn; the agent saw a 503 and its exact retry is
re-served the same turn by the worker).  Workers cooperate by buffering
responses whose send failed mid-disconnect and re-sending them after
reconnecting — which is also why a disconnected worker's in-flight requests
stay pending rather than failing immediately: a transient drop resolves
transparently, and the dispatch timeout is the backstop.
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Callable
from uuid import uuid4
from fastapi import WebSocket, WebSocketDisconnect
from .engines import DEFAULT_MAX_TOKENS

logger = logging.getLogger(__name__)


def _is_error_response(response: dict[str, Any]) -> bool:
    """Whether a ``completion_response`` reports a failure.

    Classified structurally, on the presence of ``error_type`` (which a
    worker sets only when it failed) rather than on the truthiness of the
    ``error`` message.  Several exceptions a worker hits most — every httpx
    timeout, for one — stringify to ``""``, and a blank message read as
    "no error" would deliver the failure payload as a *successful* turn:
    HTTP 200 with empty content, no tool calls, and a zero-token completion
    recorded into the training data, which resets the session's delta
    baseline and buries earlier sampled tokens in a later turn's prompt.

    ``error`` alone still counts, so a third-party worker predating
    ``error_type`` keeps working.
    """
    return bool(response.get("error_type") or response.get("error"))


def _error_message(response: dict[str, Any]) -> str:
    """A never-empty description of a failed ``completion_response``."""
    return (
        str(response.get("error") or "").strip()
        or f"worker reported an unexplained {response.get('error_type') or 'unspecified'} failure"
    )


class WorkerError(RuntimeError):
    """Error reported by a worker for a specific request.

    ``error_type == "invalid_request"`` means the request itself was at
    fault (e.g. a strict token-in-token-out violation) and maps to HTTP
    400; ``"internal"`` means the worker failed and maps to HTTP 503.
    """

    def __init__(self, message: str, error_type: str = "internal") -> None:
        super().__init__(message)
        self.error_type = error_type


class _PendingRequest:
    """A pending inference request awaiting a worker response.

    Carries enough of the request (session, model, messages) to record a
    completion that arrives after its caller gave up — see
    :meth:`InferenceRelay.deliver_response`.
    """

    __slots__ = ("future", "worker_id", "session_id", "model", "messages", "expired_at")

    def __init__(
        self,
        worker_id: str,
        session_id: str,
        model: str,
        messages: list[dict[str, Any]],
    ) -> None:
        self.future: asyncio.Future = asyncio.get_running_loop().create_future()
        self.worker_id = worker_id
        self.session_id = session_id
        self.model = model
        self.messages = messages
        #: Monotonic time the caller gave up (timeout/cancel) and the entry
        #: moved to the expired table; ``None`` while the caller still waits.
        self.expired_at: float | None = None


class InferenceRelay:
    """Manages WebSocket connections to framework-side inference workers.

    The relay maintains a pool of connected workers and dispatches incoming
    inference requests (from agent HTTP calls) to available workers in a
    round-robin fashion.  Responses are correlated back via ``request_id``.

    Thread-safety: all public methods are coroutine-safe and use asyncio
    primitives.  The relay is intended to be used from a single event loop
    (the FastAPI/Uvicorn loop).
    """

    def __init__(self, request_timeout: float = 600.0) -> None:
        """
        Args:
            request_timeout: Maximum seconds to wait for a worker response
                before timing out an inference request.  The default aligns
                with the OpenAI SDK's 600s client default; raise both
                together when agents allow longer turns.  Timing out does
                not lose the turn: the request parks in the expired table
                and a late worker response is still recorded (see
                :meth:`deliver_response`).
        """
        self.request_timeout = request_timeout

        # worker_id -> WebSocket
        self._workers: dict[str, WebSocket] = {}
        # Round-robin index
        self._rr_index: int = 0
        # session_id -> worker_id (sticky bindings for strict TITO)
        self._session_workers: dict[str, str] = {}

        # request_id -> _PendingRequest
        self._pending: dict[str, _PendingRequest] = {}
        # request_id -> _PendingRequest whose caller gave up (timeout or
        # cancel): kept for a grace window so a late worker response is
        # recorded — each turn is an expensive rollout step — not dropped.
        self._expired: dict[str, _PendingRequest] = {}
        # A late completion for an expired request is handed here; the proxy
        # server installs a callback that records the turn.
        self.on_orphan_response: Callable[[_PendingRequest, dict[str, Any]], None] | None = None
        # Expired entries are kept at least this long past their timeout.
        # The engine-side HTTP timeouts are 900s, so anything the engine
        # will ever answer arrives within that of the dispatch timeout.
        self._orphan_grace = max(900.0, request_timeout)

        # Event that is set whenever at least one worker is connected.
        self._worker_available = asyncio.Event()

        # Strong references to fire-and-forget notification tasks (e.g.
        # session_closed sends), so the event loop cannot garbage-collect
        # them mid-flight before they are delivered.
        self._background_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Worker connection management
    # ------------------------------------------------------------------

    @property
    def num_workers(self) -> int:
        return len(self._workers)

    def register_worker(self, worker_id: str, ws: WebSocket) -> None:
        """Register a newly connected worker."""
        self._workers[worker_id] = ws
        self._worker_available.set()
        logger.info("Worker %s connected (total: %d)", worker_id, len(self._workers))

    def unregister_worker(self, worker_id: str, ws: WebSocket) -> None:
        """Remove a disconnected worker; its in-flight requests stay pending.

        Sticky session bindings to this worker are deliberately kept: the
        session's token stream lives only on that worker, so re-binding a
        session elsewhere would force re-tokenizing history and break
        token-in-token-out.  If the worker reconnects under the same
        ``worker_id`` (its in-memory state intact) its sessions resume;
        otherwise their requests fail in :meth:`dispatch`.

        In-flight requests are kept pending too, not failed: the worker may
        be mid-generation of an expensive turn, and worker clients re-send
        responses buffered while disconnected
        (``InferenceWorkerClient._deliver_response``), so a transient drop
        resolves transparently — the caller never notices.  If the worker
        never returns, each request fails through its own dispatch timeout,
        which also parks it in the expired table so a later delivery is
        still recorded.

        ``ws`` identifies the connection being torn down.  Workers reconnect
        under a stable ``worker_id``, so a reconnect that lands before the old
        connection's receive loop notices the drop leaves the *new* socket in
        ``_workers`` under that id.  Unregistering by id alone would then evict
        the live connection — and, because it is still looping and will never
        re-register, strand the relay with zero workers.  So a teardown whose
        socket is no longer the registered one is a no-op.
        """
        if self._workers.get(worker_id) is not ws:
            logger.info(
                "Ignoring teardown of superseded connection for worker %s; "
                "a newer connection holds this worker_id", worker_id,
            )
            return

        self._workers.pop(worker_id, None)
        if not self._workers:
            self._worker_available.clear()

        in_flight = sum(1 for pr in self._pending.values() if pr.worker_id == worker_id)
        logger.info(
            "Worker %s disconnected (total: %d, in-flight kept pending: %d)",
            worker_id, len(self._workers), in_flight,
        )

    # ------------------------------------------------------------------
    # Request dispatching
    # ------------------------------------------------------------------

    def _bound_socket(self, session_id: str) -> tuple[str, WebSocket, bool] | None:
        """The session's sticky worker and its live socket, if both exist.

        Shaped as an :meth:`_acquire_worker` result (``bound_here=False`` —
        the binding predates this call) so both of its lookup sites can
        return it directly.
        """
        bound = self._session_workers.get(session_id)
        ws = self._workers.get(bound) if bound is not None else None
        return (bound, ws, False) if ws is not None else None

    async def _acquire_worker(self, session_id: str) -> tuple[str, WebSocket, bool]:
        """Resolve the worker for a session — sticky, never re-bound.

        Returns ``(worker_id, ws, bound_here)``, where ``bound_here`` says
        this call created the sticky binding.  The caller needs that to roll
        the binding back if its send fails; sampling "was it bound?" before
        this coroutine would be wrong, because it awaits, and a concurrent
        dispatch for the same session can bind it in between — the rollback
        would then delete a binding another request is relying on.

        Strict TITO state (the session's token stream) is worker-local, so
        every turn of a session must reach the same worker.  A session is
        bound round-robin on its first request and stays bound; if its
        worker is offline it gets a grace period to reconnect (worker
        clients auto-reconnect under a stable ``worker_id``), after which
        the request fails rather than re-binding.
        """
        if session_id in self._session_workers:
            bound = self._session_workers[session_id]
            deadline = time.monotonic() + min(self.request_timeout, 30.0)
            while True:
                already = self._bound_socket(session_id)
                if already is not None:
                    return already
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Worker {bound} bound to session {session_id} did "
                        f"not reconnect; refusing to re-bind the session — "
                        f"its token stream is worker-local and re-binding "
                        f"would break token-in-token-out"
                    )
                await asyncio.sleep(0.5)

        deadline = time.monotonic() + min(self.request_timeout, 30.0)
        while not self._workers:
            # The wake-up can be stale: the worker that set the event may be
            # gone again before this coroutine resumes (Event.clear() does
            # not retract already-woken waiters), so re-arm the event and
            # keep waiting instead of indexing into an empty pool.
            self._worker_available.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "No inference workers connected to the proxy. "
                    "Ensure a framework-side worker is running and "
                    "connected via WebSocket."
                )
            try:
                await asyncio.wait_for(self._worker_available.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass

        # Another dispatch for this session may have bound it while we waited
        # above; honor that binding rather than round-robining past it.
        already = self._bound_socket(session_id)
        if already is not None:
            return already

        worker_ids = list(self._workers.keys())
        idx = self._rr_index % len(worker_ids)
        self._rr_index = idx + 1
        wid = worker_ids[idx]
        self._session_workers[session_id] = wid
        return wid, self._workers[wid], True

    def shutdown(self) -> None:
        """Fail every pending request — the proxy is stopping.

        Kept-pending requests otherwise ride out the full dispatch timeout,
        and uvicorn's graceful shutdown waits for their open HTTP requests —
        so without this, stopping a proxy with in-flight turns hangs for up
        to ``request_timeout`` seconds.
        """
        for request_id in list(self._pending):
            pending = self._pending.pop(request_id, None)
            if pending is not None and not pending.future.done():
                pending.future.set_exception(
                    RuntimeError("Proxy is shutting down; request abandoned")
                )

    def release_session(self, session_id: str) -> None:
        """Forget a session's sticky binding and tell workers to drop its
        per-session state.  Called when the proxy deletes a session."""
        self._session_workers.pop(session_id, None)
        msg = {"type": "session_closed", "session_id": session_id}
        for wid, ws in list(self._workers.items()):
            try:
                task = asyncio.get_running_loop().create_task(ws.send_json(msg))
            except RuntimeError:
                logger.warning(
                    "No running event loop; skipping session_closed "
                    "notification for %s", session_id,
                )
                break
            # Keep a strong reference until the send completes, else the
            # task can be GC'd before the worker is told to drop the session.
            self._background_tasks.add(task)
            task.add_done_callback(self._reap_background_task)

    def _reap_background_task(self, task: asyncio.Task) -> None:
        """Drop the strong reference and consume the outcome.

        Sending to a worker that disconnected moments ago is routine, and an
        unretrieved failure would otherwise surface much later as a
        garbage-collection-time "Task exception was never retrieved".
        """
        self._background_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            # Not cosmetic: that worker never dropped the session's token
            # stream, so it holds a slot until LRU eviction reclaims it —
            # which, at scale, starts killing live sessions instead.
            logger.warning(
                "session_closed notification failed (a worker may still hold "
                "the session's token stream): %s", task.exception(),
            )

    async def dispatch(
        self,
        *,
        session_id: str,
        model: str,
        messages: list[dict[str, Any]],
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
    ) -> dict[str, Any]:
        """Dispatch an inference request to a connected worker.

        Blocks (asynchronously) until the worker responds or the request
        times out.

        Returns:
            The worker's response dict containing ``prompt_token_ids``,
            ``token_ids``, ``log_probs``, ``completion_text``,
            ``content_text``, ``tool_calls``, ``finish_reason``,
            ``weight_version``.
        """
        worker_id, ws, bound_here = await self._acquire_worker(session_id)
        request_id = uuid4().hex

        pending = _PendingRequest(worker_id=worker_id, session_id=session_id, model=model, messages=messages)
        self._pending[request_id] = pending

        request_msg = {
            "type": "completion_request",
            "request_id": request_id,
            "session_id": session_id,
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "stop": stop,
            "tools": tools,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }

        try:
            await ws.send_json(request_msg)
        except Exception as e:
            self._pending.pop(request_id, None)
            # Roll back a binding *this* dispatch created.  The refusal to
            # re-bind exists to protect worker-local TITO state, but a request
            # that never reached the worker created none — and if that worker
            # is gone for good (a crash-restart gets a fresh worker_id), the
            # session would stay bound to it and 503 on every later turn.
            if bound_here:
                self._session_workers.pop(session_id, None)
            raise RuntimeError(f"Failed to send request to worker {worker_id}: {e}")

        try:
            result = await asyncio.wait_for(pending.future, timeout=self.request_timeout)
        except asyncio.TimeoutError:
            # The worker may still deliver: park the request in the expired
            # table so the (expensive) completion is recorded when it lands
            # — see deliver_response — instead of vanishing with the 503.
            self._expire(request_id, pending)
            raise RuntimeError(
                f"Inference request {request_id} timed out after "
                f"{self.request_timeout}s (worker: {worker_id})"
            )
        except asyncio.CancelledError:
            # The caller's task was torn down (e.g. a client disconnect):
            # keep the turn recordable exactly like a timeout.
            self._expire(request_id, pending)
            raise

        return result

    def _expire(self, request_id: str, pending: _PendingRequest) -> None:
        """Move a given-up request to the expired table (late deliveries are
        recorded from there) and prune entries past the grace window."""
        if self._pending.pop(request_id, None) is not None:
            pending.expired_at = time.monotonic()
            self._expired[request_id] = pending
        else:
            self._record_unclaimed(request_id, pending)
        self._prune_expired()

    def _record_unclaimed(self, request_id: str, pending: _PendingRequest) -> None:
        """Record a completion the caller never collected.

        Reached when the request is no longer pending because
        :meth:`deliver_response` already answered it, but the caller was torn
        down before consuming the result.  Nothing holds that turn now — it
        is in neither table — yet it is committed to the worker's token
        stream, so it is recorded here or lost from the session for good.
        """
        if (
            pending.future.cancelled()
            or not pending.future.done()
            or pending.future.exception() is not None
        ):
            return
        response = pending.future.result()
        if _is_error_response(response) or self.on_orphan_response is None:
            return
        logger.warning(
            "Recording completion for request %s (session %s): the worker "
            "answered but the caller was gone",
            request_id, pending.session_id,
        )
        self.on_orphan_response(pending, response)

    def _prune_expired(self) -> None:
        """Drop expired entries past the orphan grace window (their worker
        will never answer by then) and bound the table size."""
        if not self._expired:
            return
        cutoff = time.monotonic() - self._orphan_grace
        stale = [
            rid for rid, pr in self._expired.items()
            if pr.expired_at is not None and pr.expired_at < cutoff
        ]
        for rid in stale:
            self._expired.pop(rid, None)
        while len(self._expired) > 4096:
            self._expired.pop(next(iter(self._expired)))

    # ------------------------------------------------------------------
    # Response handling (called from the WebSocket receive loop)
    # ------------------------------------------------------------------

    def deliver_response(self, request_id: str, response: dict[str, Any]) -> bool:
        """Deliver a worker's response to the waiting caller.

        A completion whose caller already gave up (timeout or cancel) is not
        discarded: the worker committed the turn to the session's token
        stream, and each turn is an expensive rollout step — so it is handed
        to :attr:`on_orphan_response` for recording.  The agent saw a 503,
        and its exact retry is re-served the same turn by the worker; the
        recorder deduplicates the double delivery.

        Returns:
            True if the response reached the caller or the orphan recorder,
            False if nothing was waiting for it.
        """
        pending = self._pending.pop(request_id, None)
        if pending is None:
            pending = self._expired.pop(request_id, None)
        self._prune_expired()
        if pending is None:
            logger.warning("Received response for unknown request %s", request_id)
            return False

        if not pending.future.done():
            if _is_error_response(response):
                pending.future.set_exception(
                    WorkerError(
                        f"Worker error: {_error_message(response)}",
                        # An explicit null error_type must not read as
                        # "internal" (503, retryable) when the worker meant
                        # to reject the request.
                        error_type=response.get("error_type") or "internal",
                    )
                )
            else:
                pending.future.set_result(response)
            return True

        # The caller gave up but the worker answered anyway.  A worker error
        # committed nothing (the agent's retry regenerates cleanly), but a
        # completion is already part of the worker's token stream and must
        # reach the session record.
        if _is_error_response(response):
            logger.info(
                "Late error for given-up request %s (session %s): %s",
                request_id, pending.session_id, _error_message(response),
            )
            return True
        logger.warning(
            "Recording late completion for given-up request %s (session %s)",
            request_id, pending.session_id,
        )
        if self.on_orphan_response is not None:
            self.on_orphan_response(pending, response)
        return True

    # ------------------------------------------------------------------
    # WebSocket handler (to be mounted as a FastAPI endpoint)
    # ------------------------------------------------------------------

    async def handle_worker_ws(self, ws: WebSocket) -> None:
        """Handle an incoming worker WebSocket connection.

        This coroutine runs for the lifetime of the connection, reading
        messages and dispatching them to the appropriate handler.
        """
        await ws.accept()

        try:
            init_msg = await asyncio.wait_for(ws.receive_json(), timeout=10.0)
        except Exception as e:
            logger.error("Worker handshake failed: %s", e)
            try:
                await ws.close(code=4000, reason="Handshake timeout or invalid")
            except Exception:
                pass
            return

        if not isinstance(init_msg, dict) or init_msg.get("type") != "worker_hello":
            await ws.close(code=4001, reason="Expected worker_hello")
            return

        # A blank or non-string worker_id would register under "" or None, and
        # two such workers would collide on that one key — the second evicting
        # the first while sessions stay bound to the id, routing their
        # worker-local token streams to whichever socket happens to hold it.
        worker_id = init_msg.get("worker_id")
        if worker_id is None:
            worker_id = uuid4().hex
        elif not isinstance(worker_id, str) or not worker_id.strip():
            logger.error("Worker handshake carried an invalid worker_id: %r", worker_id)
            await ws.close(code=4001, reason="worker_hello needs a non-empty string worker_id")
            return
        else:
            worker_id = worker_id.strip()
        self.register_worker(worker_id, ws)

        try:
            # The ack must be sent inside the try: if the worker vanishes
            # mid-handshake, the finally below still unregisters it —
            # otherwise a ghost worker would stay in the pool forever,
            # absorbing round-robin dispatches on a dead socket.
            await ws.send_json({
                "type": "worker_hello_ack",
                "worker_id": worker_id,
                "status": "connected",
            })

            while True:
                data = await ws.receive_json()
                msg_type = data.get("type")

                if msg_type == "completion_response":
                    request_id = data.get("request_id")
                    if request_id:
                        self.deliver_response(request_id, data)
                    else:
                        logger.warning("Response without request_id: %s", data)

                else:
                    logger.warning("Unknown message type from worker: %s", msg_type)

        except WebSocketDisconnect as e:
            logger.info("Worker %s WebSocket disconnected (code=%s, reason=%r)", worker_id, e.code, e.reason)
        except Exception as e:
            logger.error("Worker %s WebSocket error: %s", worker_id, e)
        finally:
            self.unregister_worker(worker_id, ws)
