"""Framework-side WebSocket client that connects to a standalone proxy server
and services inference requests.

When the proxy is deployed as a standalone service, it cannot reach the
rollout engines directly.  Instead, the framework-side
``InferenceWorkerClient`` connects to the proxy's ``/ws/worker`` endpoint and:

1. Receives ``completion_request`` messages from the proxy.
2. Performs inference through the transport selected by ``transport_mode`` —
   verl (Ray actor handles via the load balancer), slime (HTTP via
   slime's sgl-router), or direct (HTTP straight to the vLLM/SGLang engines).
3. Sends ``completion_response`` messages back to the proxy.

The client supports **automatic reconnection** with configurable exponential
backoff, and can handle multiple concurrent inference requests over a single
WebSocket connection.  A completion whose send fails mid-disconnect is
buffered and re-sent after the next successful handshake: the turn is
already committed to the session's token stream and expensive to produce,
and the proxy keeps given-up requests recordable for a grace window, so a
late delivery still reaches the session record.

The tokenizer is not configured: each ``completion_request`` carries the
``model`` the proxy already validated, and the worker's rollout provider
resolves the tokenizer and tool-call parser from it via
``tokenization/mapping.json`` and pins it to the session (see
:mod:`proxyserver.rollout_provider`).

Usage::

    from proxyserver.worker_client import InferenceWorkerClient

    # verl (default): inference via the framework's Ray load balancer
    client = InferenceWorkerClient(
        proxy_ws_url="ws://proxy-host:9400/ws/worker",
        load_balancer=load_balancer,
    )
    # slime: inference via slime's sgl-router
    client = InferenceWorkerClient(
        proxy_ws_url="ws://proxy-host:9400/ws/worker",
        transport_mode="slime",
        router_url="http://<sglang-router-ip>:<port>",
    )
    # direct: inference straight against the engines
    client = InferenceWorkerClient(
        proxy_ws_url="ws://proxy-host:9400/ws/worker",
        transport_mode="direct",
        inference_engine_base_url=["http://<engine-ip>:<port>/v1"],
        inference_engine_api_key=["<key>"],
    )
    await client.start()   # Connects and starts serving requests
    # ...
    await client.stop()    # Graceful shutdown
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
from collections import OrderedDict
from typing import Any
from uuid import uuid4
from .engines import DEFAULT_MAX_TOKENS
from .model_registry import UnknownModelError, resolve_profile
from .rollout_provider import DirectRolloutProvider, RayRolloutProvider, SlimeRolloutProvider
from .token_stream import TokenStreamError

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("PROXY_LOGGING_LEVEL", "INFO"))

# Cap on completion_responses buffered while disconnected (each holds a full
# turn: token IDs, logprobs, text).  Overflow drops the oldest — by then its
# proxy-side grace window has almost certainly lapsed anyway.
_MAX_UNSENT = 1024


def _describe(e: BaseException) -> str:
    """A never-empty description of an exception.

    ``str(e)`` is empty for several failures a worker hits routinely —
    every ``httpx`` timeout among them — and the proxy must never receive a
    blank error message: an error is only distinguishable from a completion
    by what the payload carries, and a blank one used to be delivered as a
    successful empty turn.
    """
    return str(e).strip() or type(e).__name__


def _response_message(request_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Wrap a completion body in its ``completion_response`` envelope."""
    return {"type": "completion_response", "request_id": request_id, **body}


def _error_response(error: str, error_type: str) -> dict[str, Any]:
    """Build a failed completion_response payload."""
    return {
        "prompt_token_ids": [],
        "token_ids": [],
        "log_probs": [],
        "completion_text": "",
        "content_text": "",
        "tool_calls": None,
        "finish_reason": "error",
        "error": error,
        "error_type": error_type,
    }


def _encode_response(response: dict[str, Any], request_id: str) -> str:
    """Encode a ``completion_response``, falling back to a worker error.

    A payload that cannot be serialized at all (e.g. a transport handing
    back numpy scalars) must not be buffered for resend: it would fail to
    encode again on every reconnect, and since the flush only drops an entry
    after a successful send, it would wedge the worker in a reconnect loop
    that never reaches the receive loop and so never serves another request.
    Reporting the failure is recoverable; buffering it is not.
    """
    try:
        return json.dumps(response)
    except (TypeError, ValueError) as e:
        logger.error(
            "Response for request %s cannot be serialized (%s); reporting it "
            "to the proxy as a worker error", request_id, e,
        )
        return json.dumps(_response_message(request_id, _error_response(
            f"worker produced an unserializable response: {e}", error_type="internal",
        )))


class InferenceWorkerClient:
    """WebSocket client that connects to the proxy and handles inference.

    The client manages the WebSocket lifecycle including automatic
    reconnection with exponential backoff.  Each incoming
    ``completion_request`` is processed concurrently via an asyncio task.
    """

    def __init__(
        self,
        proxy_ws_url: str,
        load_balancer: Any = None,
        tool_parser_factory=None,
        worker_id: str | None = None,
        inference_engine: str | None = None,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 60.0,
        heartbeat_interval: float = 30.0,
        transport_mode: str = "verl",
        router_url: str | None = None,
        router_api_key: str | None = None,
        inference_engine_base_url: list[str] | str | None = None,
        inference_engine_api_key: list[str] | str | None = None,
    ) -> None:
        """
        Args:
            proxy_ws_url: WebSocket URL of the proxy, e.g.
                ``ws://proxy-host:9400/ws/worker``.
            load_balancer: verl only — Ray actor handle of the framework's
                request load balancer (e.g. verl's
                ``GlobalRequestLoadBalancer``).  Inference requests are
                routed via ``acquire_server`` / ``release_server`` on this
                actor.
            tool_parser_factory: Optional callable
                ``(tool_call_parser, tokenizer) -> tool_parser``, supplied by
                the training framework (e.g. verl's
                ``ToolParser.get_tool_parser``); it receives each model's
                ``tool_call_parser`` name from ``tokenization/mapping.json``.
                If not supplied, the built-in parsers
                (:mod:`proxyserver.tokenization.tool_parser`) serve the name.
            worker_id: Unique ID for this worker.  Auto-generated if None.
            inference_engine: Rollout backend — ``"vllm"`` or ``"sglang"``.
                Defaults to ``$INFERENCE_ENGINE``, then ``"vllm"`` (verl,
                direct); slime is always ``"sglang"``.
            reconnect_base_delay: Initial delay (seconds) before reconnecting.
            reconnect_max_delay: Maximum reconnection delay (seconds).
            heartbeat_interval: Seconds between WebSocket protocol-level
                keep-alive pings (handled by the ``websockets`` library;
                the proxy's server answers them automatically).
            transport_mode: ``"verl"`` (Ray load balancer), ``"slime"``
                (HTTP via slime's sgl-router), or ``"direct"`` (HTTP
                straight to the vLLM/SGLang engines).
            router_url: slime only — base URL of slime's ``sgl-router``,
                e.g. ``http://{sglang_router_ip}:{sglang_router_port}``.
            router_api_key: slime only — optional bearer token for the router.
            inference_engine_base_url: direct only — OpenAI base URLs of the
                engines, one per running instance.
            inference_engine_api_key: direct only — api_keys matching the
                base URLs (one per URL, or a single key broadcast to all).
        """
        # Build the provider eagerly so a misconfigured worker fails at
        # construction rather than on the first inference request, minutes
        # into a rollout — the providers validate their own wiring
        # (endpoint-list shape, engine names, slime's SGLang-only rule).
        if transport_mode == "verl":
            if load_balancer is None:
                raise ValueError("transport_mode='verl' requires a load_balancer actor handle")
            self._provider = RayRolloutProvider(
                load_balancer=load_balancer,
                tool_parser_factory=tool_parser_factory,
                inference_engine=inference_engine,
            )
        elif transport_mode == "slime":
            if not router_url:
                raise ValueError("transport_mode='slime' requires router_url (slime's sgl-router)")
            self._provider = SlimeRolloutProvider(
                router_url=router_url,
                tool_parser_factory=tool_parser_factory,
                inference_engine=inference_engine,
                api_key=router_api_key,
            )
        elif transport_mode == "direct":
            if not inference_engine_base_url:
                raise ValueError("transport_mode='direct' requires inference_engine_base_url (the engines' OpenAI base URLs)")
            self._provider = DirectRolloutProvider(
                base_urls=inference_engine_base_url,
                tool_parser_factory=tool_parser_factory,
                inference_engine=inference_engine,
                api_keys=inference_engine_api_key,
            )
        else:
            raise ValueError(f"Unknown transport_mode {transport_mode!r}; supported: verl, slime, direct")

        self.proxy_ws_url = proxy_ws_url
        self.worker_id = worker_id or f"worker-{uuid4().hex[:8]}"
        self.transport_mode = transport_mode
        self.inference_engine = self._provider.engine.name

        self.reconnect_base_delay = reconnect_base_delay
        self.reconnect_max_delay = reconnect_max_delay
        self.heartbeat_interval = heartbeat_interval

        self._ws = None
        self._running = False
        self._main_task: asyncio.Task | None = None
        self._handshake_ok = False
        # completion_responses whose send failed mid-disconnect, re-sent after
        # the next successful handshake.  Stored already encoded
        # (request_id -> JSON payload), so flushing can only ever fail on the
        # transport — never on the payload, which would be unrecoverable.
        self._unsent: OrderedDict[str, str] = OrderedDict()
        # In-flight inference tasks.  Tracked on the client, NOT per
        # connection: a connection loss must not wait for minutes-long
        # generations before reconnecting (the proxy keeps their requests
        # pending only until its dispatch timeout).  Tasks finishing while
        # disconnected buffer their responses; stop() drains the set.
        self._tasks: set[asyncio.Task] = set()

        logger.info(
            "Inference provider initialized (transport_mode=%s, inference_engine=%s)",
            self.transport_mode, self.inference_engine,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def _do_inference(self, request: dict[str, Any]) -> dict[str, Any]:
        """Perform inference through the framework's rollout servers."""
        messages = request.get("messages", [])
        session_id = request.get("session_id")
        model = request.get("model")
        max_tokens = request.get("max_tokens") or DEFAULT_MAX_TOKENS
        stop = request.get("stop")
        tools = request.get("tools")

        # Validate the model up front, as the proxy endpoint does.  This
        # worker has its own tokenization/mapping.json and its own tokenizer
        # directories, so deployment skew can leave it unable to serve a
        # model the proxy accepted — a condition retrying never fixes, and so
        # one that must be reported as invalid_request (400) rather than a
        # retryable 503.  Resolving here rather than catching
        # FileNotFoundError around the whole flow keeps unrelated filesystem
        # errors from the engine transport (a missing TLS cert or socket
        # path) retryable, which they are.
        try:
            resolve_profile(model)
        except (UnknownModelError, FileNotFoundError, ValueError) as e:
            logger.error("Cannot serve the requested model %r: %s", model, e)
            return _error_response(_describe(e), error_type="invalid_request")

        try:
            # Core generation via the framework transport (strict TITO
            # prompts; the model — already validated by the proxy — selects
            # the tokenizer and is pinned to the session).
            prompt_ids, token_ids, log_probs, finish_reason, completion_text, engine_meta = (
                await self._provider.generate(
                    messages=messages,
                    session_id=session_id,
                    model=model,
                    temperature=request.get("temperature"),
                    top_p=request.get("top_p"),
                    top_k=request.get("top_k"),
                    min_p=request.get("min_p"),
                    max_tokens=max_tokens,
                    stop=stop,
                    tools=tools,
                    repetition_penalty=request.get("repetition_penalty"),
                    presence_penalty=request.get("presence_penalty"),
                    frequency_penalty=request.get("frequency_penalty"),
                )
            )

            # Parse tool calls; the provider folds them into finish_reason
            # (truncation overrides a trailing, cut-off tool call).
            content_text, tool_calls_list, finish_reason = (
                await self._provider.parse_tool_calls(
                    session_id, token_ids, completion_text, finish_reason, tools=tools,
                )
            )

            # The relay protocol flattens engine_meta to the explicit keys
            # below, so meta the local slime path carries whole — e.g. the
            # routed_experts capture blob — does NOT survive relay mode.
            return {
                "prompt_token_ids": prompt_ids,
                "token_ids": token_ids,
                "log_probs": log_probs,
                "completion_text": completion_text,
                "content_text": content_text,
                "tool_calls": tool_calls_list,
                "finish_reason": finish_reason,
                "weight_version": engine_meta.get("weight_version"),
                "error": None,
            }

        except TokenStreamError as e:
            # The request violates strict token-in-token-out — the caller's
            # fault, not an inference failure.
            logger.error("Strict TITO violation: %s", e)
            return _error_response(_describe(e), error_type="invalid_request")
        except Exception as e:
            logger.error("Inference failed: %s", e, exc_info=True)
            return _error_response(_describe(e), error_type="internal")

    # ------------------------------------------------------------------
    # WebSocket communication
    # ------------------------------------------------------------------

    async def _handle_request(self, request: dict[str, Any]) -> None:
        """Process a single inference request and send the response."""
        request_id = request.get("request_id", "")
        logger.info("Processing inference request %s", request_id)

        result = await self._do_inference(request)
        await self._deliver_response(_response_message(request_id, result))

    async def _deliver_response(self, response: dict[str, Any]) -> None:
        """Send a completion_response, buffering it if the connection is down.

        An unsendable completion is exactly the lost-turn case: the sampled
        tokens are already committed to the session's token stream, so
        dropping the response would strand the session and waste the
        rollout.  The proxy keeps given-up requests recordable for a grace
        window and re-serves exact retries, so a buffered response re-sent
        after reconnect (see :meth:`_connect_and_serve`) still lands in the
        session record.

        Only *transport* failures buffer — the payload is encoded first (see
        :func:`_encode_response`), so what goes into the buffer can always
        be re-sent.
        """
        request_id = str(response.get("request_id", ""))
        payload = _encode_response(response, request_id)
        ws = self._ws
        try:
            if ws is None:
                raise ConnectionError("worker is not connected")
            await ws.send(payload)
        except Exception as e:
            logger.warning(
                "Buffering response for request %s until reconnect (%s)",
                request_id, _describe(e),
            )
            self._unsent[request_id] = payload
            while len(self._unsent) > _MAX_UNSENT:
                dropped, _ = self._unsent.popitem(last=False)
                logger.error("Unsent-response buffer full; dropping oldest (request %s)", dropped)

    async def _connect_and_serve(self) -> None:
        """Connect to the proxy and serve inference requests.

        This method handles the full lifecycle of a single WebSocket
        connection.  It returns when the connection is lost.
        """
        import websockets

        logger.info(
            "Connecting to proxy at %s (worker_id=%s)",
            self.proxy_ws_url, self.worker_id,
        )

        async with websockets.connect(
            self.proxy_ws_url,
            # Protocol-level keep-alive: the library pings on this interval
            # and the proxy's server answers automatically, so a dead
            # connection is detected without any application-level heartbeat.
            ping_interval=self.heartbeat_interval,
            max_size=100 * 1024 * 1024,  # 100 MB max message size
        ) as ws:
            self._ws = ws
            # The finally covers the handshake and flush too, not just the
            # receive loop: a failure in either leaves self._ws pointing at a
            # dead socket, and is_connected would report True for the whole
            # backoff and every later failed connect attempt.
            try:
                await ws.send(json.dumps({
                    "type": "worker_hello",
                    "worker_id": self.worker_id,
                }))

                ack_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                ack = json.loads(ack_raw)
                if ack.get("type") != "worker_hello_ack":
                    raise RuntimeError(f"Unexpected handshake response: {ack}")
                self._handshake_ok = True

                logger.info("Connected to proxy, worker %s acknowledged", self.worker_id)

                # Re-send responses buffered while disconnected, oldest first.
                # A send failure here leaves the rest buffered and falls through
                # to the reconnect loop.
                while self._unsent:
                    buffered_id, payload = next(iter(self._unsent.items()))
                    await ws.send(payload)
                    self._unsent.pop(buffered_id, None)
                    logger.info("Re-sent buffered response for request %s", buffered_id)

                # Main receive loop — handle messages concurrently.  In-flight
                # tasks are deliberately NOT awaited on connection loss:
                # blocking reconnection on minutes-long generations would hold
                # every kept-pending request at the proxy past its dispatch
                # timeout.  Tasks that finish while disconnected buffer their
                # responses (see _deliver_response) and the next handshake
                # flushes them; tasks that finish after the reconnect send on
                # the live socket directly.
                async for raw_msg in ws:
                    data = json.loads(raw_msg)
                    msg_type = data.get("type")

                    if msg_type == "completion_request":
                        task = asyncio.create_task(self._handle_request(data))
                        self._tasks.add(task)
                        task.add_done_callback(self._tasks.discard)

                    elif msg_type == "session_closed":
                        # Proxy deleted the session — drop per-session state
                        # (the model pin and strict-TITO token stream).
                        session_id = data.get("session_id")
                        if session_id:
                            self._provider.release_session(session_id)

                    else:
                        logger.warning("Unknown message type: %s", msg_type)

            finally:
                # Clear immediately: an abnormal close raises out of the
                # async-for above, a stale self._ws would keep is_connected
                # reporting True, and in-flight tasks must start buffering
                # (not send into a dead socket) while we reconnect.
                self._ws = None

    async def _run_with_reconnect(self) -> None:
        """Run the worker with automatic reconnection on disconnect."""
        consecutive_failures = 0

        while self._running:
            self._handshake_ok = False
            try:
                await self._connect_and_serve()
                cause = "Connection to proxy lost"
            except asyncio.CancelledError:
                break
            except Exception as e:
                cause = f"Worker connection error: {e}"

            if not self._running:
                break
            # A completed handshake resets the backoff: the proxy's sticky
            # sessions only wait a bounded grace period for their worker to
            # reconnect, so a transient drop after hours of healthy service
            # must retry from the base delay, not from an accumulated one.
            consecutive_failures = 1 if self._handshake_ok else consecutive_failures + 1
            delay = min(
                self.reconnect_base_delay * (2 ** (consecutive_failures - 1)),
                self.reconnect_max_delay,
            )
            logger.warning(
                "%s. Reconnecting in %.1fs (attempt %d)...",
                cause, delay, consecutive_failures,
            )
            await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the worker client.  Connects to the proxy and begins
        serving inference requests.

        This method returns immediately — the actual work happens in a
        background asyncio task.
        """
        if self._running:
            logger.warning("Worker client is already running")
            return

        self._running = True
        self._main_task = asyncio.create_task(self._run_with_reconnect())
        logger.info("Inference worker client started (id=%s)", self.worker_id)

    async def stop(self) -> None:
        """Gracefully stop the worker client."""
        self._running = False

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass

        # Cancel in-flight inference tasks — with the client stopped they
        # could never deliver their responses anyway.
        if self._tasks:
            logger.info("Cancelling %d in-flight inference tasks...", len(self._tasks))
            for task in list(self._tasks):
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)

        logger.info("Inference worker client stopped (id=%s)", self.worker_id)

    @property
    def is_connected(self) -> bool:
        """Whether the worker is currently connected to the proxy."""
        return self._ws is not None and self._running
