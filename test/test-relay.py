"""End-to-end test of the LLM proxy server in **relay mode**, strict TITO.

Topology (all in one process, but over real HTTP/WebSocket):

    50 mock agents               LLMProxyServer                MockInferenceWorker
    (OpenAI SDK)   --HTTP-->     (relay mode)    --WebSocket-->  (this file)
                                                                     |
                                                                     |  token-ID prompts
                                                                     v
                                                            real inference engine
                                                    (OPENAI_BASE_URL / OPENAI_API_KEY)

1. Start ``LLMProxyServer`` in relay mode (no completion handler).
2. Connect a mock inference worker to ``/ws/worker``.  It services
   ``completion_request`` messages **strictly token-in-token-out** using the
   package's :class:`TokenStreamManager`: turn N+1's prompt is turn N's
   prompt tokens + the sampled completion tokens + a tokenized delta of only
   the new messages.  Tokenization runs locally on the bundled tokenizer
   (never through the engine) and generation goes through
   ``/v1/completions`` with a **token-ID prompt** (``logprobs`` +
   ``return_tokens_as_token_ids`` bring the sampled token IDs / logprobs
   back, including the EOS token).
3. Run the mock agents concurrently.  Each holds a 2-turn conversation
   through the proxy via the standard OpenAI SDK using keyed routing —
   **no registration call**: ``base_url = {proxy}/v1`` and
   ``api_key = {REAL_API_KEY}{delimiter}{session_id}{delimiter}{agent_id}``;
   the proxy opens each session lazily on the first completion request.
4. Verify the relayed responses and the per-session records captured by
   the proxy's ``SessionRecorder`` — see ``common.run_agent`` for the
   strict-TITO criteria.  Tampered history must be rejected with a 400,
   never silently re-tokenized.

Run:
    python test/test-relay.py
"""

from __future__ import annotations
import asyncio
import json
import logging
from typing import Any
import httpx
import websockets
from common import (ADAPTER, ENDPOINT, ENGINE, KEY_DELIMITER, NUM_AGENTS, PROXY_API_KEY, MODEL_NAME,
                    load_bundled_tokenizer, make_rollout_server, make_verifier, run_gauntlet)
from proxyserver.engines import ABORT
from proxyserver.model_registry import build_tool_parser, resolve_model_spec
from proxyserver.server import LLMProxyServer
from proxyserver.token_stream import TokenStreamError, TokenStreamManager

logger = logging.getLogger("relay-test")

MAX_TOKENS = 8192


# ---------------------------------------------------------------------------
# Mock inference worker: proxy WebSocket <-> real inference engine (strict TITO)
# ---------------------------------------------------------------------------


class MockInferenceWorker:
    """Framework-side worker mock, mirroring ``InferenceWorkerClient``.

    Speaks the relay protocol (``worker_hello`` handshake, then
    ``completion_request`` / ``completion_response``) and fulfills each
    request strictly token-in-token-out: prompts are built by
    :class:`TokenStreamManager` and sent to the engine as token IDs —
    conversation history is never re-tokenized.

    Generation goes through the same rollout-server stand-in the worker test
    uses, and ``finish_reason`` is normalized with the same
    :class:`~proxyserver.engines.EngineAdapter` the real worker uses.  Relay
    mode therefore exercises the engine contract end-to-end too, rather than
    trusting whatever vocabulary the engine happens to emit.
    """

    def __init__(self, proxy_ws_url: str, model: str, worker_id: str = "mock-inference-worker"):
        self.proxy_ws_url = proxy_ws_url
        self.model = model
        self.worker_id = worker_id
        self.requests_served = 0
        self._server = make_rollout_server(ENDPOINT)
        self.tokenizer = load_bundled_tokenizer(model)
        self.token_streams = TokenStreamManager(self.tokenizer)
        # Clean agent-visible content exactly as the real worker's provider
        # does: the model's mapping-named parser (or NullToolParser) strips
        # special tokens and the leading reasoning block.
        self.tool_parser = build_tool_parser(
            resolve_model_spec(model).tool_call_parser, self.tokenizer
        )
        # session_id -> pinned model, mirroring the real worker's provider:
        # the relayed model claim is pinned on first use and a mid-session
        # switch is refused as a TITO violation.
        self.session_models: dict[str, str] = {}
        self._ws = None
        self._task: asyncio.Task | None = None
        self._connected = asyncio.Event()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._serve())
        await asyncio.wait_for(self._connected.wait(), timeout=10)

    async def stop(self) -> None:
        if self._ws is not None:
            await self._ws.close()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        await self._server.aclose()

    async def _serve(self) -> None:
        async with websockets.connect(self.proxy_ws_url, max_size=100 * 1024 * 1024) as ws:
            self._ws = ws
            await ws.send(json.dumps({"type": "worker_hello", "worker_id": self.worker_id}))
            ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            assert ack.get("type") == "worker_hello_ack", f"Bad handshake ack: {ack}"
            logger.info("Worker %s connected to proxy", self.worker_id)
            self._connected.set()

            tasks: set[asyncio.Task] = set()
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "completion_request":
                        task = asyncio.create_task(self._handle_request(msg))
                        tasks.add(task)
                        task.add_done_callback(tasks.discard)
                    elif msg.get("type") == "session_closed":
                        session_id = msg.get("session_id")
                        if session_id:
                            self.token_streams.drop(session_id)
                            self.session_models.pop(session_id, None)
                            logger.info("Worker: dropped token stream of %s", session_id)
            except websockets.ConnectionClosed:
                pass
            finally:
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                self._ws = None

    async def _handle_request(self, request: dict[str, Any]) -> None:
        request_id = request["request_id"]
        logger.info(
            "Worker: request %s (session=%s) -> %s",
            request_id[:8], request.get("session_id"), ENGINE,
        )
        try:
            result = await self._call_engine(request)
        except Exception as e:
            logger.error("Worker: %s call failed: %s", ENGINE, e)
            error_type = (
                "invalid_request" if isinstance(e, TokenStreamError) else "internal"
            )
            result = {
                "prompt_token_ids": [], "token_ids": [], "log_probs": [],
                "completion_text": "", "content_text": "", "tool_calls": None,
                "finish_reason": "error", "error": str(e),
                "error_type": error_type,
            }
        result["type"] = "completion_response"
        result["request_id"] = request_id
        if self._ws is not None:
            await self._ws.send(json.dumps(result))
        self.requests_served += 1

    async def _call_engine(self, request: dict[str, Any]) -> dict[str, Any]:
        session_id = request.get("session_id")
        messages = request.get("messages", [])

        # Pin the relayed model claim, as InferenceWorkerClient's provider
        # does — a mid-session switch would switch tokenizers mid-stream.
        model = request.get("model")
        pinned = self.session_models.get(session_id)
        if pinned is None:
            if not model:
                raise TokenStreamError(f"Session {session_id}: the first request must name a model")
            self.session_models[session_id] = model
        elif model and model != pinned:
            raise TokenStreamError(
                f"Session {session_id} is pinned to model {pinned!r} but the "
                f"request names {model!r}"
            )

        # Strict TITO: reuse the session's authoritative token stream and
        # tokenize only the new messages (locally, on the bundled tokenizer).
        # Raises TokenStreamError on any request that would require
        # re-tokenizing history; the error propagates to the agent.
        prompt_ids = self.token_streams.build_prompt(session_id, messages, tools=request.get("tools"))

        max_tokens = request.get("max_tokens", 2048)
        # The proxy always sends vLLM-style names; the SGLang rollout server
        # (e.g. verl's, and so
        # our stand-in) translates them into max_new_tokens/return_logprob.
        sampling_params: dict[str, Any] = {
            "temperature": request.get("temperature", 1.0),
            "top_p": request.get("top_p", 1.0),
            "max_tokens": max_tokens,
            "logprobs": True,
        }
        for key in ("stop", "repetition_penalty"):
            if request.get(key) is not None:
                sampling_params[key] = request[key]

        output = await self._server.generate(prompt_ids, sampling_params)

        # token-IDs out (includes the sampled EOS on a natural stop). Normalize
        # the engine's stop_reason exactly as InferenceWorkerClient does.
        token_ids, log_probs = output.token_ids, output.log_probs
        finish_reason = ADAPTER.finish_reason(
            output.stop_reason, num_tokens=len(token_ids), max_tokens=max_tokens
        )
        if finish_reason == ABORT:
            raise RuntimeError(f"{ENGINE} aborted the request for session {session_id}")
        if not token_ids:
            raise RuntimeError(f"{ENGINE} returned no tokens for session {session_id}")

        # The record keeps the raw decode; the agent sees the cleaned
        # content — the same split the real worker produces.
        raw_text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        # Pass `tools` exactly as the real worker does (worker_client
        # ._do_inference): the parser types each argument by its declared
        # schema, and a mock that omitted them would exercise only the
        # schema-less path and miss any regression in that typing.
        content_text, _calls = await self.tool_parser.extract_tool_calls(
            token_ids, tools=request.get("tools"),
        )

        self.token_streams.commit(session_id, messages, prompt_ids, token_ids)

        return {
            "prompt_token_ids": prompt_ids,
            "token_ids": token_ids,
            "log_probs": log_probs,
            "completion_text": raw_text,
            "content_text": content_text,
            "tool_calls": None,
            "finish_reason": finish_reason,
            "error": None,
        }


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


async def main() -> None:
    model = MODEL_NAME
    verifier = make_verifier(model)

    # relay_request_timeout matches the 900s the agents and the engine
    # HTTP clients already use — the default (600s, aligned with the OpenAI
    # SDK) is below them, and the shared engine can be slow under 50-way
    # load; the relay must not be the first layer to give up here.
    proxy = LLMProxyServer(host="127.0.0.1", port=0, api_key=PROXY_API_KEY,
                           key_delimiter=KEY_DELIMITER, relay_request_timeout=900)
    assert proxy.mode == "relay"
    proxy_url = await proxy.start()
    logger.info("Proxy started at %s (mode=%s)", proxy_url, proxy.mode)

    ws_url = proxy_url.replace("http://", "ws://") + "/ws/worker"
    worker = MockInferenceWorker(proxy_ws_url=ws_url, model=model)

    async with httpx.AsyncClient(timeout=30) as http:
        try:
            await worker.start()

            health = (await http.get(f"{proxy_url}/health")).json()
            assert health["mode"] == "relay", f"unexpected health: {health}"
            assert health["connected_workers"] == 1, f"unexpected health: {health}"

            # The shared gauntlet (common.run_gauntlet), plus the two
            # relay-specific seams injected below.
            def worker_served_all(records: list) -> None:
                assert worker.requests_served == NUM_AGENTS * 2, (
                    f"worker served {worker.requests_served} requests, "
                    f"expected {NUM_AGENTS * 2}"
                )

            def worker_dropped_stream(sid: str) -> None:
                assert sid not in worker.token_streams._sessions, \
                    f"worker did not drop the token stream of deleted session {sid}"

            await run_gauntlet(
                proxy_url, http,
                verifier=verifier, max_tokens=MAX_TOKENS,
                dump_dir=f"relay-{ENGINE}", proxy_api_key=PROXY_API_KEY,
                after_rollouts=worker_served_all,
                drop_check=worker_dropped_stream,
                drop_wait=0.2,
            )

        finally:
            await worker.stop()
            await proxy.stop()

    print("\n" + "=" * 70)
    print(f"PASS: relay mode strict TITO OK — {NUM_AGENTS} agents x 2 turns via")
    print(f"      proxy -> WebSocket worker -> {ENGINE} ({model}):")
    print("      token-ID prompts in, sampled token IDs/logprobs out, every")
    print("      turn's prompt strictly extends the session token stream,")
    print("      tampered history is rejected instead of re-tokenized,")
    print("      keyed routing (api_key-embedded session, no registration)")
    print("      rejects malformed/unauthorized keys with 401, and truncated")
    print(f"      responses are reported as finish_reason='length' ({ENGINE}).")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
