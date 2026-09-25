"""Offline e2e test of lost-response recovery in relay and local mode.

Rollout turns are expensive, so a completion that outlives some timeout in
the chain must neither vanish from the session record nor strand its
session.  This drives the real proxy stack — ``LLMProxyServer`` over real
HTTP, with a real ``InferenceWorkerClient`` over WebSocket in relay mode
and the injected local completion handler in local mode — and the real
``BaseRolloutProvider`` generate flow, with only the engine transport
scripted (canned tokens, controllable latency), and pins the recovery
paths end to end:

1. **Relay timeout** — the agent gets a 503, but the worker's late response
   is still recorded (orphan path), and an exact retry of the request is
   re-served the committed turn without a second generation.  The turn
   lands in the record exactly once.
2. **Retry racing its still-in-flight original** — a retry that lands while
   the timed-out original is *still generating* (the common shape when a
   turn outlives the aligned agent/relay timeouts) parks on the provider's
   per-session lock, is re-served the original's turn once it commits, and
   never triggers a second generation — which would fork the token stream.
3. **Worker disconnect mid-generation** — the proxy keeps the in-flight
   request pending, the worker buffers the response it could not send and
   re-sends it after reconnecting, and the *original* caller receives a
   normal 200: a transient drop is invisible to the agent.
4. **Concurrent duplicate** — an SDK-style automatic retry racing its
   original request serializes on the proxy's per-session lock and is then
   re-served, so the engine runs once and the record gains one turn.
5. **Local-mode client disconnect** — the agent times out and drops the
   connection mid-turn; the shielded pipeline finishes detached, the turn
   is committed and recorded with no retry involved, and the exact retry is
   re-served it.  (Whether a disconnect cancels the endpoint task is
   Starlette-version-dependent; the shield pins this outcome either way.)
6. **Session deleted mid-generation** — the cancelled-trial shape: the
   driver deletes the session while a turn is still in flight.  The late
   delivery is dropped (no retry is coming; recording it would re-create
   the record and worker-side token stream that nothing ever frees again)
   and straggler requests are refused with 410 instead of occupying an
   engine.

No inference engine and no ``transformers`` needed.

Run:
    python test/test-relay-recovery.py
"""

from __future__ import annotations
import asyncio
import sys
import time
from pathlib import Path
from typing import Any
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver.server import LLMProxyServer, make_local_completion_handler  # noqa: E402
from proxyserver.worker_client import InferenceWorkerClient  # noqa: E402
from offline_common import ScriptedProvider, check  # noqa: E402

# Any model in tokenization/mapping.json; the tokenizer itself is faked.
MODEL = "Qwen3.5-9B"
PROXY_API_KEY = "recovery-test-key"
DELIMITER = "-"
AGENT_ID = "agent"


class _Harness:
    """Shared agent-side helpers over a running proxy (``self.url``)."""

    url: str = ""

    async def chat(self, http: httpx.AsyncClient, session_id: str,
                   messages: list[dict[str, Any]], timeout: float = 30.0) -> httpx.Response:
        return await http.post(
            f"{self.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {PROXY_API_KEY}{DELIMITER}{session_id}{DELIMITER}{AGENT_ID}"},
            json={"model": MODEL, "messages": messages, "max_tokens": 32},
            timeout=timeout,
        )

    async def turns_recorded(self, http: httpx.AsyncClient, session_id: str) -> int:
        resp = await http.get(f"{self.url}/sessions/{session_id}")
        return len(resp.json()["turns"]) if resp.status_code == 200 else 0

    async def wait_turns(self, http: httpx.AsyncClient, session_id: str,
                         n: int, deadline_s: float) -> None:
        deadline = time.monotonic() + deadline_s
        while time.monotonic() < deadline:
            if await self.turns_recorded(http, session_id) >= n:
                return
            await asyncio.sleep(0.1)
        raise AssertionError(
            f"session {session_id}: expected {n} recorded turns within {deadline_s}s"
        )


class Stack(_Harness):
    """One relay-mode proxy + one real worker client on a scripted provider."""

    def __init__(self, relay_request_timeout: float, reconnect_delay: float = 0.2):
        self.provider = ScriptedProvider()
        self.proxy = LLMProxyServer(
            host="127.0.0.1", port=0, api_key=PROXY_API_KEY,
            key_delimiter=DELIMITER, relay_request_timeout=relay_request_timeout,
            save_rollout_sessions=False,
        )
        self._reconnect_delay = reconnect_delay
        self.worker: InferenceWorkerClient | None = None
        self.url = ""

    async def __aenter__(self) -> "Stack":
        self.url = await self.proxy.start()
        self.worker = InferenceWorkerClient(
            proxy_ws_url=self.url.replace("http://", "ws://") + "/ws/worker",
            transport_mode="verl", load_balancer=object(),  # replaced below
            worker_id="recovery-worker",
            reconnect_base_delay=self._reconnect_delay,
            reconnect_max_delay=self._reconnect_delay,
        )
        self.worker._provider = self.provider
        await self.worker.start()
        deadline = time.monotonic() + 10
        async with httpx.AsyncClient() as http:
            while time.monotonic() < deadline:
                health = (await http.get(f"{self.url}/health")).json()
                if health.get("connected_workers") == 1:
                    return self
                await asyncio.sleep(0.05)
        raise AssertionError("worker did not connect")

    async def __aexit__(self, *exc) -> None:
        await self.worker.stop()
        await self.proxy.stop()


class LocalStack(_Harness):
    """One local-mode proxy (injected completion handler) on a scripted provider."""

    def __init__(self):
        self.provider = ScriptedProvider()
        self.proxy = LLMProxyServer(
            host="127.0.0.1", port=0, api_key=PROXY_API_KEY,
            key_delimiter=DELIMITER,
            completion_handler=make_local_completion_handler(self.provider),
            on_session_deleted=self.provider.release_session,
            save_rollout_sessions=False,
        )
        self.url = ""

    async def __aenter__(self) -> "LocalStack":
        self.url = await self.proxy.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.proxy.stop()


TURN1 = [{"role": "user", "content": "hi"}]


def turn2(reply: str) -> list[dict[str, Any]]:
    return TURN1 + [{"role": "assistant", "content": reply},
                    {"role": "user", "content": "more"}]


async def test_relay_timeout_recovery() -> None:
    print("Relay timeout: late completion recorded, exact retry re-served")
    async with Stack(relay_request_timeout=1.0) as stack, httpx.AsyncClient() as http:
        sid = "trial_timeout"
        first = await stack.chat(http, sid, TURN1)
        assert first.status_code == 200, first.text
        reply = first.json()["choices"][0]["message"]["content"]

        # Turn 2 outlives the relay timeout: the agent sees a 503...
        stack.provider.delay = 2.5
        resp = await stack.chat(http, sid, turn2(reply))
        check("caller gets 503 when the relay times out", resp.status_code == 503)
        check("only the first turn is recorded so far", await stack.turns_recorded(http, sid) == 1)

        # ...but the worker finishes anyway and the late response is
        # recorded through the orphan path.
        await stack.wait_turns(http, sid, 2, deadline_s=5.0)
        check("late completion recorded after the 503 (orphan path)", True)

        # An exact retry is re-served the committed turn — no new generation.
        stack.provider.delay = 0.0
        engine_calls = stack.provider.engine_calls
        retry = await stack.chat(http, sid, turn2(reply))
        check("exact retry answers 200", retry.status_code == 200)
        check("retry did not trigger a second generation",
              stack.provider.engine_calls == engine_calls)
        check("retry content is the committed completion",
              retry.json()["choices"][0]["message"]["content"] == "<900><901><902>")
        check("the turn is recorded exactly once (dedup)",
              await stack.turns_recorded(http, sid) == 2)


async def test_relay_retry_races_inflight_original() -> None:
    print("\nRetry racing its timed-out, still-generating original serializes at the provider")
    async with Stack(relay_request_timeout=1.2) as stack, httpx.AsyncClient() as http:
        sid = "trial_inflight_retry"
        first = await stack.chat(http, sid, TURN1)
        assert first.status_code == 200, first.text
        reply = first.json()["choices"][0]["message"]["content"]

        # Turn 2 outlives the relay timeout; the agent sees a 503 while the
        # worker is STILL mid-generation...
        stack.provider.delay = 2.0
        engine_calls = stack.provider.engine_calls
        resp = await stack.chat(http, sid, turn2(reply))
        check("original gets 503 while its generation is still running",
              resp.status_code == 503)

        # ...and the SDK-style retry lands immediately, well before the
        # original commits.  The proxy's own session lock is already
        # released (the original's request returned 503), so only the
        # provider's per-session lock stands between the retry and a second
        # generation of the same turn: the retry must park there and be
        # re-served the original's completion once it commits.
        stack.provider.delay = 0.0  # a rogue second generation would be instant
        retry = await stack.chat(http, sid, turn2(reply))
        check("immediate retry answers 200", retry.status_code == 200)
        check("retry carries the original generation's completion",
              retry.json()["choices"][0]["message"]["content"] == "<900><901><902>")
        check("the turn was generated exactly once",
              stack.provider.engine_calls == engine_calls + 1)
        await stack.wait_turns(http, sid, 2, deadline_s=5.0)
        await asyncio.sleep(0.3)  # let the orphan delivery land too
        check("the turn is recorded exactly once (orphan + re-serve dedup)",
              await stack.turns_recorded(http, sid) == 2)


async def test_worker_disconnect_recovery() -> None:
    print("\nWorker disconnect mid-generation: buffered response reaches the waiting caller")
    async with Stack(relay_request_timeout=15.0, reconnect_delay=1.5) as stack, \
            httpx.AsyncClient() as http:
        sid = "trial_drop"
        first = await stack.chat(http, sid, TURN1)
        assert first.status_code == 200, first.text
        reply = first.json()["choices"][0]["message"]["content"]

        # Kick off turn 2 (slow), then kill the WebSocket mid-generation.
        stack.provider.delay = 1.0
        engine_calls = stack.provider.engine_calls
        request = asyncio.create_task(stack.chat(http, sid, turn2(reply)))
        await asyncio.sleep(0.3)
        await stack.worker._ws.close()

        # The worker finishes while disconnected, buffers the unsendable
        # response, reconnects, re-sends — and the caller, whose request
        # stayed pending across the drop, gets a normal 200.
        resp = await request
        check("original caller still receives 200 across the drop", resp.status_code == 200)
        check("response is the generated completion",
              resp.json()["choices"][0]["message"]["content"] == "<900><901><902>")
        check("generation ran exactly once", stack.provider.engine_calls == engine_calls + 1)
        check("unsent buffer drained after reconnect", not stack.worker._unsent)
        check("turn recorded exactly once", await stack.turns_recorded(http, sid) == 2)


async def test_concurrent_duplicate_serialized() -> None:
    print("\nConcurrent duplicate (SDK-style auto-retry) serializes and re-serves")
    async with Stack(relay_request_timeout=15.0) as stack, httpx.AsyncClient() as http:
        sid = "trial_race"
        first = await stack.chat(http, sid, TURN1)
        assert first.status_code == 200, first.text
        reply = first.json()["choices"][0]["message"]["content"]

        stack.provider.delay = 0.8
        engine_calls = stack.provider.engine_calls
        a, b = await asyncio.gather(
            stack.chat(http, sid, turn2(reply)),
            stack.chat(http, sid, turn2(reply)),
        )
        check("both the original and the racing duplicate answer 200",
              a.status_code == 200 and b.status_code == 200)
        check("both carry the same completion",
              a.json()["choices"][0]["message"]["content"]
              == b.json()["choices"][0]["message"]["content"]
              == "<900><901><902>")
        check("the engine ran exactly once for the pair",
              stack.provider.engine_calls == engine_calls + 1)
        check("the record gained exactly one turn",
              await stack.turns_recorded(http, sid) == 2)


async def test_local_disconnect_recovery() -> None:
    print("\nLocal mode: caller disconnect mid-turn is shielded — turn recorded, retry re-served")
    async with LocalStack() as stack, httpx.AsyncClient() as http:
        sid = "trial_local_drop"
        first = await stack.chat(http, sid, TURN1)
        assert first.status_code == 200, first.text
        reply = first.json()["choices"][0]["message"]["content"]

        # Turn 2 outlives the agent's client timeout: the client gives up
        # and drops the connection while the engine call is still running.
        # Depending on the Starlette version this may cancel the endpoint
        # task — the shielded pipeline must survive either way.
        stack.provider.delay = 1.5
        engine_calls = stack.provider.engine_calls
        try:
            await stack.chat(http, sid, turn2(reply), timeout=0.5)
            raise AssertionError("expected the client-side timeout to fire")
        except httpx.TimeoutException:
            pass

        # The pipeline finishes detached: the turn commits and is recorded
        # with no retry involved.
        await stack.wait_turns(http, sid, 2, deadline_s=5.0)
        check("turn recorded despite the disconnect (shielded pipeline)", True)
        check("generation ran exactly once", stack.provider.engine_calls == engine_calls + 1)

        # An exact retry is re-served the committed turn, not regenerated.
        stack.provider.delay = 0.0
        retry = await stack.chat(http, sid, turn2(reply))
        check("exact retry answers 200", retry.status_code == 200)
        check("retry is the committed completion",
              retry.json()["choices"][0]["message"]["content"] == "<900><901><902>")
        check("retry did not regenerate", stack.provider.engine_calls == engine_calls + 1)
        check("the turn is recorded exactly once", await stack.turns_recorded(http, sid) == 2)


async def test_opening_turn_disconnect_recovery() -> None:
    print("\nThe *opening* turn survives a disconnect too — one generation, one record")
    # The opening request carries no assistant message, so nothing about it
    # looks like a continuation — and it used to fall through every recovery
    # path: its retry re-rendered the conversation, paid for a second
    # generation of what is usually the rollout's longest turn, and recorded a
    # phantom branch the agent never saw next to the real one.
    async with LocalStack() as stack, httpx.AsyncClient() as http:
        sid = "trial_opening_drop"
        stack.provider.delay = 1.5
        try:
            await stack.chat(http, sid, TURN1, timeout=0.5)
            raise AssertionError("expected the client-side timeout to fire")
        except httpx.TimeoutException:
            pass

        await stack.wait_turns(http, sid, 1, deadline_s=5.0)
        check("opening turn recorded despite the disconnect", True)
        check("generation ran exactly once", stack.provider.engine_calls == 1)

        stack.provider.delay = 0.0
        retry = await stack.chat(http, sid, TURN1)
        check("exact retry of the opening request answers 200", retry.status_code == 200)
        check("retry is re-served the committed completion",
              retry.json()["choices"][0]["message"]["content"] == "<900><901><902>")
        check("retry did not re-sample the opening turn", stack.provider.engine_calls == 1)
        check("the opening turn is recorded exactly once",
              await stack.turns_recorded(http, sid) == 1)

        # The rollout continues normally on the re-served turn.
        reply = retry.json()["choices"][0]["message"]["content"]
        follow = await stack.chat(http, sid, turn2(reply))
        check("the conversation continues on the re-served turn", follow.status_code == 200)
        check("...as a second recorded turn, generated once",
              await stack.turns_recorded(http, sid) == 2 and stack.provider.engine_calls == 2)

        # A *different* opening request still opens a fresh conversation.
        other = await stack.chat(http, sid, [{"role": "user", "content": "phase 2"}])
        check("a different assistant-free request still re-samples",
              other.status_code == 200 and stack.provider.engine_calls == 3)


async def test_blank_worker_error_is_not_a_completion() -> None:
    print("\nA worker failure with a blank message is an error, not an empty turn")
    # str(e) is "" for every httpx timeout — the single most likely worker
    # failure — so classifying on the truthiness of the error message
    # delivered the failure payload as a *successful* turn: HTTP 200 with
    # empty content and no tool calls (nothing for the agent to act on), plus
    # a zero-token turn recorded into the training data, which resets the
    # recorder's delta baseline and buries earlier sampled tokens inside a
    # later turn's prompt_token_ids.
    from proxyserver.relay import InferenceRelay, WorkerError
    from proxyserver.worker_client import _describe, _error_response, _response_message

    import httpx as _httpx
    check("httpx timeouts really do stringify to ''", str(_httpx.ReadTimeout("")) == "")
    check("the worker never emits a blank error message",
          _describe(_httpx.ReadTimeout("")) == "ReadTimeout")

    async def dispatch_answered_with(relay, response) -> asyncio.Task:
        """Dispatch one request and answer it with ``response``."""
        sent = asyncio.get_running_loop().create_future()

        class _WS:
            async def send_json(self, msg):
                sent.set_result(msg["request_id"])

        relay.register_worker("w", _WS())
        task = asyncio.create_task(relay.dispatch(session_id="s", model=MODEL, messages=[]))
        relay.deliver_response(await sent, response)
        return task

    relay = InferenceRelay(request_timeout=5.0)
    recorded: list[dict[str, Any]] = []
    relay.on_orphan_response = lambda pending, response: recorded.append(response)

    # A payload built the way a worker built it before _describe existed.
    blank = _response_message("r1", _error_response("", error_type="internal"))
    try:
        await (await dispatch_answered_with(relay, blank))
        raise AssertionError("FAIL: a blank worker error was delivered as a completion")
    except WorkerError as e:
        check("a blank-message failure raises WorkerError", True)
        check("...with a message the operator can act on", str(e).strip() != "Worker error:")
    check("the empty turn never reached the recorder", recorded == [])

    # An explicit null error_type must still mean 400, not a retryable 503.
    relay2 = InferenceRelay(request_timeout=5.0)
    try:
        await (await dispatch_answered_with(
            relay2, {"error": "bad history", "error_type": None}))
        raise AssertionError("FAIL: worker error not raised")
    except WorkerError as e:
        check("a null error_type falls back to internal, not None",
              e.error_type == "internal")


async def test_deleted_session_drops_late_turn() -> None:
    print("\nDeleted session: the in-flight turn is dropped, stragglers are refused")
    # The cancelled-trial shape: the driver deletes the session while a turn
    # is still generating.  Turn-preservation exists for callers that retry;
    # a deleted session has no retry coming, so the late delivery must be
    # dropped — recording it would re-create the record (and the worker-side
    # token stream) that DELETE just freed, and the driver deletes exactly
    # once, so nothing would ever free them again.
    async with Stack(relay_request_timeout=1.0) as stack, httpx.AsyncClient() as http:
        sid = "trial_cancelled"
        stream = f"{sid}{DELIMITER}{AGENT_ID}"
        first = await stack.chat(http, sid, TURN1)
        assert first.status_code == 200, first.text
        reply = first.json()["choices"][0]["message"]["content"]

        # Turn 2 outlives the relay timeout (503) and is still generating
        # when the driver cancels the trial and deletes the session.
        stack.provider.delay = 2.5
        resp = await stack.chat(http, sid, turn2(reply))
        check("caller gets 503 when the relay times out", resp.status_code == 503)
        resp = await http.delete(f"{stack.url}/sessions/{sid}")
        check("delete answers 200 while the turn is still generating",
              resp.status_code == 200)

        # The worker finishes anyway; give its late delivery time to land.
        await asyncio.sleep(2.5)
        resp = await http.get(f"{stack.url}/sessions/{sid}")
        check("the late completion did not resurrect the record",
              resp.status_code == 404)
        check("the late commit did not resurrect the worker's token stream",
              not stack.provider.models.has_session(stream))

        # A straggler request of the cancelled trial is refused outright —
        # serving it would occupy an engine with a turn nobody trains on and
        # re-create per-session state nothing ever cleans up again.
        stack.provider.delay = 0.0
        engine_calls = stack.provider.engine_calls
        resp = await stack.chat(http, sid, turn2(reply))
        check("a straggler request answers 410", resp.status_code == 410)
        check("the straggler never reached an engine",
              stack.provider.engine_calls == engine_calls)
        resp = await http.get(f"{stack.url}/sessions/{sid}")
        check("the straggler did not re-create the record", resp.status_code == 404)


async def main() -> None:
    await test_blank_worker_error_is_not_a_completion()
    await test_relay_timeout_recovery()
    await test_relay_retry_races_inflight_original()
    await test_worker_disconnect_recovery()
    await test_concurrent_duplicate_serialized()
    await test_local_disconnect_recovery()
    await test_opening_turn_disconnect_recovery()
    await test_deleted_session_drops_late_turn()
    print("\n" + "=" * 70)
    print("PASS: lost-response recovery — relay timeouts record the late turn\n"
          "      and re-serve exact retries (even retries racing their\n"
          "      still-generating original, which serialize per session at\n"
          "      the provider and never re-sample); a worker disconnect\n"
          "      mid-generation buffers and re-delivers to the still-waiting\n"
          "      caller; racing duplicates serialize and generate exactly\n"
          "      once; a local-mode client disconnect leaves the shielded\n"
          "      pipeline to finish, record, and re-serve the turn — the\n"
          "      session's opening turn included; and a session deleted\n"
          "      mid-generation (a cancelled trial) drops its late turn and\n"
          "      refuses its stragglers instead of resurrecting freed state.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
