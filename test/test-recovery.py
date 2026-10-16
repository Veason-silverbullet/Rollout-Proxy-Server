"""Offline recovery tests over real HTTP with a scripted inference provider.

Checks concurrent retries, detached generation after client timeouts,
opening-turn replay, and deletion of sessions with active inference.
Run: python test/test-recovery.py
"""

from __future__ import annotations
import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver.model_registry import ModelRegistry  # noqa: E402
from proxyserver.rollout_provider import BaseRolloutProvider  # noqa: E402
from proxyserver.server import LLMProxyServer, make_completion_handler  # noqa: E402
from offline_common import GatedProvider, ScriptedProvider, check  # noqa: E402

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


class ProxyStack(_Harness):
    """One proxy (injected completion handler) on a scripted provider."""

    def __init__(self, provider=None):
        self.provider = provider if provider is not None else ScriptedProvider()
        self.proxy = LLMProxyServer(
            host="127.0.0.1", port=0, api_key=PROXY_API_KEY,
            key_delimiter=DELIMITER,
            completion_handler=make_completion_handler(self.provider),
            on_session_deleted=self.provider.release_session,
            save_rollout_sessions=False,
        )
        self.url = ""

    async def __aenter__(self) -> "ProxyStack":
        self.url = await self.proxy.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.proxy.stop()


TURN1 = [{"role": "user", "content": "hi"}]


def turn2(reply: str) -> list[dict[str, Any]]:
    return TURN1 + [{"role": "assistant", "content": reply},
                    {"role": "user", "content": "more"}]


async def test_concurrent_duplicate_serialized() -> None:
    print("\nConcurrent duplicate (SDK-style auto-retry) serializes and re-serves")
    async with ProxyStack() as stack, httpx.AsyncClient() as http:
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


async def disconnect_during_generation(stack, http, sid, messages) -> None:
    """Drop a real socket only after the engine has entered its blocking gate."""
    stack.provider.gate = asyncio.Event()
    pending = asyncio.create_task(stack.chat(http, sid, messages, timeout=0.5))
    try:
        started = await asyncio.wait_for(stack.provider.started.get(), 5)
        assert started == f"{sid}{DELIMITER}{AGENT_ID}"
        try:
            await pending
        except httpx.ReadTimeout:
            pass
        else:
            raise AssertionError("expected the client-side read timeout to fire")
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


async def test_disconnect_recovery() -> None:
    print("\nOpening and continuation turns survive disconnects and replay exactly once")
    for continuation in (False, True):
        provider = GatedProvider()
        async with ProxyStack(provider) as stack, httpx.AsyncClient() as http:
            sid = f"trial_drop_{continuation}"
            messages = TURN1
            if continuation:
                first = await stack.chat(http, sid, messages)
                assert first.status_code == 200, first.text
                messages = turn2(first.json()["choices"][0]["message"]["content"])
            baseline = int(continuation)

            try:
                await disconnect_during_generation(stack, http, sid, messages)
                check("the disconnected turn is still generating", not provider.finished.is_set())
            finally:
                if provider.gate is not None:
                    provider.gate.set()
            await stack.wait_turns(http, sid, baseline + 1, deadline_s=5.0)
            check("turn recorded without a retry after disconnect", provider.engine_calls == baseline + 1)

            retry = await stack.chat(http, sid, messages)
            check("exact retry answers 200", retry.status_code == 200)
            reply = retry.json()["choices"][0]["message"]["content"]
            check("retry is the committed completion", reply == "<900><901><902>")
            check("retry neither regenerates nor duplicates the record",
                  provider.engine_calls == baseline + 1
                  and await stack.turns_recorded(http, sid) == baseline + 1)

            follow = await stack.chat(http, sid, messages + [
                {"role": "assistant", "content": reply}, {"role": "user", "content": "next"},
            ])
            check("the conversation continues on the re-served turn",
                  follow.status_code == 200 and provider.engine_calls == baseline + 2
                  and await stack.turns_recorded(http, sid) == baseline + 2)
            other = await stack.chat(http, sid, [{"role": "user", "content": "phase 2"}])
            check("a different assistant-free request still re-samples",
                  other.status_code == 200 and provider.engine_calls == baseline + 3)


async def test_cancelled_handler_retry_races_original() -> None:
    """Force handler cancellation, independently of HTTP server disconnect behavior."""
    started = asyncio.Event()
    finish = asyncio.Event()

    class GatedProvider(ScriptedProvider):
        async def _call_engine(self, prompt_ids, sampling_params, session_id):
            self.engine_calls += 1
            started.set()
            await finish.wait()
            return [900], [-0.1], "stop", {"weight_version": "7"}

    provider = GatedProvider()
    proxy = LLMProxyServer(
        api_key=PROXY_API_KEY, save_rollout_sessions=False,
        completion_handler=make_completion_handler(provider),
        on_session_deleted=provider.release_session,
    )
    sid = "cancelled_handler"
    headers = {"Authorization": f"Bearer {PROXY_API_KEY}-{sid}-{AGENT_ID}"}
    body = {"model": MODEL, "messages": TURN1}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as http:
        original = asyncio.create_task(http.post("/v1/chat/completions", headers=headers, json=body))
        retry = None
        try:
            await asyncio.wait_for(started.wait(), 5)
            original.cancel()
            result = await asyncio.gather(original, return_exceptions=True)
            check("HTTP handler cancellation propagates", isinstance(result[0], asyncio.CancelledError))
            check("generation survives in a detached task", bool(proxy._orphan_tasks))
            retry = asyncio.create_task(http.post("/v1/chat/completions", headers=headers, json=body))
            await asyncio.sleep(0.05)
            check("retry waits while original is generating", not retry.done() and provider.engine_calls == 1)
            finish.set()
            response = await asyncio.wait_for(retry, 5)
            check("retry receives the committed completion", response.status_code == 200)
            turns = proxy.recorder.dump_session(sid)["turns"]
            check("one generation and one recorded turn", provider.engine_calls == 1 and len(turns) == 1)
            check("cached turn keeps its original policy version", turns[0]["weight_version"] == "7")
        finally:
            finish.set()
            original.cancel()
            if retry is not None:
                retry.cancel()
            await asyncio.gather(original, *([retry] if retry is not None else []), return_exceptions=True)
            await proxy.delete_session(sid)


async def test_deleted_session_drops_late_turn() -> None:
    print("\nDeleted session: active generation is drained and stragglers are refused")
    provider = GatedProvider()
    async with ProxyStack(provider) as stack, httpx.AsyncClient() as http:
        sid = "trial_cancelled"
        stream = f"{sid}{DELIMITER}{AGENT_ID}"
        first = await stack.chat(http, sid, TURN1)
        assert first.status_code == 200, first.text
        reply = first.json()["choices"][0]["message"]["content"]

        try:
            await disconnect_during_generation(stack, http, sid, turn2(reply))
            active = list(stack.proxy._session_tasks.get(sid, ()))
            assert active and not provider.finished.is_set()
            resp = await http.delete(f"{stack.url}/sessions/{sid}")
            check("delete answers 200 while generation is held", resp.status_code == 200)
            check("delete cancels generation and drains all session work",
                  provider.cancelled.is_set() and provider.finished.is_set()
                  and all(task.done() for task in active))
        finally:
            if provider.gate is not None:
                provider.gate.set()

        resp = await http.get(f"{stack.url}/sessions/{sid}")
        check("the completion did not resurrect the record", resp.status_code == 404)
        check("the commit did not resurrect the provider's token stream",
              not provider.models.has_session(stream))

        engine_calls = provider.engine_calls
        resp = await stack.chat(http, sid, turn2(reply))
        check("a straggler request answers 410", resp.status_code == 410)
        check("the straggler never reached an engine", provider.engine_calls == engine_calls)
        resp = await http.get(f"{stack.url}/sessions/{sid}")
        check("the straggler did not re-create the record", resp.status_code == 404)


async def test_detached_pipeline_serializes_through_recording() -> None:
    """A retry cannot overtake a committed turn still parsing or recording."""
    for stage in ("parsing", "recording"):
        for continuation in (False, True):
            entered = asyncio.Event()
            finish_parse = asyncio.Event()
            finish_record = threading.Event()

            class GatedProvider(ScriptedProvider):
                gate_next_parse = False

                async def parse_completion(self, *args, **kwargs):
                    if self.gate_next_parse:
                        self.gate_next_parse = False
                        entered.set()
                        await finish_parse.wait()
                    return await super().parse_completion(*args, **kwargs)

            provider = GatedProvider()
            proxy = LLMProxyServer(
                api_key=PROXY_API_KEY, save_rollout_sessions=False,
                completion_handler=make_completion_handler(provider),
                on_session_deleted=provider.release_session,
            )
            sid = f"late_{stage}_{continuation}"
            headers = {"Authorization": f"Bearer {PROXY_API_KEY}-{sid}-{AGENT_ID}"}
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as http:
                async def chat(messages):
                    return await http.post("/v1/chat/completions", headers=headers, json={"model": MODEL, "messages": messages})

                messages = TURN1
                if continuation:
                    first = await chat(messages)
                    assert first.status_code == 200, first.text
                    messages = turn2(first.json()["choices"][0]["message"]["content"])
                baseline = int(continuation)
                provider.next_tokens = [910]
                if stage == "parsing":
                    provider.gate_next_parse = True
                else:
                    record = proxy.recorder.record_completion
                    loop = asyncio.get_running_loop()
                    gate_next_record = True

                    def gated_record(*args, **kwargs):
                        nonlocal gate_next_record
                        if gate_next_record:
                            gate_next_record = False
                            loop.call_soon_threadsafe(entered.set)
                            if not finish_record.wait(5):
                                raise AssertionError("recording gate was not released")
                        return record(*args, **kwargs)

                    proxy.recorder.record_completion = gated_record

                original = asyncio.create_task(chat(messages))
                retry = None
                try:
                    await asyncio.wait_for(entered.wait(), 5)
                    original.cancel()
                    result = await asyncio.gather(original, return_exceptions=True)
                    assert isinstance(result[0], asyncio.CancelledError)
                    detached = list(proxy._orphan_tasks)
                    assert detached
                    retry = asyncio.create_task(chat(messages))
                    await asyncio.sleep(0.1)
                    check(f"{stage}, continuation={continuation}: retry waits for recording",
                          not retry.done())
                    assert len(proxy.recorder.dump_session(sid)["turns"]) == baseline

                    finish_parse.set()
                    finish_record.set()
                    response = await asyncio.wait_for(retry, 5)
                    assert response.status_code == 200, response.text
                    reply = response.json()["choices"][0]["message"]
                    provider.next_tokens = [920]
                    follow = await chat(messages + [reply, {"role": "user", "content": "next"}])
                    assert follow.status_code == 200, follow.text
                    await asyncio.wait_for(asyncio.gather(*detached), 5)
                    turns = proxy.recorder.dump_session(sid)["turns"]
                    check("the detached turn is recorded once, before the following turn",
                          len(turns) == baseline + 2
                          and [t["completion_token_ids"] for t in turns[baseline:]] == [[910], [920]]
                          and [t["new_conversation"] for t in turns] == [True] + [False] * (baseline + 1)
                          and provider.engine_calls == baseline + 2)
                finally:
                    finish_parse.set()
                    finish_record.set()
                    original.cancel()
                    if retry is not None:
                        retry.cancel()
                    await asyncio.gather(original, *([retry] if retry is not None else []),
                                         return_exceptions=True)
                    await proxy.delete_session(sid)


async def test_tool_choice_constraints_rejected_before_generation_or_replay() -> None:
    from unittest.mock import patch
    from proxyserver.tokenization.tool_parser import BaseToolParser, ToolCall

    class Parser(BaseToolParser):
        async def extract_tool_calls(self, token_ids, tools=None):
            return "", [ToolCall(name="run", arguments="{}")]

    provider = ScriptedProvider()
    bundle = provider.models.resolve(MODEL)
    bundle.tool_parser = Parser(bundle.tokenizer)
    with TemporaryDirectory() as root:
        proxy = LLMProxyServer(
            api_key=PROXY_API_KEY, save_rollout_sessions=False, error_log_dir=root,
            completion_handler=make_completion_handler(provider),
            on_session_deleted=provider.release_session,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as http:
            async def chat(**extra):
                return await http.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {PROXY_API_KEY}-choice-agent"},
                    json={"model": MODEL, "messages": TURN1, "tools": [{
                        "type": "function", "function": {"name": "run"},
                    }], **extra},
                )

            unsupported = ("none", "required", {"type": "function", "function": {"name": "run"}},
                           "invalid", False, [], {})
            with patch.object(provider, "generate", wraps=provider.generate) as generate:
                for choice in unsupported:
                    response = await chat(tool_choice=choice)
                    assert response.status_code == 400 and "tool_choice" in response.json()["error"]
                assert generate.call_count == provider.engine_calls == 0
                assert not provider.models.has_session("choice-agent")

                first = await chat(tool_choice="auto")
                assert first.status_code == 200, first.text
                choices = first.json()["choices"]
                assert choices[0]["finish_reason"] == "tool_calls"
                assert choices[0]["message"]["tool_calls"]

                # These messages now match a cached tool-call turn. Reject
                # constraints before replay, including requests using SSE.
                for choice in unsupported:
                    response = await chat(tool_choice=choice, stream=True)
                    assert response.status_code == 400 and "tool_choice" in response.json()["error"]
                assert generate.call_count == provider.engine_calls == 1

                # Rejections neither spend the replay allowance nor poison
                # collection. Omitted/null choices retain automatic behavior.
                for extra in ({}, {"tool_choice": None}):
                    retry = await chat(**extra)
                    assert retry.status_code == 200, retry.text
                    assert retry.json()["choices"] == choices
                assert generate.call_count == 3 and provider.engine_calls == 1
                record = await http.get("/sessions/choice")
                assert record.status_code == 200, record.text
                assert len(record.json()["turns"]) == 1
                assert record.json()["turns"][0]["tool_calls"] == choices[0]["message"]["tool_calls"]
        await proxy.delete_session("choice")
    check("unsupported tool choices fail before generation/replay; auto, null and omitted choices work", True)


async def test_tool_call_retry_identity() -> None:
    """Replay the same assistant message, including IDs used by tool results."""
    from proxyserver.tokenization.tool_parser import BaseToolParser, ToolCall

    class Reasoning:
        calls = 0

        def extract(self, text):
            self.calls += 1
            return f"Plan {self.calls}"

    class Parser(BaseToolParser):
        calls = 0

        async def extract_tool_calls(self, token_ids, tools=None):
            self.calls += 1
            return "Running tools.", [ToolCall(name="run", arguments='{"x":1}'),
                                      ToolCall(name="inspect", arguments='{}')]

    provider = ScriptedProvider()
    reasoning = Reasoning()
    parser = Parser(provider.models.resolve(MODEL).tokenizer, reasoning=reasoning)
    provider.models.resolve(MODEL).tool_parser = parser
    proxy = LLMProxyServer(
        api_key=PROXY_API_KEY, save_rollout_sessions=False,
        completion_handler=make_completion_handler(provider),
        on_session_deleted=provider.release_session,
    )
    sid = "tool_retry"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as http:
        async def chat(messages, agent=AGENT_ID, streaming=False):
            response = await http.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {PROXY_API_KEY}-{sid}-{agent}"},
                json={"model": MODEL, "messages": messages, "stream": streaming},
            )
            assert response.status_code == 200, response.text
            return response

        try:
            first = (await chat(TURN1)).json()
            retry = (await chat(TURN1)).json()
            message = first["choices"][0]["message"]
            check("retry preserves the complete assistant message", retry["choices"] == first["choices"])
            check("retry reuses content, reasoning and tool calls", parser.calls == reasoning.calls == provider.engine_calls == 1
                  and message["reasoning_content"] == "Plan 1")
            assert len({tc["id"] for tc in message["tool_calls"]}) == 2

            streamed = await chat(TURN1, streaming=True)
            chunks = [json.loads(line[6:]) for line in streamed.text.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            delta = chunks[0]["choices"][0]["delta"]
            assert [{k: v for k, v in tc.items() if k != "index"} for tc in delta["tool_calls"]] == message["tool_calls"]
            assert delta["content"] == message["content"]
            assert delta["reasoning_content"] == message["reasoning_content"]
            assert parser.calls == reasoning.calls == 1, "streaming reparsed the cached completion"
            assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

            # Each tool result references the ID delivered by the retry.
            tool_results = [{"role": "tool", "tool_call_id": tc["id"], "content": "ok"}
                            for tc in retry["choices"][0]["message"]["tool_calls"]]
            next_messages = TURN1 + [message] + tool_results
            following = (await chat(next_messages)).json()
            repeated = (await chat(next_messages)).json()
            assert repeated["choices"] == following["choices"]
            next_ids = {tc["id"] for tc in following["choices"][0]["message"]["tool_calls"]}
            assert next_ids.isdisjoint(tc["id"] for tc in message["tool_calls"])
            turns = (await http.get(f"/sessions/{sid}")).json()["turns"]
            check("recorded calls match the retry and its subsequent tool results",
                  len(turns) == 2 and turns[0]["tool_calls"] == message["tool_calls"]
                  and turns[1]["request_messages"] == tool_results
                  and parser.calls == reasoning.calls == provider.engine_calls == 2
                  and turns[0]["reasoning_content"] == message["reasoning_content"]
                  and turns[1]["reasoning_content"] == following["choices"][0]["message"]["reasoning_content"])

            # Identical sampled tokens in another conversation/agent are new calls.
            fresh = (await chat([{"role": "user", "content": "new task"}])).json()
            other = (await chat(TURN1, agent="other")).json()
            all_ids = [tc["id"] for response in (first, following, fresh, other)
                       for tc in response["choices"][0]["message"]["tool_calls"]]
            check("new turns, conversations and agents get distinct call IDs", len(set(all_ids)) == 8)
        finally:
            await proxy.delete_session(sid)


async def test_normalized_opening_retry_recorded_once() -> None:
    print("\nEquivalent text representations replay an opening turn without recording it twice")

    class RecordingProvider(ScriptedProvider):
        def __init__(self):
            super().__init__()
            self.prompts = []

        async def _call_engine(self, prompt_ids, sampling_params, session_id):
            self.prompts.append(list(prompt_ids))
            return await super()._call_engine(prompt_ids, sampling_params, session_id)

    plain = [{"role": "user", "content": "hi\nthere"}]
    multipart = [{"role": "user", "content": [
        {"type": "text", "text": "hi"}, {"type": "text", "text": "there"},
    ]}]
    for opening, repeated in ((plain, multipart), (multipart, plain)):
        provider = RecordingProvider()
        with TemporaryDirectory() as session_dir:
            proxy = LLMProxyServer(
                api_key=PROXY_API_KEY, session_dir=session_dir,
                completion_handler=make_completion_handler(provider),
                on_session_deleted=provider.release_session,
            )
            sid = "normalized_retry"
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as http:
                async def chat(messages):
                    response = await http.post(
                        "/v1/chat/completions",
                        headers={"Authorization": f"Bearer {PROXY_API_KEY}-{sid}-{AGENT_ID}"},
                        json={"model": MODEL, "messages": messages},
                    )
                    assert response.status_code == 200, response.text
                    return response.json()

                try:
                    first = await chat(opening)
                    retry = await chat(repeated)
                    assert retry["choices"] == first["choices"]
                    assert provider.engine_calls == 1
                    turns = (await http.get(f"/sessions/{sid}")).json()["turns"]
                    assert len(turns) == 1, "normalized opening retry was recorded twice"
                    assert turns[0]["request_messages"] == opening
                    path = proxy.session_store.run_dir / f"{sid}.json"
                    assert len(json.loads(path.read_text())["turns"]) == 1

                    new_message = {"role": "user", "content": "continue"}
                    await chat(repeated + [retry["choices"][0]["message"], new_message])
                    turns = (await http.get(f"/sessions/{sid}")).json()["turns"]
                    assert provider.engine_calls == len(turns) == 2
                    assert not turns[1]["new_conversation"]
                    assert turns[1]["request_messages"] == [new_message]
                    assert (turns[0]["prompt_token_ids"] + turns[0]["completion_token_ids"]
                            + turns[1]["prompt_token_ids"]) == provider.prompts[1]
                    assert len(json.loads(path.read_text())["turns"]) == 2
                    check("one recorded opening, original messages preserved, continuation tokens reconstruct exactly", True)
                finally:
                    await proxy.delete_session(sid)


async def test_unresolved_recording_failure_refuses_collection() -> None:
    print("\nA recording failure after the turn is committed fails loudly")
    # The engine has committed the turn by the time recording runs, so the
    # failure cannot be undone: the completion is re-served to an exact
    # retry, and a record missing the turn would silently omit its sampled
    # tokens from training. The handler must answer 500 (so the agent's
    # retry re-attempts the recording) and mark the session's recording
    # failed, so GET /sessions/{id} refuses to serve it and the driver's
    # collection fails the trial into a whole-task retry.
    async with ProxyStack() as stack, httpx.AsyncClient() as http:
        sid = "trial_record_fail"

        def exploding_record(**kwargs):
            raise RuntimeError("recorder exploded")

        original = stack.proxy.recorder.record_completion
        stack.proxy.recorder.record_completion = exploding_record
        try:
            resp = await stack.chat(http, sid, TURN1)
            check("the completion answers 500 when recording fails",
                  resp.status_code == 500 and "record" in resp.text)
            check("the failed session refuses collection",
                  (await http.get(f"{stack.url}/sessions/{sid}")).status_code == 500)

            # An exact retry re-serves the committed turn and re-attempts
            # the recording: still deterministic, still 500, still refused.
            retry = await stack.chat(http, sid, TURN1)
            check("a deterministic recording failure repeats on the retry",
                  retry.status_code == 500)
            check("the session is still refused after the retry",
                  (await http.get(f"{stack.url}/sessions/{sid}")).status_code == 500)

            # Restoring the recorder alone leaves the turn missing; an
            # unrelated session is unaffected.
            stack.proxy.recorder.record_completion = original
            ok = await stack.chat(http, sid + "2", TURN1)
            check("an unrelated session records and collects normally",
                  ok.status_code == 200
                  and (await http.get(f"{stack.url}/sessions/{sid}2")).status_code == 200)
            check("the session stays refused until its turn is recovered",
                  (await http.get(f"{stack.url}/sessions/{sid}")).status_code == 500)
        finally:
            stack.proxy.recorder.record_completion = original


async def test_parser_failure_blocks_collection_until_retry() -> None:
    print("\nParser failures after commit block collection until an exact retry records the turn")
    from unittest.mock import patch

    class RecordingProvider(ScriptedProvider):
        def __init__(self):
            super().__init__()
            self.prompts = []

        async def _call_engine(self, prompt_ids, sampling_params, session_id):
            self.prompts.append(list(prompt_ids))
            return await super()._call_engine(prompt_ids, sampling_params, session_id)

    for continuation in (False, True):
        provider = RecordingProvider()
        proxy = LLMProxyServer(
            api_key=PROXY_API_KEY, completion_handler=make_completion_handler(provider),
            on_session_deleted=provider.release_session, save_rollout_sessions=False,
        )
        sid = f"parser_failure_{continuation}"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as http:
            async def chat(messages, agent=AGENT_ID):
                return await http.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {PROXY_API_KEY}-{sid}-{agent}"},
                    json={"model": MODEL, "messages": messages},
                )

            try:
                messages = TURN1
                if continuation:
                    opening = await chat(messages)
                    assert opening.status_code == 200, opening.text
                    messages = turn2(opening.json()["choices"][0]["message"]["content"])

                parser = provider.models.resolve(MODEL).tool_parser
                with patch.object(parser, "extract_completion", side_effect=RuntimeError("parser exploded")):
                    failed = await chat(messages)
                    assert failed.status_code == 500 and "parser exploded" in failed.text
                    assert (await http.get(f"/sessions/{sid}")).status_code == 500
                    assert provider.models.has_session(f"{sid}-{AGENT_ID}")
                    assert len(proxy.recorder.get_session(sid).turns) == int(continuation)

                    repeated_failure = await chat(messages)
                    assert repeated_failure.status_code == 500, repeated_failure.text
                    assert (await http.get(f"/sessions/{sid}")).status_code == 500
                    assert provider.engine_calls == 1 + int(continuation)

                expected_stream = provider.prompts[-1] + provider.next_tokens
                # Restoring the parser and successfully serving another agent
                # cannot repair this agent's missing committed turn.
                other = await chat(TURN1, agent="other")
                assert other.status_code == 200, other.text
                assert (await http.get(f"/sessions/{sid}")).status_code == 500
                engine_calls = provider.engine_calls

                repeated = messages[:-1] + [{
                    **messages[-1],
                    "content": [{"type": "text", "text": messages[-1]["content"]}],
                }]
                recovered = await chat(repeated)
                assert recovered.status_code == 200, recovered.text
                assert provider.engine_calls == engine_calls
                collected = await http.get(f"/sessions/{sid}")
                assert collected.status_code == 200, collected.text
                turns = [t for t in collected.json()["turns"] if t["agent_id"] == AGENT_ID]
                assert len(turns) == 1 + int(continuation)
                rebuilt = [token for turn in turns for token in
                           turn["prompt_token_ids"] + turn["completion_token_ids"]]
                assert rebuilt == expected_stream
                assert turns[-1]["completion_logprobs"] == [-0.1] * len(provider.next_tokens)
                assert turns[-1]["content"] == recovered.json()["choices"][0]["message"]["content"]
                check(f"{'continuation' if continuation else 'opening'}: parser retry repairs collection without resampling", True)
            finally:
                await proxy.delete_session(sid)


async def test_recording_failure_exact_retry_recovers() -> None:
    print("\nAn exact retry repairs recording errors before and after the in-memory append")
    from unittest.mock import patch

    class RecordingProvider(ScriptedProvider):
        def __init__(self):
            super().__init__()
            self.prompts = []

        async def _call_engine(self, prompt_ids, sampling_params, session_id):
            self.prompts.append(list(prompt_ids))
            return await super()._call_engine(prompt_ids, sampling_params, session_id)

    for continuation in (False, True):
        for failure_at in ("record_completion", "_snapshot", "_save"):
            provider = RecordingProvider()
            proxy = LLMProxyServer(
                api_key=PROXY_API_KEY, completion_handler=make_completion_handler(provider),
                on_session_deleted=provider.release_session, save_rollout_sessions=False,
            )
            sid = f"retry_{continuation}_{failure_at}"
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as http:
                async def chat(messages):
                    return await http.post(
                        "/v1/chat/completions",
                        headers={"Authorization": f"Bearer {PROXY_API_KEY}-{sid}-{AGENT_ID}"},
                        json={"model": MODEL, "messages": messages},
                    )

                try:
                    messages = TURN1
                    if continuation:
                        opening = await chat(messages)
                        assert opening.status_code == 200, opening.text
                        messages = turn2(opening.json()["choices"][0]["message"]["content"])

                    # Keep the fault active during the retry: a post-append
                    # failure must recover through dedup without persisting
                    # again, while a missing turn must actually be appended.
                    original = getattr(proxy.recorder, failure_at)
                    calls = 0

                    def fail_once(*args, **kwargs):
                        nonlocal calls
                        calls += 1
                        if calls == 1:
                            raise RuntimeError(f"injected failure at {failure_at}")
                        return original(*args, **kwargs)

                    parser = provider.models.resolve(MODEL).tool_parser
                    with patch.object(proxy.recorder, failure_at, side_effect=fail_once), \
                         patch.object(parser, "extract_completion", wraps=parser.extract_completion) as decode:
                        first = await chat(messages)
                        assert first.status_code == 500, first.text
                        assert (await http.get(f"/sessions/{sid}")).status_code == 500
                        # Normalized-equivalent requests are exact retries.
                        repeated = messages[:-1] + [{
                            **messages[-1],
                            "content": [{"type": "text", "text": messages[-1]["content"]}],
                        }]
                        retry = await chat(repeated)
                        assert retry.status_code == 200, retry.text
                        assert decode.call_count == 1, "recording recovery reparsed a committed completion"

                    collected = await http.get(f"/sessions/{sid}")
                    assert collected.status_code == 200, collected.text
                    turns = collected.json()["turns"]
                    assert len(turns) == provider.engine_calls == 1 + int(continuation)
                    rebuilt = [token for turn in turns for token in
                               turn["prompt_token_ids"] + turn["completion_token_ids"]]
                    assert rebuilt == provider.prompts[-1] + provider.next_tokens
                    if continuation:
                        assert not turns[-1]["new_conversation"]
                    check(f"{'continuation' if continuation else 'opening'} / {failure_at}: retry restores collection without duplication", True)
                finally:
                    await proxy.delete_session(sid)


async def test_glm_tool_call_retry_recording() -> None:
    bundle = ModelRegistry().resolve("GLM-5.3-Flash")
    tools = [{"type": "function", "function": {
        "name": "lookup", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }}]
    call_text = "<tool_call>lookup<arg_key>city</arg_key><arg_value>上海</arg_value></tool_call>"
    class Provider(BaseRolloutProvider):
        def __init__(self):
            super().__init__(tokenizer_loader=lambda _: bundle.tokenizer)
            self.calls = 0

        async def _call_engine(self, prompt_ids, sampling_params, session_id):
            self.calls += 1
            text = "<think>plan</think>" + call_text + "<|observation|>" if self.calls == 1 else "<think></think>sunny<|user|>"
            tokens = bundle.tokenizer.encode(text, add_special_tokens=False)
            return tokens, [-0.1] * len(tokens), "stop", {}

    provider = Provider()
    with TemporaryDirectory() as tmp:
        proxy = LLMProxyServer(api_key="test", completion_handler=make_completion_handler(provider),
                               on_session_deleted=provider.release_session, save_rollout_sessions=False, error_log_dir=tmp)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://proxy",
                                     headers={"Authorization": "Bearer test-glm-agent"}) as client:
            body = {"model": "GLM-5.3-Flash", "messages": [{"role": "user", "content": "weather?"}], "tools": tools}
            response = await client.post("/v1/chat/completions", json=body)
            assert response.status_code == 200, response.text
            message = response.json()["choices"][0]["message"]
            retry = await client.post("/v1/chat/completions", json=body)
            check("GLM HTTP retry preserves tool-call IDs without resampling",
                  retry.json()["choices"][0]["message"] == message and provider.calls == 1
                  and message["reasoning_content"] == "plan")
            body["messages"] += [message, {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": "sunny"}]
            response = await client.post("/v1/chat/completions", json=body)
            assert response.status_code == 200, response.text
            record = (await client.get("/sessions/glm")).json()
            turns = record["turns"]
            reconstructed = turns[0]["prompt_token_ids"] + turns[0]["completion_token_ids"] + turns[1]["prompt_token_ids"]
            check("recorded deltas retain exactly one observation header",
                  len(turns) == 2 and bundle.tokenizer.decode(reconstructed).count("<|observation|>") == 1
                  and response.json()["choices"][0]["message"]["content"] == "sunny")
            await client.delete("/sessions/glm")
            check("GLM deletion releases the stream", not provider.models.has_session("glm-agent"))


async def test_gpt_oss_tool_call_retry_recording() -> None:
    registry = ModelRegistry()
    model = "gpt-oss-20b"
    opening = [{"role": "user", "content": "Look up the value"}]
    tools = [{"type": "function", "function": {
        "name": "lookup", "description": "Look up a value",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
    }}]
    call = ('<|channel|>analysis<|message|>Need a lookup.<|end|>'
            '<|start|>assistant<|channel|>commentary<|message|>Checking.<|end|>'
            '<|start|>assistant to=functions.lookup<|channel|>commentary'
            '<|constrain|>json<|message|>{"key": "x"}<|call|>')
    final_text = ('<|channel|>analysis<|message|>Use the result.<|end|>'
             '<|start|>assistant<|channel|>final<|message|>The value is 42.<|return|>')

    class Provider(BaseRolloutProvider):
        def __init__(self):
            super().__init__()
            self.models = registry
            self.prompts, self.params = [], []

        async def _call_engine(self, prompt_ids, sampling_params, session_id):
            self.prompts.append(list(prompt_ids))
            self.params.append(dict(sampling_params))
            tok = self.models.resolve(model).tokenizer
            ids = tok.encode(call if len(self.prompts) == 1 else final_text, add_special_tokens=False)
            return ids, [-0.25] * len(ids), "stop", {"weight_version": "7"}

    provider = Provider()
    with TemporaryDirectory() as root:
        proxy = LLMProxyServer(api_key="test", completion_handler=make_completion_handler(provider),
                               on_session_deleted=provider.release_session, save_rollout_sessions=False, error_log_dir=root)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as http:
            async def chat(messages, **extra):
                return await http.post("/v1/chat/completions", headers={"Authorization": "Bearer test-trial-agent"},
                                       json={"model": model, "messages": messages, "tools": tools, **extra})
            first = await chat(opening, stop_token_ids=[42])
            assert first.status_code == 200, first.text
            first = first.json()
            message = first["choices"][0]["message"]
            check("HTTP response exposes tools and reasoning", first["choices"][0]["finish_reason"] == "tool_calls"
                  and message["reasoning_content"] == "Need a lookup." and message["content"] == "Checking.")
            repeated = await chat(opening)
            check("retry reuses tool IDs and does not sample again", repeated.json()["choices"] == first["choices"]
                  and len(provider.prompts) == 1)
            messages = opening + [message, {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": "42"}]
            final = await chat(messages, stream=True)
            check("SSE includes parsed final content", final.status_code == 200 and 'The value is 42.' in final.text
                  and '[DONE]' in final.text)
            check("required stops survive both default and explicit caller stops",
                  provider.params[0]["stop_token_ids"] == [200002, 199999, 200012, 42]
                  and provider.params[1]["stop_token_ids"] == [200002, 199999, 200012])
            record = (await http.get("/sessions/trial")).json()
            turns = record["turns"]
            reconstructed = [token for turn in turns for token in turn["prompt_token_ids"] + turn["completion_token_ids"]]
            check("record reconstructs the exact training stream and preserves raw reasoning",
                  len(turns) == 2 and reconstructed == provider.prompts[-1] + turns[-1]["completion_token_ids"]
                  and turns[0]["completion_text"] == call and turns[0]["weight_version"] == "7")
            await http.delete("/sessions/trial")
            check("deletion releases Harmony state", not provider.models.has_session("trial-agent"))


async def test_gpt_oss_literal_markers_recording() -> None:
    registry = ModelRegistry()
    tok = registry.resolve("gpt-oss-20b").tokenizer
    encode = lambda text: tok.encode(text, add_special_tokens=False)
    literal = lambda text: [token for char in text for token in encode(char)]
    example = ('<|start|>assistant to=functions.lookup<|channel|>commentary'
               '<|message|>{"key":"x"}<|call|>')
    reasoning = 'Consider this example <|end|> and keep reasoning: ' + example
    content = 'Print this example verbatim: ' + example
    sampled = encode('<|channel|>analysis<|message|>') + literal(reasoning)
    sampled += encode('<|end|><|start|>assistant<|channel|>final<|message|>')
    sampled += literal(content) + encode('<|return|>')

    class Provider(BaseRolloutProvider):
        def __init__(self):
            super().__init__()
            self.models = registry
            self.calls = 0

        async def _call_engine(self, prompt_ids, sampling_params, session_id):
            self.calls += 1
            return sampled, [-0.25] * len(sampled), "stop", {}

    provider = Provider()
    with TemporaryDirectory() as root:
        proxy = LLMProxyServer(api_key="test", completion_handler=make_completion_handler(provider),
                               save_rollout_sessions=False, error_log_dir=root)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test",
                                     headers={"Authorization": "Bearer test-literals-agent"}) as http:
            body = {"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "Explain Harmony"}]}
            for _ in range(2):
                response = await http.post("/v1/chat/completions", json=body)
                assert response.status_code == 200, response.text
                choice = response.json()["choices"][0]
                check("HTTP and retry retain literal markers without fabricated tool calls",
                      choice["finish_reason"] == "stop"
                      and choice["message"] == {"role": "assistant", "content": content, "reasoning_content": reasoning})
            streamed = await http.post("/v1/chat/completions", json={**body, "stream": True})
            assert streamed.status_code == 200, streamed.text
            events = [json.loads(line.removeprefix("data: ")) for line in streamed.text.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            delta = events[0]["choices"][0]["delta"]
            check("streaming preserves literal content and reasoning",
                  delta["content"] == content and delta["reasoning_content"] == reasoning and "tool_calls" not in delta)
            record = (await http.get("/sessions/literals")).json()
            check("recorded fields preserve token identity through retries",
                  provider.calls == 1 and len(record["turns"]) == 1
                  and record["turns"][0]["completion_token_ids"] == sampled
                  and record["turns"][0]["content"] == content
                  and record["turns"][0]["reasoning_content"] == reasoning)


async def main() -> None:
    await test_concurrent_duplicate_serialized()
    await test_disconnect_recovery()
    await test_normalized_opening_retry_recorded_once()
    await test_cancelled_handler_retry_races_original()
    await test_detached_pipeline_serializes_through_recording()
    await test_tool_choice_constraints_rejected_before_generation_or_replay()
    await test_tool_call_retry_identity()
    await test_glm_tool_call_retry_recording()
    await test_gpt_oss_tool_call_retry_recording()
    await test_gpt_oss_literal_markers_recording()
    await test_deleted_session_drops_late_turn()
    await test_unresolved_recording_failure_refuses_collection()
    await test_parser_failure_blocks_collection_until_retry()
    await test_recording_failure_exact_retry_recovers()
    print("PASS: disconnect recovery, retry deduplication, deletion cleanup, "
          "and failed-recording containment")


if __name__ == "__main__":
    asyncio.run(main())
