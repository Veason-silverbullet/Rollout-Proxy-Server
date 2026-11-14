"""Offline HTTP compaction, routing, and retry contracts. No inference server."""
import asyncio
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from copy import deepcopy
from uuid import uuid4
import httpx
from offline_common import RecordingProvider, check
from proxyserver.service import LLMProxyServer, make_completion_handler, stream_id
from proxyserver.rollout_base import BaseRolloutProvider
from proxyserver.token_stream import as_token_ids
from proxyserver.record_validation import validate_record_structure
from proxyserver.training import record_to_streams


def parse_record(record):
    return record_to_streams(record, roles={"agent": "agent"})


class RoutedProvider(RecordingProvider):
    def __init__(self):
        super().__init__()
        self.priors = []
        self.fail_next = False

    async def _call_engine(self, prompt_ids, sampling_params, session_id, routed_experts_prior=None):
        self.priors.append(routed_experts_prior)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("transient engine failure")
        tokens, probs, stop, meta = await super()._call_engine(prompt_ids, sampling_params, session_id)
        start = routed_experts_prior[0] if routed_experts_prior else 0
        rows = len(prompt_ids) + len(tokens) - 1 - start
        meta["routed_experts"] = dict(start=start, rows=rows, cols=1, dtype="uint8",
                                      data=base64.b64encode(bytes([self.engine_calls]) * rows).decode())
        return tokens, probs, stop, meta


async def test_retained_tool_history():
    """Real templates must render wire-format calls without changing replay identity."""
    class RealTokenizerProvider(BaseRolloutProvider):
        def __init__(self):
            super().__init__()
            self.prompts = []

        async def _call_engine(self, prompt_ids, sampling_params, session_id, **kwargs):
            self.prompts.append(list(prompt_ids))
            tokenizer = self.models.resolve(self._session_models[session_id]).tokenizer
            tokens = tokenizer.encode("Done.", add_special_tokens=False)
            return tokens, [-0.1] * len(tokens), "stop", {}

    provider = RealTokenizerProvider()
    proxy = LLMProxyServer(api_key="test", save_rollout_sessions=False,
                           completion_handler=make_completion_handler(provider))
    tools = [{"type": "function", "function": {
        "name": "lookup", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "options": {"type": "object"},
        }},
    }}]
    models = ("Qwen3.5-9B", "Qwen3.6-27B", "Qwen3.8-27B", "GLM-5.3-Flash",
              "Llama-3.1-8B-Instruct", "gpt-oss-20b")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
        for model in models:
            bundle = provider.models.resolve(model)
            tokenizer, manager = bundle.tokenizer, bundle.token_streams

            async def chat(sid, messages, segment):
                # Raw JSON also exercises non-finite values in object arguments;
                # httpx's json= encoder rejects those before they reach the proxy.
                return await client.post("/v1/chat/completions", content=json.dumps({
                    "model": model, "messages": messages, "tools": tools,
                    "rollout": {"segment_index": segment},
                }), headers={"Authorization": f"Bearer test|{sid}|agent", "Content-Type": "application/json"})

            for segment in (0, 1):
                for arguments in ({}, {"city": "上海", "options": {"tags": ["a", "b"], "enabled": True,
                    "count": 2, "value": None, "finite": 1e308, "large_integer": 10**100,
                    "literal": "NaN Infinity"}}):
                    for as_string in (True, False):
                        sid = uuid4().hex
                        if segment:
                            response = await chat(sid, [{"role": "user", "content": "Start."}], 0)
                            assert response.status_code == 200, response.text
                        messages = [
                            {"role": "system", "content": "Summary of earlier work."},
                            {"role": "user", "content": "Look up the city."},
                            {"role": "assistant", "content": None, "tool_calls": [{
                                "id": "call_1", "type": "function", "function": {
                                    "name": "lookup", "arguments": json.dumps(arguments) if as_string else arguments,
                                },
                            }]},
                            {"role": "tool", "tool_call_id": "call_1", "name": "lookup", "content": "Found it."},
                            {"role": "user", "content": "Continue."},
                        ]
                        original = deepcopy(messages)
                        # Independent oracle: pass an object directly to the bundled template.
                        template_messages = deepcopy(messages)
                        template_messages[2]["content"] = ""
                        template_messages[2]["tool_calls"][0]["function"]["arguments"] = arguments
                        expected = as_token_ids(tokenizer.apply_chat_template(
                            template_messages, tools=tools, tokenize=True, add_generation_prompt=False,
                        ), what="retained tool history") + tokenizer.encode(manager.generation_prompt, add_special_tokens=False)
                        response = await chat(sid, messages, segment)
                        assert response.status_code == 200, (model, segment, response.text)
                        assert provider.prompts[-1] == expected, model
                        record = proxy.recorder.dump_session(sid)
                        assert record["turns"][-1]["request_messages"] == original
                        calls = len(provider.prompts)
                        retry = await chat(sid, messages, segment)
                        assert retry.status_code == 200 and retry.json()["choices"] == response.json()["choices"]
                        assert len(provider.prompts) == calls
                        assert proxy.recorder.dump_session(sid) == record
                        # Equivalent renderings must not become equivalent replay identities.
                        changed = deepcopy(messages)
                        changed[2]["tool_calls"][0]["function"]["arguments"] = arguments if as_string else json.dumps(arguments)
                        assert manager.cached_turn(stream_id(sid, "agent"), changed, segment_index=segment) is None
                        assert messages == original

                        continuation = original + [response.json()["choices"][0]["message"],
                                                   {"role": "user", "content": "Next step."}]
                        for path, value in (
                            ((2, "tool_calls", 0, "function", "arguments"),
                             '{"city":"changed"}' if as_string else {"city": "changed"}),
                            ((2, "tool_calls", 0, "function", "name"), "changed_lookup"),
                            ((2, "tool_calls", 0, "id"), "changed_call"),
                            ((2, "tool_calls"), []),
                            ((2, "reasoning_content"), "Changed retained reasoning."),
                            ((2, "thinking"), "Changed retained thinking."),
                            ((3, "tool_call_id"), "changed_call"),
                            ((3, "name"), "changed_lookup"),
                        ):
                            edited = deepcopy(continuation)
                            target = edited
                            for key in path[:-1]:
                                target = target[key]
                            target[path[-1]] = value
                            rejected = await chat(sid, edited, segment)
                            assert rejected.status_code == 400, (model, segment, path, rejected.text)
                            assert "history was modified" in rejected.json()["error"]
                            assert len(provider.prompts) == calls
                            assert proxy.recorder.dump_session(sid) == record

                        # Rejections leave the original history usable, with exact sampled tokens.
                        sampled = record["turns"][-1]["completion_token_ids"]
                        following = await chat(sid, continuation, segment)
                        assert following.status_code == 200, (model, segment, following.text)
                        assert len(provider.prompts) == calls + 1
                        prefix = expected + sampled
                        assert provider.prompts[-1][:len(prefix)] == prefix

            # Invalid retained arguments fail before inference and leave the opening retryable.
            invalid_arguments = ["{", "", "[]", "null", "1", "true", '"text"']
            invalid_arguments.extend('{"options":{"values":[' + value + ']}}'
                                     for value in ("NaN", "Infinity", "-Infinity", "1e309", "-1e309"))
            invalid_arguments.extend({"options": {"values": [float(value)]}}
                                     for value in ("nan", "inf", "-inf"))
            integer_limit = sys.get_int_max_str_digits()
            if integer_limit:
                invalid_arguments.append('{"count":' + "9" * (integer_limit + 1) + '}')
            for segment in (0, 1):
                sid = uuid4().hex
                if segment:
                    response = await chat(sid, [{"role": "user", "content": "Start."}], 0)
                    assert response.status_code == 200, response.text
                before = proxy.recorder.dump_session(sid)
                for value in invalid_arguments:
                    invalid = deepcopy(messages)
                    invalid[2]["tool_calls"][0]["function"]["arguments"] = value
                    calls = len(provider.prompts)
                    response = await chat(sid, invalid, segment)
                    assert response.status_code == 400, (model, value, response.text)
                    assert "arguments" in response.json()["error"]
                    assert len(provider.prompts) == calls
                    assert manager.segment_context(stream_id(sid, "agent"), segment) == (segment, True)
                    recorded = proxy.recorder.dump_session(sid)
                    assert recorded["turns"] == (before["turns"] if before else [])
                response = await chat(sid, messages, segment)
                assert response.status_code == 200, response.text
            check(f"{model}: retained calls render exactly; recordings, retries, and invalid-argument handling stay correct", True)

            if model in ("gpt-oss-20b", "Qwen3.5-9B"):
                calls = [{"id": f"call_{name}", "type": "function", "function": {
                    "name": name, "arguments": '{"city":"Paris"}',
                }} for name in ("lookup", "lookup_again")]
                for segment in (0, 1):
                    for named_results in (False, True):
                        sid = uuid4().hex
                        if segment:
                            assert (await chat(sid, [{"role": "user", "content": "Start."}], 0)).status_code == 200
                        results = [{"role": "tool", "tool_call_id": call["id"], "content": "Found it.",
                                    **({"name": call["function"]["name"]} if named_results else {})}
                                   for call in calls]
                        parallel = [
                            {"role": "user", "content": "Run both lookups."},
                            {"role": "assistant", "content": None, "tool_calls": calls},
                            *results, {"role": "user", "content": "Continue."},
                        ]
                        before = proxy.recorder.dump_session(sid)
                        engine_calls = len(provider.prompts)
                        response = await chat(sid, parallel, segment)
                        if model != "gpt-oss-20b":
                            assert response.status_code == 200, response.text
                            rendered = tokenizer.decode(provider.prompts[-1], skip_special_tokens=False)
                            assert "<function=lookup>" in rendered and "<function=lookup_again>" in rendered
                            continue
                        assert response.status_code == 400, response.text
                        assert "messages[1].tool_calls" in response.json()["error"]
                        assert "retained assistant message" in response.json()["error"]
                        assert len(provider.prompts) == engine_calls
                        assert manager.segment_context(stream_id(sid, "agent"), segment) == (segment, True)
                        record = proxy.recorder.dump_session(sid)
                        assert record["turns"] == (before["turns"] if before else [])

                        # Correct the rejected opening without changing its segment.
                        sequential = [parallel[0]]
                        for call, result in zip(calls, results):
                            sequential.extend([{"role": "assistant", "content": None, "tool_calls": [call]}, result])
                        sequential.append(parallel[-1])
                        response = await chat(sid, sequential, segment)
                        assert response.status_code == 200, response.text
                        rendered = tokenizer.decode(provider.prompts[-1], skip_special_tokens=False)
                        for name in ("lookup", "lookup_again"):
                            assert f"<|start|>assistant to=functions.{name}<|channel|>" in rendered
                            assert f"<|start|>functions.{name} to=assistant" in rendered
                        retry = await chat(sid, sequential, segment)
                        assert retry.status_code == 200 and retry.json()["choices"] == response.json()["choices"]
                        assert len(provider.prompts) == engine_calls + 1
                check(f"{model}: retained parallel-call limits are profile-specific and rejected openings stay retryable", True)


async def test_retained_argument_types():
    provider = RecordingProvider()
    proxy = LLMProxyServer(api_key="test", save_rollout_sessions=False,
                           completion_handler=make_completion_handler(provider))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
        for segment in (0, 1):
            for original_value, edited_value in ((True, 1), (False, 0), (1, True), (0, False),
                                                 (2, 2.0), (2.0, 2)):
                sid = uuid4().hex

                async def chat(messages, index=segment):
                    return await client.post("/v1/chat/completions", json={
                        "model": "Qwen3.5-9B", "messages": messages,
                        "rollout": {"segment_index": index},
                    }, headers={"Authorization": f"Bearer test|{sid}|agent"})

                if segment:
                    assert (await chat([{"role": "user", "content": "Start."}], 0)).status_code == 200
                messages = [
                    {"role": "user", "content": "Look up the value."},
                    {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "call_1", "type": "function", "function": {
                            "name": "lookup", "arguments": {"options": [{"value": original_value}]},
                        },
                    }]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "Found it."},
                ]
                first = await chat(messages)
                assert first.status_code == 200, first.text
                record = proxy.recorder.dump_session(sid)
                calls = provider.engine_calls
                edited = deepcopy(messages)
                edited[1]["tool_calls"][0]["function"]["arguments"]["options"][0]["value"] = edited_value
                suffix = [first.json()["choices"][0]["message"], {"role": "user", "content": "Continue."}]
                for changed in (edited, edited + suffix):
                    rejected = await chat(changed)
                    assert rejected.status_code == 400, (original_value, edited_value, rejected.text)
                    assert provider.engine_calls == calls
                    assert proxy.recorder.dump_session(sid) == record
                retry = await chat(messages)
                assert retry.status_code == 200 and retry.json()["choices"] == first.json()["choices"]
                assert provider.engine_calls == calls
                following = await chat(messages + suffix)
                assert following.status_code == 200, following.text
                assert provider.engine_calls == calls + 1
    check("nested argument type edits reject retries and continuations without consuming the original turn", True)


async def test_eviction_survives_further_cache_pressure():
    for capacity in (1, 2):
        provider = RecordingProvider()
        manager = provider.models.resolve("Qwen3.5-9B").token_streams
        manager.max_sessions = capacity
        proxy = LLMProxyServer(api_key="test", save_rollout_sessions=False,
                               completion_handler=make_completion_handler(provider),
                               on_session_deleted=provider.release_session)
        sid = "a" * 32
        others = [f"{i + 1:032x}" for i in range(2 * capacity)]
        opening = [{"role": "user", "content": "solve"}]
        admin = {"Authorization": "Bearer test"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            async def chat(session, messages, segment=None):
                body = {"model": "Qwen3.5-9B", "messages": messages}
                if segment is not None:
                    body["rollout"] = {"segment_index": segment}
                return await client.post("/v1/chat/completions", json=body,
                    headers={"Authorization": f"Bearer test|{session}|agent"})

            first = await chat(sid, opening, 0)
            assert first.status_code == 200, first.text
            original = proxy.recorder.dump_session(sid)
            # Keep the original stream idle through repeated evictions.
            for other in others:
                response = await chat(other, opening, 0)
                assert response.status_code == 200, response.text
            assert not manager.has_session(stream_id(sid, "agent"))
            calls = provider.engine_calls
            continuation = opening + [first.json()["choices"][0]["message"], {"role": "user", "content": "continue"}]
            for messages, segment in ((continuation, 0), (continuation, None), (opening, 0), (opening, 1)):
                response = await chat(sid, messages, segment)
                assert response.status_code == 400, response.text
                assert "evicted" in response.text
            assert provider.engine_calls == calls
            assert proxy.recorder.dump_session(sid) == original

            response = await client.post(f"/sessions/{sid}/finalize", headers=admin)
            assert response.status_code == 200, response.text
            response = await client.get(f"/sessions/{sid}", headers=admin)
            assert response.status_code == 200, response.text
            streams = parse_record(response.json())
            assert len(streams) == 1
            assert streams[0].tokens == provider.prompts[0] + provider.next_tokens

            for session in (sid, *others):
                response = await client.delete(f"/sessions/{session}", headers=admin)
                assert response.status_code == 200, response.text
                assert stream_id(session, "agent") not in provider._session_models
            assert not manager._sessions and not manager._evicted
            check(f"cache pressure at capacity {capacity} cannot revive evicted streams; deletion releases markers", True)


async def test_commit_survives_eviction():
    class GatedProvider(RoutedProvider):
        def __init__(self):
            super().__init__()
            self.gate_stream = None
            self.started = asyncio.Event()
            self.resume = asyncio.Event()

        async def _call_engine(self, prompt_ids, sampling_params, session_id, routed_experts_prior=None):
            result = await super()._call_engine(prompt_ids, sampling_params, session_id, routed_experts_prior)
            if session_id == self.gate_stream:
                self.started.set()
                await self.resume.wait()
            return result

    # Explicit continuations and new compacted segments.
    for index in (0, 1):
        provider = GatedProvider()
        manager = provider.models.resolve("Qwen3.5-9B").token_streams
        manager.max_sessions = 1
        proxy = LLMProxyServer(api_key="test", save_rollout_sessions=False,
                               completion_handler=make_completion_handler(provider),
                               on_session_deleted=provider.release_session)
        sid = "a" * 32
        agent_stream = stream_id(sid, "agent")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            async def chat(session, messages, segment=0):
                body = {"model": "Qwen3.5-9B", "messages": messages,
                        "rollout": {"segment_index": segment}}
                return await client.post("/v1/chat/completions", json=body,
                    headers={"Authorization": f"Bearer test|{session}|agent"})

            opening = [{"role": "user", "content": "solve"}]
            first = await chat(sid, opening, 0)
            assert first.status_code == 200, first.text
            messages = opening + [first.json()["choices"][0]["message"], {"role": "user", "content": "continue"}]
            if index == 1:
                messages = [{"role": "system", "content": "Compacted memory"}, *messages[1:]]
            provider.gate_stream = agent_stream
            pending = asyncio.create_task(chat(sid, messages, index))
            try:
                await asyncio.wait_for(provider.started.wait(), 5)
                # The completion exists, but another rollout evicts its stream
                # before generation returns to commit/record it.
                other = await chat("b" * 32, opening, 0)
                assert other.status_code == 200, other.text
                assert not manager.has_session(agent_stream)
                assert agent_stream in provider._session_models
                assert provider.engine_calls == 3
            finally:
                provider.resume.set()
                response = await asyncio.wait_for(pending, 5)
            assert response.status_code == 200, response.text
            retry = await chat(sid, messages, index)
            assert retry.status_code == 200, retry.text
            assert retry.json()["choices"] == response.json()["choices"]
            assert provider.engine_calls == 3
            data = proxy.recorder.dump_session(sid)
            assert [turn["segment_index"] for turn in data["turns"]] == [0, index]
            validate_record_structure(data, require_logprobs=True, require_routed_experts=True)
            streams = parse_record(data)
            if index == 1:
                assert streams[0].tokens == provider.prompts[0] + provider.next_tokens
            assert streams[-1].tokens == provider.prompts[1] + provider.next_tokens
            assert manager.has_session(agent_stream)

            # Eviction BEFORE prompt construction must still reject the request
            # without sampling. Committing an admitted turn is the exception.
            assert (await chat("c" * 32, opening, 0)).status_code == 200
            calls = provider.engine_calls
            assert (await chat(sid, messages, index)).status_code == 400
            assert provider.engine_calls == calls
            check(f"in-flight segment {index!r} survives eviction; retry and routes remain exact", True)


async def main():
    provider = RoutedProvider()
    proxy = LLMProxyServer(api_key="test", save_rollout_sessions=False, completion_handler=make_completion_handler(provider))
    sid = "a" * 32
    headers = {"Authorization": f"Bearer test|{sid}|agent"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
        async def chat(messages, index, purpose="primary"):
            return await client.post("/v1/chat/completions", headers=headers, json={
                "model": "Qwen3.5-9B", "messages": messages,
                "rollout": {"segment_index": index, "purpose": purpose},
            })

        first = [{"role": "user", "content": "solve"}]
        response = await chat(first, 0)
        assert response.status_code == 200, response.text
        frozen = [{"role": "system", "content": "Summary of previous work"},
                  {"role": "assistant", "content": "retained old action"},
                  {"role": "user", "content": "continue"}]
        assert (await chat(frozen, 0)).status_code == 400
        assert (await chat(frozen, 2)).status_code == 400
        assert (await chat(frozen, None)).status_code == 400
        implicit_reset = await client.post("/v1/chat/completions", headers=headers, json={
            "model": "Qwen3.5-9B", "messages": [{"role": "user", "content": "replacement context"}],
        })
        assert implicit_reset.status_code == 400
        assert provider.engine_calls == 1
        provider.fail_next = True
        assert (await chat(frozen, 1)).status_code == 500
        assert len(proxy.recorder.dump_session(sid)["turns"]) == 1
        response = await chat(frozen, 1)
        assert response.status_code == 200, response.text
        assert provider.priors[-1] is None
        compacted_prompt = provider.prompts[-1]
        assert compacted_prompt[:6] == [100, 101, 102, 103, 104, 105]
        assert provider.next_tokens[0] not in compacted_prompt
        retry = await chat(frozen, 1)
        assert retry.status_code == 200, retry.text
        assert provider.engine_calls == 2
        assert len(proxy.recorder.dump_session(sid)["turns"]) == 2
        assert (await chat(frozen, 1, "compaction")).status_code == 400
        check("explicit boundary, frozen history, failed generation and opening retry", True)

        continuation = frozen + [response.json()["choices"][0]["message"], {"role": "user", "content": "summarize"}]
        response = await chat(continuation, 1, "compaction")
        assert response.status_code == 200, response.text
        assert provider.prompts[-1][:len(compacted_prompt) + 3] == compacted_prompt + provider.next_tokens
        assert provider.priors[-1] == (len(compacted_prompt) + 2, 1)
        followup = continuation + [response.json()["choices"][0]["message"], {"role": "user", "content": "finish"}]
        response = await chat(followup, 1)
        assert response.status_code == 200, response.text
        data = proxy.recorder.dump_session(sid)
        validate_record_structure(data, require_logprobs=True, require_routed_experts=True)
        streams = parse_record(data)
        assert [(s.agent_id, s.segment_index) for s in streams] == [("agent", 0), ("agent", 1)]
        assert sum(streams[0].loss_mask) == 3
        assert sum(streams[1].loss_mask) == 6
        assert streams[1].loss_mask[streams[1].completion_ends[1] - 3:streams[1].completion_ends[1]] == [0, 0, 0]
        assert all(s.routed_experts["rows"] == len(s.tokens) - 1 for s in streams)
        assert data["segment_routed_experts"]["agent"][0]["data"] != data["segment_routed_experts"]["agent"][1]["data"]
        assert len([stream for stream in streams if any(stream.loss_mask)]) == 2
        check("summary masks, exact continuation tokens and independent routing captures", True)

        # Same frozen prompt at the next index is a new generation, not a retry.
        record_completion = proxy.recorder.record_completion
        def fail_record(**kwargs):
            raise RuntimeError("transient recorder failure")
        proxy.recorder.record_completion = fail_record
        assert (await chat(frozen, 2)).status_code == 500
        calls = provider.engine_calls
        proxy.recorder.record_completion = record_completion
        response = await chat(frozen, 2)
        assert response.status_code == 200, response.text
        assert provider.engine_calls == calls
        data = proxy.recorder.dump_session(sid)
        assert [t["segment_index"] for t in data["turns"]] == [0, 1, 1, 1, 2]
        validate_record_structure(data, require_routed_experts=True)
        assert (await chat(first, 0)).status_code == 400
        check("recording-failure retry repairs the same segment; stale indices fail", True)

        # Initial frozen histories start at segment 0 and may be summary-only.
        headers["Authorization"] = f"Bearer test|{'b' * 32}|agent"
        response = await chat(frozen, 0, "compaction")
        assert response.status_code == 200, response.text
        summary_streams = parse_record(proxy.recorder.dump_session("b" * 32))
        assert len(summary_streams) == 1 and not any(summary_streams[0].loss_mask)
        check("summary-only segments produce no training sample", True)


if __name__ == "__main__":
    asyncio.run(test_retained_argument_types())
    asyncio.run(test_retained_tool_history())
    asyncio.run(main())
    asyncio.run(test_eviction_survives_further_cache_pressure())
    asyncio.run(test_commit_survives_eviction())
