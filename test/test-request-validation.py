"""Regression tests for request admission and immutable conversation inputs."""

from __future__ import annotations
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import threading
from unittest.mock import patch
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from offline_common import FakeTokenizer, GatedProvider, ScriptedProvider, check
from proxyserver.config import load_config
from proxyserver.model_registry import ModelRegistry, resolve_profile, validate_sampling_overrides
from proxyserver.server import LLMProxyServer, make_completion_handler
from proxyserver.token_stream import TokenStreamError, as_token_ids
from proxyserver.tokenization.tool_parser import DecodedCompletion

MODEL = "Qwen3.5-9B"
OPENING = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {
    "name": "lookup", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
}}]
CHANGED_TOOLS = deepcopy(TOOLS)
CHANGED_TOOLS[0]["function"]["parameters"]["properties"]["key"]["type"] = "integer"


async def test_http_admission() -> None:
    provider = ScriptedProvider()
    with tempfile.TemporaryDirectory() as tmp:
        proxy = LLMProxyServer(
            api_key="test", key_delimiter="::", error_log_dir=tmp,
            completion_handler=make_completion_handler(provider), save_rollout_sessions=False,
            get_sampling_overrides=provider.get_sampling_overrides,
            set_sampling_overrides=provider.set_runtime_sampling_overrides,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app, raise_app_exceptions=False), base_url="http://test",
        ) as http:
            headers = {"Authorization": "Bearer test::admission::agent"}
            base = {"model": MODEL, "messages": OPENING}
            invalid = [None, [], {}, {**base, "messages": [None]}, {**base, "messages": []},
                       {**base, "messages": [{"role": "user", "content": 12}]},
                       {**base, "messages": [{"role": "unknown", "content": "hi"}]},
                       {**base, "messages": [{"role": "assistant", "audio": {"id": "audio_1"}}]},
                       {**base, "messages": [{"role": "assistant", "tool_calls": "bad"}]},
                       {**base, "messages": [{"role": "assistant", "tool_calls": [None]}]},
                       {**base, "messages": [{"role": "assistant", "tool_calls": [
                           {"type": "function", "function": "bad"},
                       ]}]}]
            for part in ({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                         {"type": "input_audio", "input_audio": {}}, {"type": "file", "file": {}},
                         {"type": "unknown"}):
                invalid.append({**base, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Do not lose this attachment"}, part,
                ]}]})
            for field, value in (
                ("stream", "false"), ("n", 2), ("n", True), ("response_format", {"type": "json_object"}),
                ("tool_choice", "required"), ("parallel_tool_calls", False), ("logprobs", True),
                ("top_logprobs", 2), ("logit_bias", {"42": 1}), ("stream_options", {"include_usage": False}),
                ("reasoning_effort", True), ("reasoning_effort", {}), ("reasoning_effort", "invalid"),
                ("modalities", ["audio"]), ("max_tokens", 0), ("max_completion_tokens", 1.5), ("seed", True),
                ("stop_token_ids", [-1]), ("temperature", -1), ("top_p", 0), ("min_p", 2), ("top_k", 1.5),
                ("stop", ""), ("tools", "lookup"),
                ("tools", [{"type": "function", "function": {"name": "lookup", "strict": True}}]),
            ):
                invalid.append({**base, field: value})
            for body in invalid:
                response = await http.post("/v1/chat/completions", headers=headers, content=json.dumps(body))
                assert response.status_code == 400, (body, response.status_code, response.text)
            for raw in ('{', json.dumps(base)[:-1] + ', "temperature": 1e309}'):
                response = await http.post("/v1/chat/completions", headers=headers, content=raw)
                assert response.status_code == 400, response.text
            assert provider.engine_calls == 0 and proxy.recorder.get_session("admission") is None
            check("malformed, multimodal, invalid sampling, and unsupported requests fail before admission", True)

            accepted = {**base, "n": 1, "response_format": {"type": "text"}, "parallel_tool_calls": True,
                        "tool_choice": "auto", "logprobs": False, "top_logprobs": 0, "logit_bias": {},
                        "stream_options": {"include_usage": True}, "metadata": {"trial": "one"}, "reasoning_effort": "high"}
            response = await http.post("/v1/chat/completions", headers=headers, json=accepted)
            assert response.status_code == 200, response.text
            for _ in range(6):  # More than the replay allowance: rejected requests must not consume it.
                response = await http.post("/v1/chat/completions", headers=headers, json={**base, "n": 2})
                assert response.status_code == 400, response.text
            response = await http.post("/v1/chat/completions", headers=headers, json={**base, "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ]})
            assert response.status_code == 200 and provider.engine_calls == 1, response.text
            check("supported defaults work; rejected retries do not consume the replay allowance", True)

            bare = {"Authorization": "Bearer test"}
            response = await http.put("/sampling_overrides", headers=bare, json={"temperature": 0.7})
            assert response.status_code == 200
            before = (await http.get("/sampling_overrides")).json()
            for raw in ('{"temperature": 1e309}', '{"temperature": NaN}', '{"top_p": Infinity}',
                        '{"temperature": -1}', '{"top_p": 0}', '{"top_k": 0}', '{"top_k": 2.5}',
                        '{"min_p": 2}', '{"repetition_penalty": 0}', '{"presence_penalty": 3}',
                        '{"frequency_penalty": -3}'):
                response = await http.put("/sampling_overrides", headers=bare, content=raw)
                assert response.status_code == 400, (raw, response.text)
                state = await http.get("/sampling_overrides")
                assert state.status_code == 200 and state.json() == before, state.text
            check("nonfinite and out-of-range runtime updates leave the active policy and GET endpoint intact", True)

            tools_headers = {"Authorization": "Bearer test::tools::agent"}
            response = await http.post("/v1/chat/completions", headers=tools_headers, json={**base, "tools": TOOLS})
            assert response.status_code == 200, response.text
            messages = OPENING + [response.json()["choices"][0]["message"], {"role": "user", "content": "next"}]
            calls = provider.engine_calls
            for tool_fields in ({"tools": CHANGED_TOOLS}, {"tools": []}, {"tools": None}, {}):
                response = await http.post("/v1/chat/completions", headers=tools_headers,
                                           json={**base, "messages": messages, **tool_fields})
                assert response.status_code == 400 and provider.engine_calls == calls, response.text
            response = await http.post("/v1/chat/completions", headers=tools_headers,
                                       json={**base, "messages": messages, "tools": TOOLS})
            assert response.status_code == 200 and provider.engine_calls == calls + 1, response.text
            check("changed or removed continuation schemas return 400; the original schema can still advance", True)
            response = await http.post("/v1/chat/completions", headers=headers,
                                       json={**base, "messages": OPENING + [
                                           {"role": "assistant", "content": "answer"},
                                           {"role": "user", "content": "next"},
                                       ], "tools": TOOLS})
            assert response.status_code == 400 and provider.engine_calls == calls + 1, response.text
            check("adding tools to a conversation opened without them also fails", True)


async def test_reasoning_effort_templates() -> None:
    class CapturingProvider(ScriptedProvider):
        def __init__(self):
            super().__init__()
            self.models = ModelRegistry()
            self.prompts = []

        async def _call_engine(self, prompt_ids, sampling_params, session_id):
            self.prompts.append(list(prompt_ids))
            return await super()._call_engine(prompt_ids, sampling_params, session_id)

    provider = CapturingProvider()
    with tempfile.TemporaryDirectory() as tmp:
        proxy = LLMProxyServer(
            api_key="test", key_delimiter="::", error_log_dir=tmp,
            completion_handler=make_completion_handler(provider), save_rollout_sessions=False,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=proxy.app, raise_app_exceptions=False), base_url="http://test",
        ) as http:
            for model in ("Qwen3.8-27B", "GLM-5.3-Flash", "gpt-oss-20b"):
                profile = resolve_profile(model)
                tokenizer = provider.models.resolve(model).tokenizer
                ending = profile.assistant_turn_endings[0] if profile.assistant_turn_endings else profile.assistant_turn_end
                provider.next_tokens = tokenizer.encode("answer" + ending, add_special_tokens=False)
                prompts = []
                for effort in profile.reasoning_efforts:
                    headers = {"Authorization": f"Bearer test::{profile.name}_{effort}::agent"}
                    base = {"model": model, "messages": OPENING, "reasoning_effort": effort}
                    calls = provider.engine_calls
                    response = await http.post("/v1/chat/completions", headers=headers, json=base)
                    assert response.status_code == 200, response.text
                    expected = as_token_ids(tokenizer.apply_chat_template(
                        OPENING, add_generation_prompt=False, tokenize=True, reasoning_effort=effort,
                    ), what="expected effort prompt") + tokenizer.encode(profile.generation_prompt, add_special_tokens=False)
                    assert provider.prompts[-1] == expected
                    prompts.append(expected)

                    changed = next(value for value in profile.reasoning_efforts if value != effort)
                    for _ in range(6):
                        rejected = await http.post("/v1/chat/completions", headers=headers,
                                                   json={**base, "reasoning_effort": changed})
                        assert rejected.status_code == 400, rejected.text
                    replay = await http.post("/v1/chat/completions", headers=headers, json=base)
                    assert replay.status_code == 200 and provider.engine_calls == calls + 1, replay.text
                    assert replay.json()["choices"] == response.json()["choices"]

                    continuation = {**base, "messages": OPENING + [response.json()["choices"][0]["message"],
                                                                   {"role": "user", "content": "next"}]}
                    rejected = await http.post("/v1/chat/completions", headers=headers,
                                               json={**continuation, "reasoning_effort": changed})
                    assert rejected.status_code == 400 and provider.engine_calls == calls + 1, rejected.text
                    continued = await http.post("/v1/chat/completions", headers=headers, json=continuation)
                    assert continued.status_code == 200, continued.text
                    committed = expected + provider.next_tokens
                    assert provider.prompts[-1][:len(committed)] == committed
                    delta = tokenizer.decode(provider.prompts[-1][len(committed):], skip_special_tokens=False)
                    assert "next" in delta and "Reasoning" not in delta, delta

                    # A different opening can choose another effort on the same stream.
                    fresh = {**base, "messages": [{"role": "user", "content": "fresh"}], "reasoning_effort": changed}
                    response = await http.post("/v1/chat/completions", headers=headers, json=fresh)
                    assert response.status_code == 200, response.text
                    assert provider.prompts[-1] == as_token_ids(tokenizer.apply_chat_template(
                        fresh["messages"], add_generation_prompt=False, tokenize=True, reasoning_effort=changed,
                    ), what="expected new conversation") + tokenizer.encode(profile.generation_prompt, add_special_tokens=False)
                assert len({tuple(prompt) for prompt in prompts}) == len(profile.reasoning_efforts)

                headers = {"Authorization": f"Bearer test::{profile.name}_default::agent"}
                calls = provider.engine_calls
                rejected = await http.post("/v1/chat/completions", headers=headers,
                                           json={"model": model, "messages": OPENING, "reasoning_effort": "minimal"})
                assert rejected.status_code == 400 and provider.engine_calls == calls, rejected.text
                assert proxy.recorder.get_session(f"{profile.name}_default") is None
                response = await http.post("/v1/chat/completions", headers=headers,
                                           json={"model": model, "messages": OPENING})
                assert response.status_code == 200, response.text
                expected = as_token_ids(tokenizer.apply_chat_template(
                    OPENING, add_generation_prompt=False, tokenize=True,
                ), what="expected default prompt") + tokenizer.encode(profile.generation_prompt, add_special_tokens=False)
                assert provider.prompts[-1] == expected
                replay = await http.post("/v1/chat/completions", headers=headers,
                                         json={"model": model, "messages": OPENING,
                                               "reasoning_effort": profile.default_reasoning_effort})
                assert replay.status_code == 200 and provider.engine_calls == calls + 1, replay.text
            # An engine rejection before commit must not poison a different
            # effort's prompt, even when its messages are identical.
            headers = {"Authorization": "Bearer test::effort_rejection::agent"}
            base = {"model": "gpt-oss-20b", "messages": OPENING, "reasoning_effort": "low"}
            request = httpx.Request("POST", "http://engine/generate")
            error = httpx.HTTPStatusError("rejected", request=request,
                                         response=httpx.Response(400, request=request))
            with patch.object(provider, "_call_engine", side_effect=error) as engine:
                for _ in range(3):
                    rejected = await http.post("/v1/chat/completions", headers=headers, json=base)
                    assert rejected.status_code == 400, rejected.text
                assert engine.await_count == 2
                rejected = await http.post("/v1/chat/completions", headers=headers,
                                           json={**base, "reasoning_effort": "high"})
                assert rejected.status_code == 400 and engine.await_count == 3, rejected.text
            response = await http.post("/v1/chat/completions", headers=headers,
                                       json={**base, "reasoning_effort": "high"})
            assert response.status_code == 200, response.text
    check("reasoning effort reaches bundled templates, preserves continuations, and validates retries", True)


def test_config_sampling() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.yaml"
        for value in (".inf", ".nan", "-0.1"):
            path.write_text(f"sampling_overrides:\n  temperature: {value}\n")
            try:
                load_config(path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid sampling policy loaded: {value}")
    validate_sampling_overrides({"temperature": 0, "top_k": -1, "top_p": 1, "min_p": 0,
                                 "repetition_penalty": 1, "presence_penalty": -2, "frequency_penalty": 2}, "boundaries")
    check("invalid YAML sampling policies fail at startup; valid boundary values remain accepted", True)


async def test_tools_and_parse_recovery() -> None:
    provider = ScriptedProvider()
    tools = deepcopy(TOOLS)
    first = await provider.generate(OPENING, "schema", model=MODEL, tools=tools)
    tools[0]["function"]["name"] = "mutated_after_commit"
    # A parser failure after commit must be recoverable using the original
    # prompt schema, even if the caller changes tools on the retry.
    seen = []

    async def parse(ids, tools=None, completion_text=None):
        seen.append(deepcopy(tools))
        if len(seen) == 1:
            raise RuntimeError("transient parser failure")
        return DecodedCompletion(content="answer", reasoning_content=None, tool_calls=[])

    parser = provider.models.resolve(MODEL).tool_parser
    with patch.object(parser, "extract_completion", side_effect=parse):
        try:
            await provider.parse_completion("schema", first[1], first[4], first[3], tools=CHANGED_TOOLS)
        except RuntimeError:
            pass
        else:
            raise AssertionError("parser failure was not exercised")
        replay = await provider.generate(OPENING, "schema", model=MODEL, tools=CHANGED_TOOLS)
        await provider.parse_completion("schema", replay[1], replay[4], replay[3], tools=CHANGED_TOOLS)
    assert seen == [TOOLS, TOOLS] and provider.engine_calls == 1
    messages = OPENING + [{"role": "assistant", "content": first[4]}, {"role": "user", "content": "next"}]
    second = await provider.generate(messages, "schema", model=MODEL, tools=TOOLS)
    assert second[0][:len(first[0] + first[1])] == first[0] + first[1]
    fresh = [{"role": "user", "content": "A new conversation"}]
    await provider.generate(fresh, "schema", model=MODEL, tools=CHANGED_TOOLS)
    await provider.generate(fresh + [{"role": "assistant", "content": "answer"}, {"role": "user", "content": "next"}],
                            "schema", model=MODEL, tools=CHANGED_TOOLS)
    check("tool schemas are copied, retries repair parsing with the original schema, and new conversations can change tools", True)


async def test_concurrent_model_pin() -> None:
    provider = ScriptedProvider()
    first_loading = threading.Event()
    second_loading = threading.Event()

    def load(path):
        if "Llama" in path:
            second_loading.set()
        else:
            first_loading.set()
            # Before the fix both first calls entered model resolution. This
            # gate lets that race complete, while serialized resolution times
            # out and then rejects the second model against the first pin.
            second_loading.wait(timeout=0.5)
        return FakeTokenizer()

    provider.models._tokenizer_loader = load
    first = asyncio.create_task(provider.generate(OPENING, "race", model=MODEL))
    assert await asyncio.to_thread(first_loading.wait, 2)
    second = asyncio.create_task(provider.generate(OPENING, "race", model="Llama-3.1-8B-Instruct"))
    results = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 5)
    assert isinstance(results[0], tuple), results
    assert isinstance(results[1], TokenStreamError) and "pinned" in str(results[1]), results
    assert provider.engine_calls == 1 and not second_loading.is_set()
    replay = await provider.generate(OPENING, "race", model=MODEL)
    assert replay == results[0] and provider.engine_calls == 1
    check("concurrent direct first calls install one model pin and preserve its replay", True)


async def test_effective_generation_parameters() -> None:
    for params in ({"temperature": float("inf")}, {"top_k": 0}, {"top_p": 0},
                   {"max_tokens": True}, {"seed": False}, {"stop_token_ids": [True]},
                   {"stop": ""}):
        provider = ScriptedProvider()
        try:
            await provider.generate(OPENING, "invalid", model=MODEL, **params)
        except TokenStreamError:
            pass
        else:
            raise AssertionError(f"invalid effective parameters reached generation: {params}")
        assert provider.engine_calls == 0
        assert provider.models.resolve(MODEL).token_streams.committed_turn("invalid", []) is None
    provider = ScriptedProvider()
    provider.models.resolve(MODEL).sampling_defaults["temperature"] = float("nan")
    try:
        await provider.generate(OPENING, "defaults", model=MODEL)
    except TokenStreamError:
        pass
    else:
        raise AssertionError("invalid bundled default reached generation")
    assert provider.engine_calls == 0
    provider.set_runtime_sampling_overrides({"temperature": 0})
    result = await provider.generate(OPENING, "defaults", model=MODEL, temperature=0.9)
    assert result[-1]["sampling_params"]["temperature"] == 0
    check("effective parameters are validated before the engine; runtime precedence and greedy decoding survive", True)


async def test_stream_concurrency_and_loading_cancellation() -> None:
    provider = GatedProvider()
    provider.gate = asyncio.Event()
    tools = deepcopy(TOOLS)
    first = asyncio.create_task(provider.generate(OPENING, "same", model=MODEL, tools=tools))
    assert await asyncio.wait_for(provider.started.get(), 2) == "same"
    tools[0]["function"]["name"] = "mutated_during_generation"
    retry = asyncio.create_task(provider.generate(OPENING, "same", model=MODEL, tools=CHANGED_TOOLS))
    other = asyncio.create_task(provider.generate(OPENING, "other", model=MODEL, tools=CHANGED_TOOLS))
    try:
        assert await asyncio.wait_for(provider.started.get(), 2) == "other"
        assert provider.engine_calls == 2
    finally:
        provider.gate.set()
        results = await asyncio.wait_for(asyncio.gather(first, retry, other), 5)
    assert results[0] == results[1] and provider.engine_calls == 2
    streams = provider.models.resolve(MODEL).token_streams
    assert streams.committed_turn("same", results[0][1]).tools == TOOLS
    assert streams.committed_turn("other", results[2][1]).tools == CHANGED_TOOLS

    provider = ScriptedProvider()
    loading, release = threading.Event(), threading.Event()

    def load(path):
        loading.set()
        assert release.wait(5), "test did not release tokenizer loading"
        return FakeTokenizer()

    provider.models._tokenizer_loader = load
    first = asyncio.create_task(provider.generate(OPENING, "loading", model=MODEL))
    retry = None
    try:
        assert await asyncio.to_thread(loading.wait, 2)
        first.cancel()
        await asyncio.sleep(0)  # Deliver cancellation while the loader is blocked.
        assert not first.done() and provider._session_lock("loading").locked()
        retry = asyncio.create_task(provider.generate(OPENING, "loading", model=MODEL))
        await asyncio.sleep(0)
        assert provider.engine_calls == 0 and not retry.done()
    finally:
        release.set()
        results = await asyncio.wait_for(asyncio.gather(
            first, *([retry] if retry is not None else []), return_exceptions=True,
        ), 5)
    assert isinstance(results[0], asyncio.CancelledError) and isinstance(results[1], tuple), results
    assert provider.engine_calls == 1 and not provider._session_lock("loading").locked()
    check("same-stream retries serialize, other streams progress, and cancellation drains model loading under the lock", True)


async def main() -> None:
    test_config_sampling()
    await test_http_admission()
    await test_reasoning_effort_templates()
    await test_tools_and_parse_recovery()
    await test_concurrent_model_pin()
    await test_effective_generation_parameters()
    await test_stream_concurrency_and_loading_cancellation()
    print("PASS: request validation, policy updates, tool schema consistency, and concurrent model pinning")


if __name__ == "__main__":
    asyncio.run(main())
