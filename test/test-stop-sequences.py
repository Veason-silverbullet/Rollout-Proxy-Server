"""Text stops affect responses, never the sampled trajectory (offline HTTP).

Run: HF_HUB_OFFLINE=1 python3 test/test-stop-sequences.py
"""

import asyncio
import httpx
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxyserver.model_registry import ModelRegistry
from proxyserver.rollout_provider import DirectRolloutProvider, SGLangRolloutProvider, VLLMRolloutProvider
from proxyserver.service import LLMProxyServer, make_completion_handler
from offline_common import check

MODEL = "Qwen3.5-9B"
MESSAGES = [{"role": "user", "content": "Continue"}]
MISSING = object()


async def exercise(transport, engine, tokenizer, model=MODEL):
    calls = []

    async def upstream(request):
        if request.method == "GET":
            return httpx.Response(200, json={
                "data": [{"id": model, "max_model_len": 32768}],
                "served_model_name": model, "context_length": 32768,
            })
        body = json.loads(request.content)
        calls.append(body)
        if engine == "sglang":
            reason = {"type": finish}
            if matched is not MISSING:
                reason["matched"] = matched
            payload = {"output_ids": sampled, "meta_info": {
                "finish_reason": reason,
                "output_token_logprobs": [[-0.1, token, None] for token in sampled],
            }}
        else:
            choice = {"finish_reason": finish}
            if matched is not MISSING:
                choice["stop_reason"] = matched
            if transport == "direct":
                assert request.url.path == "/v1/completions"
                choice["logprobs"] = {
                    "tokens": [f"token_id:{token}" for token in sampled],
                    "token_logprobs": [-0.1] * len(sampled),
                }
            else:
                assert request.url.path == "/inference/v1/generate"
                choice["token_ids"] = sampled
                choice["logprobs"] = {"content": [
                    {"token": f"token_id:{token}", "logprob": -0.1} for token in sampled
                ]}
            payload = {"choices": [choice]}
        return httpx.Response(200, json=payload)

    loader = lambda _: tokenizer
    if transport == "direct":
        provider = DirectRolloutProvider("http://engine", inference_engine=engine, tokenizer_loader=loader)
        await provider._clients[0].aclose()
        provider._clients[0] = httpx.AsyncClient(base_url="http://engine", transport=httpx.MockTransport(upstream))
    else:
        cls = VLLMRolloutProvider if engine == "vllm" else SGLangRolloutProvider
        provider = cls("http://engine", context_length=32768, tokenizer_loader=loader)
        await provider._http.aclose()
        provider._http = httpx.AsyncClient(base_url="http://engine", transport=httpx.MockTransport(upstream))

    with TemporaryDirectory() as root:
        proxy = LLMProxyServer(
            api_key="test", save_rollout_sessions=False, error_log_dir=root,
            completion_handler=make_completion_handler(provider),
        )
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://proxy") as client:
                async def chat(sid, stop, messages=MESSAGES, stream=False):
                    return await client.post(
                        "/v1/chat/completions", headers={"Authorization": f"Bearer test|{sid}|agent"},
                        json={"model": model, "messages": messages, "max_tokens": 64, "stop": stop, "stream": stream},
                    )

                # Reported matches override guesses from the request. Null and
                # integer matches are EOS/token stops, not textual stops.
                cases = [
                    ("ordinary", "answerSTOP", "STOP", "STOP", "stop", "answer"),
                    ("inside_token", "alphabet", "pha", "pha", "stop", "al"),
                    ("multi_token", "answerSTOP HERE", ["absent", "STOP HERE"], "STOP HERE", "stop", "answer"),
                    ("empty", "STOP", "STOP", "STOP", "stop", ""),
                    ("reported", "alphabet", ["pha", "bet"], "bet", "stop", "alpha"),
                    ("legacy", "alphabet", ["bet", "pha"], MISSING, "stop", "al"),
                    ("unmatched", "answer", "STOP", MISSING, "stop", "answer"),
                    ("length", "answerSTOP", "STOP", MISSING, "length", "answerSTOP"),
                    ("eos", "answerSTOP", "STOP", None, "stop", "answerSTOP"),
                    ("token_stop", "answerSTOP", "STOP", 123, "stop", "answerSTOP"),
                ]
                if model == MODEL and engine == "sglang":
                    cases += [
                        ("special_boundary", "abc<|vision_start|>alphabet", "<|vision_start|>al",
                         "<|vision_start|>al", "stop", "abc"),
                        ("special_false_match", "ab<|vision_start|>cd abcd", "abcd", "abcd", "stop", "abcd"),
                    ]
                if model == MODEL:
                    unicode_ids = tokenizer.encode("🦄", add_special_tokens=False)
                    assert len(unicode_ids) > 1
                    interrupted = unicode_ids[:1] + tokenizer.encode("<|vision_start|>", add_special_tokens=False) + unicode_ids[1:]
                    for label, prefix in (("plain", unicode_ids), ("interrupted", interrupted)):
                        cases += [
                            (f"unicode_{label}", prefix + tokenizer.encode("STOP", add_special_tokens=False),
                             "STOP", "STOP", "stop", "🦄"),
                            (f"unicode_partial_{label}", prefix + tokenizer.encode("alphabet", add_special_tokens=False),
                             "pha", "pha", "stop", "🦄al"),
                        ]
                        if engine == "vllm":
                            cases.append((f"unicode_match_{label}", prefix, "🦄", "🦄", "stop", ""))
                if model == "gpt-oss-20b":
                    final = '<|channel|>final<|message|>'
                    analysis = '<|channel|>analysis<|message|>plan<|end|>'
                    cases = [
                        ("ordinary", final + "answerSTOP", "STOP", "STOP", "stop", "answer"),
                        ("partial_body_token", final + "alphabet", "pha", "pha", "stop", "al"),
                    ]
                    if engine == "vllm":
                        cases += [
                            ("marker_spelling", analysis + '<|start|>assistant' + final + 'weekend',
                             "end", "end", "stop", "week"),
                            ("cross_marker", analysis + '<|start|>assistant',
                             "planassistant", "planassistant", "stop", ""),
                        ]
                    else:
                        # SGLang checks stop strings in the raw token decode,
                        # so "end" matches the earlier message-ending marker.
                        cases.append(("marker_spelling", analysis, "end", "end", "stop", ""))
                for sid, text, stop, matched, finish, expected in cases:
                    sampled = text if isinstance(text, list) else tokenizer.encode(text, add_special_tokens=False)
                    text = tokenizer.decode(sampled, skip_special_tokens=False)
                    if sid == "inside_token":
                        assert len(sampled) == 1, "test must cover truncation within a real token"
                    response = await chat(sid, stop)
                    assert response.status_code == 200, response.text
                    choice = response.json()["choices"][0]
                    assert (choice["message"]["content"] or "") == expected, (sid, choice)
                    if model == "gpt-oss-20b":
                        reasoning = ("plan" if engine == "vllm" else "plan<|") if sid == "marker_spelling" else None
                        assert choice["message"].get("reasoning_content") == reasoning, (sid, choice)
                    assert choice["finish_reason"] == finish
                    assert response.json()["usage"]["completion_tokens"] == len(sampled)
                    record = (await client.get(f"/sessions/{sid}")).json()["turns"][0]
                    assert record["completion_text"] == text
                    assert record["completion_token_ids"] == sampled
                    assert record["completion_logprobs"] == [-0.1] * len(sampled)
                    assert (record["content"] or "") == expected

                    count = len(calls)
                    retry = await chat(sid, "different")
                    assert retry.json()["choices"] == response.json()["choices"] and len(calls) == count
                    sse = await chat(sid, "different", stream=True)
                    chunks = [json.loads(line[6:]) for line in sse.text.splitlines()
                              if line.startswith("data: ") and line != "data: [DONE]"]
                    assert "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks) == expected
                    assert len(calls) == count

                    if sid in ("inside_token", "partial_body_token", "unicode_interrupted", "unicode_match_interrupted"):
                        prompt_key = "input_ids" if engine == "sglang" else ("prompt" if transport == "direct" else "token_ids")
                        original = calls[-1][prompt_key] + sampled
                        messages = MESSAGES + [choice["message"], {"role": "user", "content": "More"}]
                        continued = await chat(sid, stop, messages)
                        assert continued.status_code == 200, continued.text
                        assert calls[-1][prompt_key][:len(original)] == original

                # A parser failure must not let a retry's new stop policy
                # reinterpret an already committed completion.
                prefix = '<|channel|>final<|message|>' if model == "gpt-oss-20b" else ""
                sampled = tokenizer.encode(prefix + "answerSTOP", add_special_tokens=False)
                matched, finish = MISSING, "stop"
                parser = provider.models.resolve(model).tool_parser
                with patch.object(parser, "extract_completion", side_effect=RuntimeError("injected parser failure")):
                    assert (await chat("repair", "STOP")).status_code == 500
                count = len(calls)
                repaired = await chat("repair", "answer")
                assert repaired.status_code == 200, repaired.text
                assert repaired.json()["choices"][0]["message"]["content"] == "answer"
                assert len(calls) == count
                check(f"{transport}/{engine}/{model}: stop responses, raw records, SSE, replay, recovery, strict continuation", True)
        finally:
            await proxy.stop()
            await provider.aclose()


async def parser_boundaries(bundle):
    tokenizer, parser = bundle.tokenizer, bundle.tool_parser
    text = '<think>planSTOP</think>answer'
    parsed = await parser.extract_completion(tokenizer.encode(text, add_special_tokens=False),
                                             completion_text=text, stop_strings=("STOP",))
    # ThinkReasoning omits incomplete reasoning blocks, as it does for length
    # truncation. Neither the suffix nor the answer may escape the boundary.
    assert not parsed.content and parsed.reasoning_content is None and not parsed.tool_calls
    text = '<think>plan</think>answerSTOP'
    parsed = await parser.extract_completion(tokenizer.encode(text, add_special_tokens=False),
                                             completion_text=text, stop_strings=("STOP",))
    assert parsed.content == "answer" and parsed.reasoning_content == "plan"
    text = '<tool_call><function=lookup><parameter=key>alphabet</parameter></function></tool_call>'
    parsed = await parser.extract_completion(tokenizer.encode(text, add_special_tokens=False),
                                             completion_text=text, stop_strings=("pha",))
    assert not parsed.tool_calls and "alphabet" not in (parsed.content or "")

    # Content and reasoning must share the boundary in the engine's text view,
    # even when removing a special token creates or hides a potential match.
    for text, stop, skip, content, reasoning in [
        ('<think>plan</think>abc<|vision_start|>alphabet', '<|vision_start|>al', False, 'abc', 'plan'),
        ('abc<|vision_start|>', 'vision', False, 'abc<|', None),
        ('<think>plan<|vision_start|></think>supervision', 'vision', True, 'super', 'plan<|vision_start|>'),
        ('<think>abc<|vision_start|>def</think>answer', 'cdef', True, '', None),
    ]:
        parsed = await parser.extract_completion(
            tokenizer.encode(text, add_special_tokens=False), completion_text=text,
            stop_strings=(stop,), stop_skip_special_tokens=skip,
        )
        assert parsed.content == content and parsed.reasoning_content == reasoning, (text, parsed)
        assert not parsed.tool_calls
    text = 'abc<|vision_start|>alphabet'
    literal = [token for char in text for token in tokenizer.encode(char, add_special_tokens=False)]
    parsed = await parser.extract_completion(literal, stop_strings=('<|vision_start|>al',), stop_skip_special_tokens=True)
    assert parsed.content == 'abc' and not parsed.tool_calls
    unicode_ids = tokenizer.encode('🦄', add_special_tokens=False)
    interrupted = unicode_ids[:1] + tokenizer.encode('<|vision_start|>', add_special_tokens=False) + unicode_ids[1:]
    prefix = tokenizer.encode('<think>plan</think>', add_special_tokens=False)
    parsed = await parser.extract_completion(prefix + interrupted, stop_strings=('🦄',), stop_skip_special_tokens=True)
    assert not parsed.content and parsed.reasoning_content == 'plan' and not parsed.tool_calls
    prefix = tokenizer.encode('<tool_call><function=lookup><parameter=key>', add_special_tokens=False)
    parsed = await parser.extract_completion(prefix + interrupted, stop_strings=('🦄',), stop_skip_special_tokens=True)
    assert parsed.content == tokenizer.decode(prefix, skip_special_tokens=True) and not parsed.tool_calls
    check("generic parsers share the engine stop boundary across special tokens and reasoning", True)

    harmony = ModelRegistry().resolve("gpt-oss-20b")
    encode = lambda text: harmony.tokenizer.encode(text, add_special_tokens=False)
    example = '<|start|>assistant to=functions.lookup<|message|>{"x":1}<|call|>'
    # Ordinary vocabulary tokens spelling markers must stay ordinary text.
    literal = [token for char in example for token in encode(char)]
    ids = encode('<|channel|>final<|message|>') + literal + encode('STOP<|return|>')
    parsed = await harmony.tool_parser.extract_completion(ids, stop_strings=("STOP",))
    assert parsed.content == example and not parsed.tool_calls and not parsed.reasoning_content
    for stop in ("pha", "<|call|>"):
        ids = encode(' to=functions.lookup<|channel|>commentary<|message|>{"key":"alphabet"}<|call|>')
        parsed = await harmony.tool_parser.extract_completion(ids, stop_strings=(stop,))
        assert not parsed.tool_calls
    ids = encode('<|channel|>analysis<|message|>planSTOP<|end|><|start|>assistant<|channel|>final<|message|>answer')
    parsed = await harmony.tool_parser.extract_completion(ids, stop_strings=("STOP",))
    assert not parsed.content and parsed.reasoning_content == "plan"

    # Engine-skipped markers retain their structural identity. Literal marker
    # spellings remain matchable, including when they spell a function handoff.
    ids = encode('<|channel|>final<|message|>') + literal + encode('<|return|>')
    parsed = await harmony.tool_parser.extract_completion(
        ids, stop_strings=("<|call|>",), stop_skip_special_tokens=True,
    )
    assert parsed.content == example.removesuffix('<|call|>') and not parsed.tool_calls
    ids = encode(' to=functions.lookup<|channel|>commentary<|message|>{"key":"alphabet"}<|call|>')
    for stop in ("missing", "<|call|>"):
        parsed = await harmony.tool_parser.extract_completion(ids, stop_strings=(stop,), stop_skip_special_tokens=True)
        assert len(parsed.tool_calls) == 1 and parsed.tool_calls[0].arguments == '{"key":"alphabet"}'
    parsed = await harmony.tool_parser.extract_completion(ids, stop_strings=("pha",), stop_skip_special_tokens=True)
    assert not parsed.tool_calls and parsed.content == '{"key":"al'
    unicode_ids = encode('🦄')
    interrupted = unicode_ids[:1] + encode('<|reserved_200000|>') + unicode_ids[1:]
    for channel in ('final', 'analysis'):
        ids = encode(f'<|channel|>{channel}<|message|>') + interrupted
        parsed = await harmony.tool_parser.extract_completion(ids, stop_strings=('🦄',), stop_skip_special_tokens=True)
        assert not parsed.content and not parsed.reasoning_content and not parsed.tool_calls
    check("stops precede reasoning/tool parsing; Harmony retains structural token identity", True)


async def main():
    bundle = ModelRegistry().resolve(MODEL)
    for transport in ("direct", "native"):
        for engine in ("vllm", "sglang"):
            await exercise(transport, engine, bundle.tokenizer)
    harmony = ModelRegistry().resolve("gpt-oss-20b")
    for transport in ("direct", "native"):
        for engine in ("vllm", "sglang"):
            await exercise(transport, engine, harmony.tokenizer, model="gpt-oss-20b")
    await parser_boundaries(bundle)
    print("PASS: textual stop boundaries")


if __name__ == "__main__":
    asyncio.run(main())
