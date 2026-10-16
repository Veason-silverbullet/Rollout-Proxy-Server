"""VeRL-owned native HTTP contracts through the shared SGLang/vLLM providers.

Fixtures match VeRL's SGLang server and its pinned upstream vLLM 0.29.0
token endpoint.
Run: python test/test-verl.py
"""

from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxyserver.rollout_provider import SGLangRolloutProvider, VLLMRolloutProvider
from proxyserver.server import LLMProxyServer, make_completion_handler
from offline_common import FakeTokenizer, check

MODEL = "Qwen3.5-9B"
MESSAGES = [{"role": "user", "content": "hello"}]
TOKENS = [900, 901]


async def exercise(engine):
    calls = []
    mode = "normal"
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def upstream(request):
        body = json.loads(request.content)
        calls.append((request, body))
        if mode == "wait":
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        if mode == "error":
            return httpx.Response(503, text="rollout server unavailable")
        params = body["sampling_params"]
        if engine == "sglang":
            assert request.url.path == "/generate"
            assert body["return_logprob"] is True
            assert "max_new_tokens" in params and "max_tokens" not in params
            assert params["sampling_seed"] == 0
            payload = {"output_ids": TOKENS, "meta_info": {
                "finish_reason": {"type": "abort" if mode == "abort" else "stop"},
                "output_token_logprobs": [[-0.1, 900, None], [-0.2, 901, None]],
            }}
        else:
            assert request.url.path == "/inference/v1/generate"
            assert params["logprobs"] == 1 and params["seed"] == 0
            assert "return_sampling_mask" not in params, "not an upstream SamplingParams field"
            payload = {"choices": [{
                "index": 0, "token_ids": TOKENS,
                "finish_reason": "abort" if mode == "abort" else "stop",
                "logprobs": {"content": [
                    {"token": "token_id:900", "logprob": -0.1, "top_logprobs": []},
                    {"token": "token_id:901", "logprob": -0.2, "top_logprobs": []},
                ]},
                # Enabled at engine startup, not by the generation request.
                "sampling_mask": [[900, 7], [901]],
            }], "usage": {"completion_tokens": 2}}
            if mode == "bad":
                payload["choices"][0]["logprobs"]["content"][0]["token"] = "token_id:999"
            if mode == "no_mask":
                payload["choices"][0].pop("sampling_mask")
        # Native endpoints do not invent the actor wrapper's global_steps.
        return httpx.Response(200, json=payload)

    cls = SGLangRolloutProvider if engine == "sglang" else VLLMRolloutProvider
    with patch.dict(sys.modules, ray=None):
        provider = cls("http://verl-server", context_length=4096,
                       sampling_overrides={"top_p": 0.9}, tokenizer_loader=lambda path: FakeTokenizer())
        await provider._http.aclose()
        provider._http = httpx.AsyncClient(base_url="http://verl-server", transport=httpx.MockTransport(upstream))
        proxy = LLMProxyServer(api_key="test", save_rollout_sessions=False,
                               completion_handler=make_completion_handler(provider),
                               on_session_deleted=provider.release_session)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://proxy") as client:
                async def chat(sid="trial", messages=MESSAGES):
                    return await client.post("/v1/chat/completions",
                                             headers={"Authorization": f"Bearer test-{sid}-agent"},
                                             json={"model": MODEL, "messages": messages, "max_tokens": 8, "seed": 0})

                first = await chat()
                assert first.status_code == 200, first.text
                retry = await chat()
                assert retry.json()["choices"] == first.json()["choices"] and len(calls) == 1
                continued = MESSAGES + [first.json()["choices"][0]["message"], {"role": "user", "content": "continue"}]
                assert (await chat(messages=continued)).status_code == 200
                record = (await client.get("/sessions/trial")).json()
                assert len(record["turns"]) == 2
                assert record["turns"][0]["completion_token_ids"] == TOKENS
                assert record["turns"][0]["completion_logprobs"] == [-0.1, -0.2]
                assert all(turn.get("weight_version") is None for turn in record["turns"])
                prompt_key = "input_ids" if engine == "sglang" else "token_ids"
                first_prompt = calls[0][1][prompt_key]
                assert calls[1][1][prompt_key][:len(first_prompt) + 2] == first_prompt + TOKENS
                header = "x-smg-routing-key" if engine == "sglang" else "x-session-id"
                assert all(call[0].headers[header] == "trial-agent" for call in calls)
                if engine == "vllm":
                    assert record["turns"][0]["rollout_top_p_token_offsets"] == [0, 2, 3]
                check(f"{engine}: native HTTP, strict continuation, logprobs, retries, no fabricated weight version", True)

                failures = [("error", 502), ("abort", 500)]
                if engine == "vllm":
                    failures += [("bad", 500), ("no_mask", 500)]
                for mode, expected in failures:
                    response = await chat(sid=mode)
                    assert response.status_code == expected, response.text
                    assert not (await client.get(f"/sessions/{mode}")).json()["turns"]
                mode = "wait"
                pending = asyncio.create_task(chat(sid="waiting"))
                await asyncio.wait_for(started.wait(), 3)
                assert (await client.delete("/sessions/waiting")).status_code == 200
                await asyncio.wait_for(cancelled.wait(), 3)
                await asyncio.gather(pending, return_exceptions=True)
                assert not provider.models.has_session("waiting-agent")
                mode = "normal"
                assert (await chat(sid="survivor")).status_code == 200
                await client.delete("/sessions/trial")
                assert not provider.models.has_session("trial-agent")
                check(f"{engine}: failed/aborted responses stay uncommitted; deletion cancels HTTP and releases streams", True)
        finally:
            await proxy.stop()
            await provider.aclose()


async def main():
    for engine in ("sglang", "vllm"):
        await exercise(engine)


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    asyncio.run(asyncio.wait_for(main(), 30))
    print("PASS: VeRL native HTTP contracts")
