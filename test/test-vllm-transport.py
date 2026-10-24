"""Offline vLLM generation token protocol, configuration, recording, and cleanup tests.

HTTP payloads use the vLLM generation contract pinned in README.md. Run: python test/test-vllm-transport.py
"""

import asyncio
import base64
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxyserver.config import load_config
from proxyserver.engines import EngineError
from proxyserver.rollout_provider import VLLMRolloutProvider, _pack_vllm_routed_experts, _vllm_generate
from proxyserver.server import LLMProxyServer, make_completion_handler
from offline_common import FakeTokenizer, check

MODEL = "Qwen3.5-9B"
TOKENS = [900, 901]
PROMPT = [1, 2, 3]
MESSAGES = [{"role": "user", "content": "hello"}]


def npy_blob(values):
    buffer = io.BytesIO()
    np.save(buffer, values)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def vllm_response():
    return {
        "choices": [{"token_ids": list(TOKENS), "finish_reason": "stop",
                     "logprobs": {"content": [{"token_id": token, "logprob": -0.1} for token in TOKENS]},
                     "sampling_mask": [[900, 7], [901]]}],
        "weight_version": 0,
        "usage": {"completion_tokens": len(TOKENS)},
    }


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeHttp:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def post(self, url, json, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return FakeResponse(self.payload)


async def test_wire_contract():
    http = FakeHttp(vllm_response())
    params = {"max_tokens": 8, "logprobs": True, "top_p": 0.9, "top_k": -1,
              "stop": ["END"], "stop_token_ids": [99], "seed": 42, "min_p": None}
    tokens, probs, reason, meta = await _vllm_generate(http, PROMPT, params, "trial-agent")
    call = http.calls[0]
    body = call["json"]
    check("native endpoint and token-ID prompt", call["url"] == "/inference/v1/generate" and body["token_ids"] == PROMPT)
    check("vLLM generation affinity header", call["headers"] == {"x-session-id": "trial-agent"})
    check("upstream sampling dialect", body["sampling_params"] == {
        "max_tokens": 8, "logprobs": 1, "top_p": 0.9, "top_k": -1, "stop": ["END"],
        "stop_token_ids": [99], "seed": 42, "include_stop_str_in_output": True,
    })
    check("caller parameters unchanged", params["logprobs"] is True and "return_sampling_mask" not in params)
    check("sampled tokens and logprobs aligned", tokens == TOKENS and probs == [-0.1, -0.1] and reason == "stop")
    check("weight zero and top-p support retained", meta == {
        "weight_version": "0", "rollout_top_p_token_ids": [900, 7, 901], "rollout_top_p_token_offsets": [0, 2, 3],
    })
    result = await _vllm_generate(http, PROMPT, {"top_p": 1.0}, None)
    check("full-distribution sampling has no replay metadata", result[3] == {"weight_version": "0"})
    check("absent stream has no affinity header", http.calls[-1]["headers"] is None)
    result = await _vllm_generate(http, PROMPT, {"top_p": 1.0, "top_k": -1, "min_p": 0.0}, None)
    check("disabled top-k and min-p need no replay metadata", result[3] == {"weight_version": "0"})
    result = await _vllm_generate(http, PROMPT, {"temperature": 0.0, "top_p": 0.9, "top_k": 20}, None)
    check("greedy sampling needs no replay metadata", result[3] == {"weight_version": "0"})
    for truncation in ({"top_p": 1.0, "top_k": 20}, {"min_p": 0.05}):
        result = await _vllm_generate(http, PROMPT, truncation, None)
        check(f"{truncation} retains sampling support", result[3]["rollout_top_p_token_offsets"] == [0, 2, 3])
        payload = vllm_response()
        del payload["choices"][0]["sampling_mask"]
        try:
            await _vllm_generate(FakeHttp(payload), PROMPT, truncation, "s")
        except EngineError as e:
            check(f"{truncation} without a mask fails before recording", "top_k > 0" in str(e))
        else:
            raise AssertionError(f"truncated sampling without a mask accepted: {truncation}")
    for finish in ("length", {"type": "length"}, "abort", "cancelled", {"type": "abort"}):
        payload = vllm_response()
        payload["choices"][0]["finish_reason"] = finish
        expected = "length" if "length" in str(finish) else "abort"
        result = await _vllm_generate(FakeHttp(payload), PROMPT, {}, "s")
        check(f"finish reason {finish}", result[2] == expected)


async def test_small_positive_temperatures():
    # vLLM raises small positive temperatures to 0.01 before its greedy check.
    # They still need support capture when any sampling cutoff applies.
    for temperature in (1e-12, 1e-6, 1e-5):
        for cutoff in ({"top_p": 0.9}, {"top_k": 20}, {"min_p": 0.05}):
            params = {"temperature": temperature, **cutoff}
            http = FakeHttp(vllm_response())
            result = await _vllm_generate(http, PROMPT, params, "small-positive")
            check(f"{params} retains sampling support",
                  result[3].get("rollout_top_p_token_ids") == [900, 7, 901]
                  and result[3].get("rollout_top_p_token_offsets") == [0, 2, 3])
            check("positive temperature reaches the engine unchanged",
                  http.calls[-1]["json"]["sampling_params"]["temperature"] == temperature)
            payload = vllm_response()
            del payload["choices"][0]["sampling_mask"]
            try:
                await _vllm_generate(FakeHttp(payload), PROMPT, params, "small-positive")
            except EngineError as e:
                check(f"{params} requires a sampling mask", "sampling mask" in str(e))
            else:
                raise AssertionError(f"small positive temperature accepted without a mask: {params}")


async def test_invalid_payloads():
    payloads = [None, [], {}, {"choices": []}, {"choices": [None]}, {"choices": [{}, {}]}]
    for field, values in {
        "token_ids": [None, "900", [True, 901], [-1, 901], [900.5, 901]],
        "logprobs": [None, {}, {"content": []}, {"content": [None, None]}],
        "sampling_mask": [None, [], [[900]], [[900], []], [[True, 900], [901]], [[7], [901]], [[900, 900], [901]]],
        "finish_reason": ["error", {"type": "error"}],
    }.items():
        for value in values:
            payload = vllm_response()
            payload["choices"][0][field] = value
            payloads.append(payload)
    for value in (float("nan"), float("inf"), -float("inf"), True, "-0.1", 10 ** 1000):
        payload = vllm_response()
        payload["choices"][0]["logprobs"]["content"][0]["logprob"] = value
        payloads.append(payload)
    for value in (901, True, None):
        payload = vllm_response()
        payload["choices"][0]["logprobs"]["content"][0]["token_id"] = value
        payloads.append(payload)
    for value in ("token_id:901", "900", "<decoded text>", "token_id:0900", None, 900):
        payload = vllm_response()
        entry = payload["choices"][0]["logprobs"]["content"][0]
        del entry["token_id"]
        entry["token"] = value
        payloads.append(payload)
    for value in (3, True, "2"):
        payload = vllm_response()
        payload["usage"]["completion_tokens"] = value
        payloads.append(payload)
    for payload in payloads:
        try:
            await _vllm_generate(FakeHttp(payload), PROMPT, {"top_p": 0.9}, "s")
        except EngineError:
            pass
        else:
            raise AssertionError(f"invalid response accepted: {payload}")
    check(f"{len(payloads)} malformed responses fail before recording", True)


async def test_routes():
    values = np.arange(16, dtype=np.int32).reshape(4, 2, 2)
    payload = vllm_response()
    payload["choices"][0]["routed_experts"] = npy_blob(values)
    http = FakeHttp(payload)
    result = await _vllm_generate(http, PROMPT, {}, "s", return_routed_experts=True)
    blob = result[3]["routed_experts"]
    check("routing offset zero requested", http.calls[0]["json"]["sampling_params"]["routed_experts_prompt_start"] == 0)
    check("numpy routing packed like SGLang", blob == {
        "data": base64.b64encode(values.astype(np.uint8).tobytes()).decode(), "rows": 4, "cols": 4, "dtype": "uint8", "start": 0,
    })
    for source in (values, values[2:]):
        payload["choices"][0]["routed_experts"] = npy_blob(source)
        result = await _vllm_generate(http, PROMPT, {}, "s", return_routed_experts=True, routed_experts_prior=(2, 4))
        delta = result[3]["routed_experts"]
        check("full and delta routing agree", delta["start"] == 2 and delta["rows"] == 2
              and base64.b64decode(delta["data"]) == values[2:].astype(np.uint8).tobytes())
        check("prior reaches engine", http.calls[-1]["json"]["sampling_params"]["routed_experts_prompt_start"] == 2)
    invalid = [None, "not base64", npy_blob(values.astype(float)), npy_blob(values[:1]),
               npy_blob(np.array([object()], dtype=object)), npy_blob(values + 256), npy_blob(values - 1),
               npy_blob(np.array(3)), npy_blob(np.empty((4, 0), dtype=int))]
    for encoded in invalid:
        try:
            _pack_vllm_routed_experts({"routed_experts": encoded}, 3, 2, "s")
        except EngineError:
            pass
        else:
            raise AssertionError("invalid routed experts accepted")
    check("invalid numpy captures refused", True)


def test_configuration():
    cfg = load_config(inference_engine="vllm", transport_mode="vllm_generate")
    check("bundled vLLM generation config", cfg.transport_mode == "vllm_generate" and cfg.inference_engine == "vllm")
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "vllm_generate.yaml"
        path.write_text("inference_engine: vllm\ntransport_mode: vllm_generate\ncontext_length: 42\nreturn_routed_experts: true\n")
        cfg = load_config(path)
        check("vLLM generation accepts context and routing settings", cfg.context_length == 42 and cfg.return_routed_experts)
        for body in ("inference_engine: sglang\ntransport_mode: vllm_generate\n",
                     "inference_engine: vllm\ntransport_mode: vllm_generate\ncontext_length: 0\n"):
            path.write_text(body)
            try:
                load_config(path)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid vLLM generation configuration accepted")
    for kwargs in ({"inference_engine": "sglang"}, {"context_length": 0}):
        try:
            VLLMRolloutProvider("http://router", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid vLLM generation provider accepted")


async def test_provider_and_recording():
    calls = []
    gate = asyncio.Event()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    mode = "normal"

    async def router(request):
        body = json.loads(request.content)
        calls.append((request, body))
        assert request.url.path == "/inference/v1/generate", "no discovery or global abort requests"
        if mode == "wait":
            started.set()
            try:
                await gate.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        if mode == "error":
            return httpx.Response(503, text="engine unavailable")
        payload = vllm_response()
        if mode == "abort":
            payload["choices"][0]["finish_reason"] = "cancelled"
        if mode == "bad":
            payload["choices"][0]["token_ids"] = [900]
        if "routed_experts_prompt_start" in body["sampling_params"]:
            start = body["sampling_params"]["routed_experts_prompt_start"]
            rows = len(body["token_ids"]) + len(TOKENS) - 1 - start
            payload["choices"][0]["routed_experts"] = npy_blob(np.ones((rows, 2, 2), dtype=np.int32))
        return httpx.Response(200, json=payload)

    with patch.dict("os.environ", {"INFERENCE_ENGINE": "sglang"}):
        provider = VLLMRolloutProvider("http://router/", api_key="router-secret", context_length=4096,
                                      return_routed_experts=True, sampling_overrides={"top_p": 0.9},
                                      tokenizer_loader=lambda path: FakeTokenizer())
    check("vLLM generation defaults to vLLM despite environment", provider.engine == "vllm")
    headers = provider._http.headers
    await provider.aclose()
    provider._http = httpx.AsyncClient(base_url=provider.router_url, headers=headers, transport=httpx.MockTransport(router))
    with tempfile.TemporaryDirectory() as root:
        proxy = LLMProxyServer(api_key="test", completion_handler=make_completion_handler(provider),
                               on_session_deleted=provider.release_session, session_dir=root, error_log_dir=root,
                               get_routed_experts_config=provider.get_routed_experts_config,
                               set_routed_experts_config=provider.set_runtime_routed_experts)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://proxy") as client:
            async def generate(sid="trial", agent="agent", messages=None):
                return await client.post("/v1/chat/completions", headers={"Authorization": f"Bearer test|{sid}|{agent}"},
                                         json={"model": MODEL, "messages": messages or MESSAGES, "seed": 42, "stop_token_ids": [99]})

            first = await generate()
            check("vLLM generation reaches HTTP recording", first.status_code == 200)
            check("router auth and stream affinity", calls[0][0].headers["authorization"] == "Bearer router-secret"
                  and calls[0][0].headers["x-session-id"] == "trial-agent")
            check("budget clamped and generation controls forwarded", calls[0][1]["sampling_params"]["max_tokens"] + len(calls[0][1]["token_ids"]) == 4096
                  and calls[0][1]["sampling_params"]["seed"] == 42 and calls[0][1]["sampling_params"]["stop_token_ids"] == [99])
            again = await generate()
            check("exact retry uses cached tokens and metadata", again.json()["choices"] == first.json()["choices"] and len(calls) == 1)
            continuation = MESSAGES + [first.json()["choices"][0]["message"], {"role": "user", "content": "next"}]
            check("second turn succeeds", (await generate(messages=continuation)).status_code == 200)
            record = (await client.get("/sessions/trial")).json()
            turns = record["turns"]
            check("record reconstructs exact second prompt", calls[1][1]["token_ids"] ==
                  turns[0]["prompt_token_ids"] + TOKENS + turns[1]["prompt_token_ids"])
            check("top-p data and weight persisted", all(t["rollout_top_p_token_offsets"] == [0, 2, 3]
                  and t["weight_version"] == "0" for t in turns))
            check("routing deltas accumulated", record["routed_experts"]["agent"]["rows"] == len(calls[1][1]["token_ids"]) + 1)
            check("independent agent has own stream", (await generate(agent="second")).status_code == 200
                  and calls[-1][0].headers["x-session-id"] == "trial-second")
            saved = json.loads(next(Path(root).rglob("trial.json")).read_text())
            check("top-p data stays out of disk snapshots by default",
                  all("rollout_top_p_token_ids" not in turn and "rollout_top_p_token_offsets" not in turn
                      for turn in saved["turns"]))
            control = await client.put("/routed_experts", json={"enabled": False}, headers={"Authorization": "Bearer test"})
            check("runtime routing toggle", control.status_code == 200 and not provider.get_routed_experts_config()["effective"])
            fresh = [{"role": "user", "content": "new conversation"}]
            check("fresh conversation without capture", (await generate(messages=fresh)).status_code == 200)
            record = (await client.get("/sessions/trial")).json()
            check("new conversation removes stale routing", "agent" not in record["routed_experts"] and "second" in record["routed_experts"])
            provider.set_runtime_routed_experts(None)
            check("clearing override restores config", provider.get_routed_experts_config()["effective"])
            for value in (1, "true", []):
                try:
                    provider.set_runtime_routed_experts(value)
                except ValueError:
                    pass
                else:
                    raise AssertionError("invalid runtime capture setting")
            for failure, expected in (("bad", 500), ("abort", 500), ("error", 502)):
                mode = failure
                response = await generate(sid=failure)
                check(f"{failure} fails without recording", response.status_code == expected
                      and not (await client.get(f"/sessions/{failure}")).json()["turns"])
            mode = "normal"
            check("retry after invalid engine result succeeds", (await generate(sid="bad")).status_code == 200)
            mode = "wait"
            pending = asyncio.create_task(generate(sid="waiting"))
            await asyncio.wait_for(started.wait(), 3)
            response = await client.delete("/sessions/waiting")
            await asyncio.wait_for(cancelled.wait(), 3)
            await asyncio.gather(pending, return_exceptions=True)
            check("deletion cancels request and releases stream", response.status_code == 200
                  and not provider.models.has_session("waiting-agent"))
            mode = "normal"
            check("other agents survive cancellation", (await generate(sid="survivor")).status_code == 200)
            await client.delete("/sessions/trial")
            check("deletion releases every agent", not provider.models.has_session("trial-agent")
                  and not provider.models.has_session("trial-second"))
        await provider.aclose()
        check("HTTP client closed", provider._http.is_closed)
    provider = VLLMRolloutProvider("http://router", context_length=3, tokenizer_loader=lambda path: FakeTokenizer())
    try:
        try:
            await provider._call_engine([1, 2, 3], {"max_tokens": 2}, "s")
        except EngineError:
            check("full prompt rejected before HTTP", True)
        else:
            raise AssertionError("full context accepted")
    finally:
        await provider.aclose()


async def test_missing_masks_fail_before_generation():
    calls = []

    async def router(request):
        calls.append(json.loads(request.content))
        payload = vllm_response()
        del payload["choices"][0]["sampling_mask"]
        return httpx.Response(200, json=payload)

    # top_p pinned to 1.0 still truncates: the bundled Qwen top_k=20 applies.
    provider = VLLMRolloutProvider("http://router", context_length=4096, sampling_overrides={"top_p": 1.0},
                                   tokenizer_loader=lambda path: FakeTokenizer())
    await provider.aclose()
    provider._http = httpx.AsyncClient(base_url=provider.router_url, transport=httpx.MockTransport(router))
    try:
        errors = []
        for sid, temperature in (("first", None), ("second", None), ("small-positive", 1e-6)):
            try:
                await provider.generate(MESSAGES, f"{sid}-agent", model=MODEL, temperature=temperature)
            except EngineError as e:
                errors.append(str(e))
            else:
                raise AssertionError("truncated sampling without a mask accepted")
        check("first missing mask names the engine setting", "enable return_sampling_mask" in errors[0])
        check("later truncated requests fail before generating",
              len(calls) == 1 and all("refusing before generating" in error for error in errors[1:]))
        greedy = await provider.generate(MESSAGES, "greedy-agent", model=MODEL, temperature=0.0)
        check("untruncated requests still generate", greedy[1] == TOKENS and len(calls) == 2)
    finally:
        await provider.aclose()


async def main():
    test_configuration()
    await test_wire_contract()
    await test_small_positive_temperatures()
    await test_invalid_payloads()
    await test_routes()
    await test_provider_and_recording()
    await test_missing_masks_fail_before_generation()
    print("PASS: vLLM generation transport, capture, recording, and cleanup")


if __name__ == "__main__":
    asyncio.run(main())
