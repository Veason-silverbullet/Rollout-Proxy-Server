"""Offline test of the training-side sampling-override layers.

The sampling precedence contract, highest to lowest:

    runtime overrides (PUT /sampling_overrides)
    > config sampling_overrides
    > request-provided params
    > the model's bundled generation_config.json

Drives the real provider stack (fake tokenizer, scripted engine) and the
real proxy HTTP surface, and pins:

1. **Config parsing** — ``sampling_overrides`` accepts distribution keys
   only; ``max_tokens``/``stop``, non-mappings and non-numeric values are
   refused at load time with the config path in the error.
2. **Layering** — each layer beats the ones below it in the params the
   engine actually receives; ``max_tokens`` stays request-owned even with
   overrides active; an empty runtime push falls back to the config layer.
3. **Endpoints** — ``GET /sampling_overrides`` is open and reports
   ``{config, runtime, effective}``; ``PUT`` requires the **bare** real API
   key (the keyed per-session form agents hold is refused), rejects
   disallowed keys with 400, and both endpoints answer 501 in relay mode,
   which holds no sampling state on this process.
4. **End to end** — a PUT over HTTP changes the sampling params of the next
   completion served through the proxy.

No inference engine and no ``transformers`` needed.

Run:
    python test/test-sampling-overrides.py
"""

from __future__ import annotations
import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Any
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver.config import load_config  # noqa: E402
from proxyserver.rollout_provider import BaseRolloutProvider  # noqa: E402
from proxyserver.server import LLMProxyServer, make_local_completion_handler  # noqa: E402
from offline_common import FakeTokenizer, check  # noqa: E402

# Any model in tokenization/mapping.json; the tokenizer itself is faked but
# the bundled generation_config.json of its profile is real (Qwen3.5:
# temperature=1.0, top_p=0.95, top_k=20 — the bottom layer of the tests).
MODEL = "Qwen3.5-9B"
PROXY_API_KEY = "sampling-overrides-test-key"
DELIMITER = "-"


class SamplingProvider(BaseRolloutProvider):
    """Real generate flow over a scripted engine, recording sampling params."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(tokenizer_loader=lambda path: FakeTokenizer(), **kwargs)
        self.seen: list[dict[str, Any]] = []

    async def _call_engine(self, prompt_ids, sampling_params, session_id):
        self.seen.append(dict(sampling_params))
        tokens = [900, 901, 902]
        return tokens, [-0.1] * len(tokens), "stop", {}


async def one_turn(provider: SamplingProvider, session_id: str, **request_params: Any) -> dict[str, Any]:
    """One first-turn generate; returns the sampling params the engine saw."""
    await provider.generate(
        messages=[{"role": "user", "content": "hi"}],
        session_id=session_id,
        model=MODEL,
        max_tokens=8,
        **request_params,
    )
    return provider.seen[-1]


# ---------------------------------------------------------------------------
# 1. Config parsing
# ---------------------------------------------------------------------------

BASE_YAML = "inference_engine: sglang\ntransport_mode: slime\n"


def load_yaml_config(body: str):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(body)
        path = f.name
    try:
        return load_config(path)
    finally:
        Path(path).unlink()


def test_config_parsing() -> None:
    cfg = load_yaml_config(BASE_YAML + "sampling_overrides:\n  temperature: 1.0\n  top_p: 1.0\n")
    check("config: valid overrides parse", cfg.sampling_overrides == {"temperature": 1.0, "top_p": 1.0})

    cfg = load_yaml_config(BASE_YAML)
    check("config: absent overrides stay None", cfg.sampling_overrides is None)

    # The removed sampling_defaults field is refused, not ignored: unknown
    # keys are normally skipped, and a stale config would otherwise change
    # sampling behavior silently.
    try:
        load_yaml_config(BASE_YAML + "sampling_defaults:\n  presence_penalty: 1.2\n")
    except ValueError as e:
        check("config: removed sampling_defaults key refused loudly", "has been removed" in str(e))
    else:
        raise AssertionError("FAIL: config accepted the removed sampling_defaults key")

    for label, body in [
        ("non-mapping", "sampling_overrides: 1.0\n"),
        ("max_tokens", "sampling_overrides:\n  max_tokens: 64\n"),
        ("stop", "sampling_overrides:\n  stop: ['x']\n"),
        ("non-numeric value", "sampling_overrides:\n  temperature: warm\n"),
    ]:
        try:
            load_yaml_config(BASE_YAML + body)
        except ValueError as e:
            check(f"config: {label} refused, error names the file", "sampling_overrides" in str(e))
        else:
            raise AssertionError(f"FAIL: config accepted {label}")


# ---------------------------------------------------------------------------
# 2. Layering
# ---------------------------------------------------------------------------


async def test_layering() -> None:
    # Bottom layer: the bundled generation_config.json fills request-omitted
    # params.  Its top_p (0.95 for Qwen3.5) is the value everything above
    # must beat.
    provider = SamplingProvider()
    params = await one_turn(provider, "gen-config")
    baseline_top_p = params["top_p"]
    check("gen_config fills omitted params", baseline_top_p is not None and baseline_top_p != 0.5)

    # Request beats gen_config.
    params = await one_turn(provider, "request", top_p=0.5)
    check("request > generation_config", params["top_p"] == 0.5)

    # Config sampling_overrides beat the request.
    provider = SamplingProvider(sampling_overrides={"top_p": 1.0})
    params = await one_turn(provider, "config-override", top_p=0.5)
    check("config sampling_overrides > request", params["top_p"] == 1.0)
    check("override leaves unrelated keys to lower layers", params["temperature"] is not None)

    # Runtime push beats the config layer; an empty push falls back to it.
    state = provider.set_runtime_sampling_overrides({"top_p": 0.7})
    check("set returns the layers", state["runtime"] == {"top_p": 0.7} and state["config"] == {"top_p": 1.0})
    check("effective = runtime over config", state["effective"] == {"top_p": 0.7})
    params = await one_turn(provider, "runtime-override", top_p=0.5)
    check("runtime > config sampling_overrides", params["top_p"] == 0.7)
    provider.set_runtime_sampling_overrides({})
    params = await one_turn(provider, "runtime-cleared", top_p=0.5)
    check("cleared runtime falls back to config layer", params["top_p"] == 1.0)

    # max_tokens stays request-owned whatever the override layers hold.
    check("max_tokens stays request-owned", params["max_tokens"] == 8)

    # The setter enforces the same contract as the config.
    for label, bad in [("max_tokens", {"max_tokens": 64}), ("non-numeric", {"temperature": "warm"}), ("non-mapping", [1.0])]:
        try:
            provider.set_runtime_sampling_overrides(bad)  # type: ignore[arg-type]
        except ValueError:
            check(f"runtime setter refuses {label}", True)
        else:
            raise AssertionError(f"FAIL: runtime setter accepted {label}")

    # A disallowed config layer fails at construction, not at request time.
    try:
        SamplingProvider(sampling_overrides={"max_tokens": 64})
    except ValueError:
        check("constructor refuses disallowed config overrides", True)
    else:
        raise AssertionError("FAIL: constructor accepted max_tokens override")


# ---------------------------------------------------------------------------
# 3 + 4. Endpoints and end-to-end
# ---------------------------------------------------------------------------


class LocalStack:
    """Local-mode proxy wired to a SamplingProvider, as cli.py wires it."""

    def __init__(self, **provider_kwargs: Any) -> None:
        self.provider = SamplingProvider(**provider_kwargs)
        self.proxy = LLMProxyServer(
            host="127.0.0.1", port=0, api_key=PROXY_API_KEY,
            key_delimiter=DELIMITER,
            completion_handler=make_local_completion_handler(self.provider),
            on_session_deleted=self.provider.release_session,
            get_sampling_overrides=self.provider.get_sampling_overrides,
            set_sampling_overrides=self.provider.set_runtime_sampling_overrides,
            save_rollout_sessions=False,
        )
        self.url = ""

    async def __aenter__(self) -> "LocalStack":
        self.url = await self.proxy.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.proxy.stop()


async def test_endpoints() -> None:
    async with LocalStack(sampling_overrides={"top_p": 1.0}) as stack, httpx.AsyncClient() as http:
        url = f"{stack.url}/sampling_overrides"
        bare = {"Authorization": f"Bearer {PROXY_API_KEY}"}

        resp = await http.get(url)
        check("GET is open", resp.status_code == 200)
        check("GET reports the layers", resp.json() == {"config": {"top_p": 1.0}, "runtime": {}, "effective": {"top_p": 1.0}})

        resp = await http.put(url, json={"temperature": 0.8})
        check("PUT without auth -> 401", resp.status_code == 401)
        keyed = {"Authorization": f"Bearer {PROXY_API_KEY}{DELIMITER}sess{DELIMITER}agent"}
        resp = await http.put(url, headers=keyed, json={"temperature": 0.8})
        check("PUT with keyed per-session api_key -> 401 (bare key required)", resp.status_code == 401)

        resp = await http.put(url, headers=bare, json={"temperature": 0.8})
        check("PUT with bare key -> 200", resp.status_code == 200)
        check("PUT echoes the effective policy", resp.json()["effective"] == {"temperature": 0.8, "top_p": 1.0})

        resp = await http.put(url, headers=bare, json={"max_tokens": 64})
        check("PUT with request-owned key -> 400", resp.status_code == 400)
        resp = await http.put(url, headers=bare, json=[1.0])
        check("PUT with non-object body -> 400", resp.status_code == 400)
        resp = await http.get(url)
        check("rejected PUTs leave the layers untouched", resp.json()["runtime"] == {"temperature": 0.8})

        # End to end: the pushed policy shapes the next completion served
        # through the proxy.
        agent = {"Authorization": f"Bearer {PROXY_API_KEY}{DELIMITER}e2e{DELIMITER}agentA"}
        resp = await http.post(
            f"{stack.url}/v1/chat/completions", headers=agent,
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 8, "temperature": 0.2, "top_p": 0.5},
        )
        check("completion through the proxy succeeds", resp.status_code == 200)
        seen = stack.provider.seen[-1]
        check("pushed override beats the agent's temperature", seen["temperature"] == 0.8)
        check("config override beats the agent's top_p", seen["top_p"] == 1.0)
        check("agent keeps max_tokens", seen["max_tokens"] == 8)

        # Clearing over HTTP falls back to the config layer.
        resp = await http.put(url, headers=bare, json={})
        check("PUT {} clears the runtime layer", resp.json()["effective"] == {"top_p": 1.0})


async def test_relay_mode_501() -> None:
    proxy = LLMProxyServer(host="127.0.0.1", port=0, api_key=PROXY_API_KEY,
                           key_delimiter=DELIMITER, save_rollout_sessions=False)
    url = await proxy.start()
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{url}/sampling_overrides")
            check("relay mode GET -> 501", resp.status_code == 501)
            resp = await http.put(f"{url}/sampling_overrides",
                                  headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
                                  json={"temperature": 1.0})
            check("relay mode PUT -> 501", resp.status_code == 501)
    finally:
        await proxy.stop()


async def main() -> None:
    print("config parsing:")
    test_config_parsing()
    print("layering:")
    await test_layering()
    print("endpoints + end to end:")
    await test_endpoints()
    print("relay mode:")
    await test_relay_mode_501()
    print("ALL SAMPLING-OVERRIDE TESTS PASS")


if __name__ == "__main__":
    asyncio.run(main())
