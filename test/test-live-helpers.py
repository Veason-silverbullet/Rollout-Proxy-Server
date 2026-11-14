"""Offline regressions for live-test endpoint selection and EOS assertions."""

from __future__ import annotations
import asyncio
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import httpx

# common initializes the standalone integration fixture at import time.
# Supply an isolated dummy endpoint without reading local cluster settings.
is_file = Path.is_file
with patch.dict(os.environ, {
    "INFERENCE_ENGINE": "vllm", "TRANSPORT_MODE": "direct",
    "OPENAI_BASE_URL": "http://unused.invalid/v1",
    "OPENAI_API_KEY": "test", "MODEL_NAME": "Qwen3.5-4B",
}), patch.object(Path, "is_file", lambda p: False if p.name == "test_engines.yaml" else is_file(p)):
    import common

spec = importlib.util.spec_from_file_location("contract_test", Path(__file__).with_name("test-contract.py"))
contract = importlib.util.module_from_spec(spec)
with patch.dict(os.environ):
    spec.loader.exec_module(contract)


def test_agent_selection() -> None:
    with patch.dict(os.environ, {"LIVE_TEST_MODE": "full"}):
        full = common.live_agent_prompts()
        assert len(full) == 50 and len({row[0] for row in full}) == 50
    with patch.dict(os.environ, {"LIVE_TEST_MODE": "smoke"}):
        smoke = common.live_agent_prompts()
        assert len(smoke) == 8 and all(row in full for row in smoke)
        assert {row[0] for row in smoke} == {f"agent_{i}" for i in (5, 6, 7, 8, 26, 30, 42, 49)}
    with patch.dict(os.environ, {"LIVE_TEST_MODE": "typo"}):
        try:
            common.live_agent_prompts()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid live mode silently changed the workload")
    print("PASS: full workload retained; smoke mode covers representative token boundaries")


async def test_native_finish_reason() -> None:
    endpoint = common.EngineEndpoint("vllm", "http://unused.invalid/v1", "test", "fake-model")
    for reason in (None, "", "stop", "length"):
        choice = {"logprobs": {"tokens": ["token_id:900"], "token_logprobs": [-0.1]}}
        if reason is not None:
            choice["finish_reason"] = reason
        client = common.VLLMEngineClient(endpoint)
        await client.aclose()
        client._http = httpx.AsyncClient(
            base_url=endpoint.base_url,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": [choice]})),
        )
        try:
            output = await client.generate([1, 2], {"max_tokens": 8, "logprobs": True})
            assert output.stop_reason == reason, "native client invented a finish reason"
        finally:
            await client.aclose()
    print("PASS: the native client preserves absent and explicit finish reasons")


async def test_context_clamped_agent() -> None:
    from offline_common import FakeTokenizer, ScriptedProvider
    from proxyserver.service import LLMProxyServer, make_completion_handler

    class ClampedProvider(ScriptedProvider):
        async def _call_engine(self, *args, **kwargs):
            ids, probs, _, meta = await super()._call_engine(*args, **kwargs)
            return ids, probs, "length", meta

    provider = ClampedProvider()
    proxy = LLMProxyServer(
        api_key=common.PROXY_API_KEY, key_delimiter=common.KEY_DELIMITER,
        completion_handler=make_completion_handler(provider), save_rollout_sessions=False,
    )
    sdk_client = common.AsyncOpenAI

    def client(**kwargs):
        return sdk_client(http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app)), **kwargs)

    with patch.object(common, "AsyncOpenAI", side_effect=client), \
            patch.object(common, "load_bundled_tokenizer", return_value=FakeTokenizer()):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as http:
            record = await common.run_agent(
                "http://test", http, "clamped", "first", "next", max_tokens=8,
                verifier=common.make_verifier(common.MODEL_NAME), proxy_api_key=common.PROXY_API_KEY,
            )
            assert len(record["turns"]) == provider.engine_calls == 2
            assert all(turn["finish_reason"] == "length" and len(turn["completion_token_ids"]) == 3
                       for turn in record["turns"])
    print("PASS: live-agent checks accept context-limited turns below the requested budget")


def test_endpoints(root: Path) -> None:
    fixture = root / "engines.yaml"
    fixture.write_text(json.dumps({engine: {
        "OPENAI_BASE_URL": [f"http://{engine}-one/v1", f"http://{engine}-two/v1"],
        "OPENAI_API_KEY": ["fixture-key"], "MODEL_NAME": "fixture-model",
    } for engine in ("vllm", "sglang")}))
    with patch.object(common, "ENGINES_YAML", fixture), patch.dict(os.environ, {}, clear=True):
        assert common.live_test_engines() == ("vllm", "sglang")
        assert common.live_test_cases() == (("direct", "vllm"), ("direct", "sglang"), ("sglang_generate", "sglang"))
        with patch.dict(os.environ, {"TRANSPORT_MODE": "direct"}):
            assert common.live_test_cases() == (("direct", "vllm"), ("direct", "sglang"))
        for engine in common.live_test_engines():
            rows = common.load_engine_endpoints(engine)
            assert len(rows) == 2
            assert [row.api_key for row in rows] == ["fixture-key", "fixture-key"]
            assert all(row.engine == engine for row in rows)

        with patch.dict(os.environ, {"MODEL_NAME": "override-model"}):
            assert common.live_test_engines() == ("vllm",)
            assert common.load_engine_endpoints("vllm") == [common.EngineEndpoint("vllm", "http://vllm-one/v1", "fixture-key", "override-model")]
        with patch.dict(os.environ, {"INFERENCE_ENGINE": "sglang", "OPENAI_BASE_URL": "http://override/v1"}):
            assert common.live_test_engines() == ("sglang",)
            assert common.load_engine_endpoints("sglang") == [common.EngineEndpoint("sglang", "http://override/v1", "fixture-key", "fixture-model")]

        fixture.unlink()
        for engine in ("vllm", "sglang"):
            with patch.dict(os.environ, {
                "INFERENCE_ENGINE": engine, "OPENAI_BASE_URL": "http://env/v1",
                "OPENAI_API_KEY": "env-key", "MODEL_NAME": "env-model",
            }):
                assert common.live_test_engines() == (engine,)
                assert common.load_engine_endpoints(engine) == [common.EngineEndpoint(
                    engine, "http://env/v1", "env-key", "env-model",
                )]
            try:
                common.load_engine_endpoints(engine)
            except RuntimeError as exc:
                assert "No endpoint" in str(exc)
            else:
                raise AssertionError("missing endpoint silently passed preflight")
    print("PASS: shared endpoint selection, overrides, and missing-fixture errors")


def test_transport_endpoints(root: Path) -> None:
    fixture = root / "routers.yaml"
    with patch.object(common, "ENGINES_YAML", fixture), patch.dict(os.environ, {}, clear=True):
        for transport, engine in (("sglang_generate", "sglang"), ("vllm_generate", "vllm")):
            # A router-only fixture needs no unrelated direct-engine section.
            fixture.write_text(json.dumps({f"{engine}-{transport}": {
                "OPENAI_BASE_URL": "http://router/v1", "OPENAI_API_KEY": "router-key",
                "MODEL_NAME": "router-model", "CONTEXT_LENGTH": 32768,
            }}))
            with patch.dict(os.environ, {"TRANSPORT_MODE": transport}):
                assert common.live_test_cases() == ((transport, engine),)
                endpoint = common.EngineEndpoint(engine, "http://router/v1", "router-key", "router-model", 32768)
                assert common.live_test_endpoints() == [(transport, endpoint)]
                assert common.load_engine_endpoints(engine) == [endpoint]
                with patch.dict(os.environ, {"CONTEXT_LENGTH": "4096", "MODEL_NAME": "override-model"}):
                    assert common.live_test_endpoints() == [(transport, common.EngineEndpoint(
                        engine, "http://router/v1", "router-key", "override-model", 4096,
                    ))]
                with patch.dict(os.environ, {"INFERENCE_ENGINE": "vllm" if engine == "sglang" else "sglang"}):
                    try:
                        common.live_test_cases()
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("incompatible engine/transport selected for live tests")

            # Explicit transport sections win over shared engine fixtures.
            fixtures = json.loads(fixture.read_text())
            fixtures[engine] = {"OPENAI_BASE_URL": "http://direct/v1", "OPENAI_API_KEY": "direct-key", "MODEL_NAME": "direct-model"}
            fixture.write_text(json.dumps(fixtures))
            assert common.load_engine_endpoints(engine, transport) == [endpoint]
            assert common.load_engine_endpoints(engine, "direct")[0].root_url == "http://direct"

            del fixtures[f"{engine}-{transport}"]
            fixture.write_text(json.dumps(fixtures))
            assert common.load_engine_endpoints(engine, transport)[0].root_url == "http://direct"

            # Standard endpoint variables also support a router with no YAML.
            fixture.unlink()
            with patch.dict(os.environ, {
                "TRANSPORT_MODE": transport, "OPENAI_BASE_URL": "http://env/v1",
                "OPENAI_API_KEY": "EMPTY", "MODEL_NAME": "env-model", "CONTEXT_LENGTH": "8192",
            }):
                assert common.live_test_endpoints() == [(transport, common.EngineEndpoint(engine, "http://env/v1", "EMPTY", "env-model", 8192))]
    print("PASS: shared transport selection scopes live execution and preflight to the selected endpoints")


def test_canonical_fixture_defaults(root: Path) -> None:
    fixture = root / "canonical.yaml"

    def section(host):
        return {"OPENAI_BASE_URL": f"http://{host}/v1", "OPENAI_API_KEY": "test", "MODEL_NAME": "model"}

    fixtures = {f"{engine}-direct": section(f"{engine}-direct") for engine in ("sglang", "vllm")}
    fixtures.update({f"{engine}-{engine}_generate": section(f"{engine}-router") for engine in ("sglang", "vllm")})
    with patch.object(common, "ENGINES_YAML", fixture), patch.dict(os.environ, {}, clear=True):
        # Both a missing shared section and an unrelated shared endpoint must
        # select the canonical router, including during default preflight.
        for shared in (False, True):
            if shared:
                fixtures.update({engine: section(f"{engine}-shared") for engine in ("sglang", "vllm")})
            fixture.write_text(json.dumps(fixtures))
            assert [(mode, ep.root_url) for mode, ep in common.live_test_endpoints()] == [
                ("direct", "http://vllm-direct"), ("direct", "http://sglang-direct"),
                ("sglang_generate", "http://sglang-router"),
            ]
            for engine in ("sglang", "vllm"):
                assert common.load_engine_endpoints(engine, f"{engine}_generate")[0].root_url == f"http://{engine}-router"
        # Import standalone scripts without running their live gauntlets.
        import runpy
        for engine, script in (("sglang", "test-sglang_generate.py"), ("vllm", "test-vllm_generate.py")):
            with patch.dict(os.environ, {}, clear=True):
                runpy.run_path(str(Path(__file__).with_name(script)), run_name="fixture_selection_test")
                assert os.environ["TRANSPORT_MODE"] == f"{engine}_generate"
                assert common.load_engine_endpoints(engine)[0].root_url == f"http://{engine}-router"
    print("PASS: default suite and standalone scripts select canonical router fixtures")


class Tokenizer:
    eos_token_id = 999

    def apply_chat_template(self, *args, **kwargs):
        return [1, 2]

    def decode(self, ids):
        return str(ids)


class Engine:
    def __init__(self, last_stop: int, last_length: int):
        self.calls = 0
        self.last_stop, self.last_length = last_stop, last_length
        self.closed = False

    async def generate(self, *args):
        self.calls += 1
        ids = ([65, self.last_stop] if self.calls == 1
               else [66] * (contract.TRUNCATE_AT - 1) + [self.last_length])
        return common.TokenOutput(ids, [-0.1] * len(ids), "stop" if self.calls == 1 else "length")

    async def aclose(self):
        self.closed = True


async def test_eos(root: Path) -> None:
    tokenizer = Tokenizer()
    endpoint = common.EngineEndpoint("vllm", "http://unused.invalid/v1", "test", "fake-model")
    with patch.object(contract, "resolve_profile", return_value=SimpleNamespace(tokenizer_path=root)):
        assert contract.expected_eos_ids(endpoint.model, tokenizer) == {999}
        config = root / "generation_config.json"
        config.write_text(json.dumps({"eos_token_id": [900, 901]}))
        assert contract.expected_eos_ids(endpoint.model, tokenizer) == {900, 901}
        for engine in ("vllm", "sglang"):
            for stop, length, error in (
                (900, 66, None), (901, 66, None),
                (65, 66, "natural stop omitted EOS"),
                (900, 901, "truncated response ends on the EOS"),
            ):
                server = Engine(stop, length)
                with patch.object(contract, "make_engine_client", return_value=server), \
                     patch.object(contract, "load_bundled_tokenizer", return_value=tokenizer):
                    try:
                        await contract.check_engine(engine, endpoint)
                    except AssertionError as exc:
                        assert error and error in str(exc), str(exc)
                    else:
                        assert error is None, f"contract accepted invalid output: {stop=}, {length=}"
                assert server.closed
        config.write_text(json.dumps({"eos_token_id": 900}))
        assert contract.expected_eos_ids(endpoint.model, tokenizer) == {900}
        config.write_text(json.dumps({"eos_token_id": []}))
        try:
            contract.expected_eos_ids(endpoint.model, tokenizer)
        except ValueError:
            pass
        else:
            raise AssertionError("contract accepted empty EOS configuration")
    print("PASS: EOS is checked against configuration, including alternate stop tokens")


if __name__ == "__main__":
    test_agent_selection()
    asyncio.run(test_native_finish_reason())
    asyncio.run(test_context_clamped_agent())
    with TemporaryDirectory() as directory:
        root = Path(directory)
        test_endpoints(root)
        test_transport_endpoints(root)
        test_canonical_fixture_defaults(root)
        asyncio.run(test_eos(root))
