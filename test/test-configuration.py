"""Offline startup contracts for HTTP transports."""

from __future__ import annotations
import asyncio
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxyserver.cli import main as cli_main, run_standalone_proxy
from proxyserver.config import TRANSPORT_MODES, default_config_path, load_config, resolve_transport_mode
from proxyserver.service import LLMProxyServer, make_completion_handler
from offline_common import ScriptedProvider, check


def config_file(root: Path, transport: str, engine: str | None = None) -> Path:
    path = root / f"{transport}.yaml"
    engine = engine or ("vllm" if transport == "vllm_generate" else "sglang")
    path.write_text(f"inference_engine: {engine}\ntransport_mode: {transport}\n")
    return path


def test_removed_transport_names() -> None:
    """Removed names must fail through API, YAML, environment, and CLI."""
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
        for name in ("slime", "vime", "verl"):
            for engine in ("sglang", "vllm"):
                config = config_file(Path(tmp), name, engine)
                with patch.dict(os.environ, {"TRANSPORT_MODE": name, "INFERENCE_ENGINE": engine}):
                    for call in (
                        lambda: resolve_transport_mode(name),
                        lambda: load_config(config),
                        load_config,
                        lambda: default_config_path(transport_mode=name),
                        lambda: run_standalone_proxy(transport_mode=name, api_key="test"),
                    ):
                        try:
                            call()
                        except ValueError as exc:
                            assert "Unknown transport_mode" in str(exc)
                        else:
                            raise AssertionError(f"Removed transport {name!r} was accepted")
            with patch.object(sys, "argv", ["proxy", "--transport-mode", name]), \
                    patch("proxyserver.cli.run_standalone_proxy") as run, \
                    redirect_stderr(io.StringIO()):
                try:
                    cli_main()
                except SystemExit as exc:
                    assert exc.code == 2
                else:
                    raise AssertionError(f"CLI accepted removed transport {name!r}")
                run.assert_not_called()
    check("removed transport names fail at every configuration entry point", True)


def test_required_configuration() -> None:
    with patch.dict(os.environ, {"TRANSPORT_MODE": "", "INFERENCE_ENGINE": ""}):
        check("supported transports", set(TRANSPORT_MODES) == {"direct", "sglang_generate", "vllm_generate"})
        check("default transport is direct", resolve_transport_mode() == "direct")
        check("default config exists", default_config_path().is_file())
        for engine in ("sglang", "vllm"):
            with patch.dict(os.environ, {"TRANSPORT_MODE": f"{engine}_generate", "INFERENCE_ENGINE": engine}):
                cfg = load_config()
                assert cfg.transport_mode == f"{engine}_generate" and cfg.inference_engine == engine
                assert cfg.router_url is None and cfg.return_routed_experts is True
                assert cfg.context_length == 262144
                assert default_config_path().name == f"{engine}-{engine}_generate.yaml"
        check("environment selects both generation protocol templates", True)
    with patch.object(sys, "argv", ["proxy", "--transport-mode", "vllm_generate", "--router-url", "http://server:8000"]), \
            patch("proxyserver.cli.run_standalone_proxy") as run:
        cli_main()
        assert run.call_args.kwargs["transport_mode"] == "vllm_generate"
        assert run.call_args.kwargs["router_url"] == "http://server:8000"
    subprocess.run([sys.executable, "-c", """
import asyncio
import sys
sys.modules['ray'] = None
from proxyserver.cli import main
from proxyserver.rollout_provider import SGLangRolloutProvider, VLLMRolloutProvider
async def run():
    for cls in (SGLangRolloutProvider, VLLMRolloutProvider):
        provider = cls('http://server', context_length=32768)
        await provider.aclose()
asyncio.run(run())
"""], check=True, cwd=Path(__file__).resolve().parents[1])
    check("proxy imports and shared providers work without Ray", True)
    try:
        LLMProxyServer()
    except ValueError as e:
        check("server requires a completion handler", "completion_handler" in str(e))
    else:
        raise AssertionError("server accepted no provider handler")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for transport, field in (("direct", "inference_engine_base_url"), ("sglang_generate", "router_url"), ("vllm_generate", "router_url")):
            config = config_file(root, transport)
            # Endpoints may be supplied later by CLI flags, so YAML loads.
            cfg = load_config(config)
            check(f"{transport} config loads before overrides",
                  cfg.transport_mode == resolve_transport_mode(transport))
            for value in (config, load_config(config)):
                try:
                    run_standalone_proxy(config=value, api_key="test")
                except ValueError as e:
                    check(f"{transport} missing target fails before startup", field in str(e))
                else:
                    raise AssertionError("missing inference target was accepted")
        config = config_file(root, "unsupported")
        try:
            load_config(config)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported transport was accepted")
        for engine in ("sglang", "vllm"):
            config = config_file(root, f"{engine}_generate", engine)
            with config.open("a") as f:
                f.write("context_length: 8192\nreturn_routed_experts: true\n")
            cfg = load_config(config)
            assert cfg.context_length == 8192 and cfg.return_routed_experts is True


def test_cli_provider_wiring() -> None:
    """CLI endpoint overrides install a provider and close it on shutdown."""
    with tempfile.TemporaryDirectory() as tmp:
        for transport, engine, provider_class, target in (
            ("direct", "sglang", "DirectRolloutProvider", {"inference_engine_base_url": ["http://engine:8000/v1"]}),
            ("sglang_generate", "sglang", "SGLangRolloutProvider", {"router_url": "http://router:32000", "context_length": 8192}),
            ("vllm_generate", "vllm", "VLLMRolloutProvider", {"router_url": "http://router:30000", "context_length": 8192}),
        ):
            provider = ScriptedProvider()
            provider.base_urls = ["http://engine:8000"]
            state = {}

            async def close():
                state["closed"] = True

            provider.aclose = close
            provider.get_routed_experts_config = lambda: {"effective": False}
            provider.set_runtime_routed_experts = lambda value: {"effective": value}

            class Server:
                session_store = None

                def __init__(self, **kwargs):
                    state["kwargs"] = kwargs

                async def start(self):
                    raise asyncio.CancelledError

                async def stop(self):
                    state["stopped"] = True

            with patch(f"proxyserver.rollout_provider.{provider_class}", return_value=provider) as factory, \
                    patch("proxyserver.cli.LLMProxyServer", Server):
                config = config_file(Path(tmp), transport, engine)
                handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
                run_standalone_proxy(config=config, api_key="test", **target)
                assert {sig: signal.getsignal(sig) for sig in handlers} == handlers
            check(f"{transport} provider constructed once", factory.call_count == 1)
            expected_target = target.get("router_url", target.get("inference_engine_base_url"))
            target_arg = "router_url" if transport != "direct" else "base_urls"
            check(f"{transport} CLI target reaches provider", factory.call_args.kwargs[target_arg] == expected_target)
            assert factory.call_args.kwargs["inference_engine"] == engine
            if transport != "direct":
                assert factory.call_args.kwargs["context_length"] == 8192
            kwargs = state["kwargs"]
            check(f"{transport} handler installed", callable(kwargs["completion_handler"]))
            check(f"{transport} deletion releases provider", kwargs["on_session_deleted"] == provider.release_session)
            check(f"{transport} sampling and fingerprints wired",
                  kwargs["get_sampling_overrides"] == provider.get_sampling_overrides
                  and kwargs["get_tokenizer_fingerprint"] == provider.tokenizer_fingerprint)
            check(f"{transport} closes server and provider", state["stopped"] and state["closed"])
            if transport != "direct":
                check(f"{transport} routing controls wired",
                      kwargs["get_routed_experts_config"] == provider.get_routed_experts_config
                      and kwargs["set_routed_experts_config"] == provider.set_runtime_routed_experts)
            if transport in ("sglang_generate", "vllm_generate"):
                # Every transport ships a bundled config; a missing one is a broken install.
                with patch("proxyserver.cli.load_config", side_effect=FileNotFoundError):
                    try:
                        run_standalone_proxy(transport_mode=transport, api_key="test", router_url=target["router_url"])
                    except FileNotFoundError:
                        check(f"{transport} missing bundled config fails loudly", True)
                    else:
                        raise AssertionError("missing bundled config was ignored")


def test_engine_inference_and_defaults() -> None:
    with tempfile.TemporaryDirectory() as tmp, \
            patch.dict(os.environ, {"INFERENCE_ENGINE": "", "TRANSPORT_MODE": ""}):
        path = Path(tmp) / "config.yaml"
        for mode, engine in (("sglang_generate", "sglang"), ("vllm_generate", "vllm")):
            canonical = f"{engine}_generate"
            other = "vllm" if engine == "sglang" else "sglang"
            path.write_text(f"transport_mode: {mode}\n")
            cfg = load_config(path)
            assert (cfg.inference_engine, cfg.transport_mode) == (engine, canonical)
            bundled = load_config(transport_mode=mode)
            assert bundled.inference_engine == engine
            assert bundled.context_length == 262144 and bundled.return_routed_experts is True
            assert default_config_path(transport_mode=mode).name == f"{engine}-{canonical}.yaml"
            with patch.dict(os.environ, {"TRANSPORT_MODE": mode}):
                assert load_config().inference_engine == engine
            with patch.dict(os.environ, {"INFERENCE_ENGINE": other}):
                for call in (lambda: load_config(path), lambda: default_config_path(transport_mode=mode)):
                    try:
                        call()
                    except ValueError as exc:
                        assert "requires inference_engine" in str(exc)
                    else:
                        raise AssertionError("conflicting environment engine accepted")
                # Explicit engine wins over the environment.
                assert load_config(path, inference_engine=engine).inference_engine == engine
                path.write_text(f"transport_mode: {mode}\ninference_engine: {engine}\n")
                assert load_config(path).inference_engine == engine
            path.write_text(f"transport_mode: {mode}\ninference_engine: {other}\n")
            try:
                load_config(path)
            except ValueError as exc:
                assert "requires inference_engine" in str(exc)
            else:
                raise AssertionError("conflicting YAML engine accepted")
        assert load_config().inference_engine == "vllm"
        # An overriding CLI mode must also reject an incompatible config engine.
        path.write_text("transport_mode: direct\ninference_engine: vllm\n")
        with patch("proxyserver.cli.LLMProxyServer") as server:
            try:
                run_standalone_proxy(config=path, transport_mode="sglang_generate",
                                     router_url="http://router", api_key="test")
            except ValueError as exc:
                assert "requires inference_engine" in str(exc)
            else:
                raise AssertionError("conflicting CLI transport accepted")
            server.assert_not_called()
    check("engine inference, precedence, conflicts, and bundled defaults", True)


def test_cli_transport_override_inference() -> None:
    class ProviderSelected(Exception):
        pass

    with tempfile.TemporaryDirectory() as tmp, \
            patch.dict(os.environ, {"INFERENCE_ENGINE": "", "TRANSPORT_MODE": ""}):
        path = Path(tmp) / "config.yaml"
        for original, selected, engine, provider_class in (
            ("vllm_generate", "sglang_generate", "sglang", "SGLangRolloutProvider"),
            ("sglang_generate", "vllm_generate", "vllm", "VLLMRolloutProvider"),
            ("direct", "sglang_generate", "sglang", "SGLangRolloutProvider"),
            ("sglang_generate", "direct", "vllm", "DirectRolloutProvider"),
        ):
            path.write_text(f"transport_mode: {original}\n")
            cfg = load_config(path)
            assert cfg.inference_engine_inferred
            # The loader keeps its file-first contract unless override is requested.
            assert load_config(path, transport_mode=selected).transport_mode == original
            for value in (path, cfg):
                with patch(f"proxyserver.rollout_provider.{provider_class}", side_effect=ProviderSelected) as factory:
                    try:
                        run_standalone_proxy(
                            config=value, transport_mode=selected, api_key="test",
                            router_url="http://router", inference_engine_base_url=["http://engine/v1"],
                        )
                    except ProviderSelected:
                        pass
                    else:
                        raise AssertionError("expected provider was not selected")
                    assert factory.call_args.kwargs["inference_engine"] == engine

        # Resolve the final mode before validating an environment-selected engine.
        path.write_text("transport_mode: vllm_generate\n")
        with patch.dict(os.environ, {"INFERENCE_ENGINE": "sglang"}), \
                patch("proxyserver.rollout_provider.SGLangRolloutProvider", side_effect=ProviderSelected):
            try:
                run_standalone_proxy(config=path, transport_mode="sglang_generate",
                                     router_url="http://router", api_key="test")
            except ProviderSelected:
                pass

        # Explicit engines from YAML, API arguments, or environment stay binding.
        path.write_text("transport_mode: direct\ninference_engine: vllm\n")
        explicit_yaml = load_config(path)
        path.write_text("transport_mode: direct\n")
        explicit_argument = load_config(path, inference_engine="vllm")
        with patch.dict(os.environ, {"INFERENCE_ENGINE": "vllm"}):
            explicit_env = load_config(path)
        for cfg in (explicit_yaml, explicit_argument, explicit_env):
            assert not cfg.inference_engine_inferred
            try:
                run_standalone_proxy(config=cfg, transport_mode="sglang_generate",
                                     router_url="http://router", api_key="test")
            except ValueError as exc:
                assert "requires inference_engine" in str(exc)
            else:
                raise AssertionError("explicit engine was overridden")
    check("CLI infers engines after transport overrides and preserves explicit engines", True)


def test_persistence_layout() -> None:
    timestamp = "0000-11-22-33-44-55"
    with tempfile.TemporaryDirectory() as tmp, \
            patch("time.strftime", return_value=timestamp):
        root = Path(tmp)
        proxy = LLMProxyServer(
            completion_handler=make_completion_handler(ScriptedProvider()),
            session_dir=root / "sessions",
            error_log_dir=root / "logs",
        )
        proxy.recorder.ensure_session("trial", model_name="test-model")
        proxy.recorder.record_completion(
            "trial", [{"role": "user", "content": "hello"}], "hi", [2], [-0.1],
            finish_reason="stop", prompt_token_ids=[1], agent_id="a",
        )
        proxy.error_log.log("trial", "generation failed")
        session_path = root / "sessions" / timestamp / "trial.json"
        log_path = root / "logs" / timestamp / "trial.log"
        check("session JSON is directly under the timestamp directory",
              session_path.is_file() and json.loads(session_path.read_text())["session_id"] == "trial")
        check("error log is directly under the timestamp directory",
              log_path.is_file() and "generation failed" in log_path.read_text())


async def test_http_surface() -> None:
    import httpx
    from starlette.routing import WebSocketRoute

    provider = ScriptedProvider()
    proxy = LLMProxyServer(
        api_key="test", save_rollout_sessions=False,
        completion_handler=make_completion_handler(provider),
    )
    check("only HTTP routes are registered", not any(isinstance(r, WebSocketRoute) for r in proxy.app.routes))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
        response = await client.get("/health")
        check("health reports process liveness and identity", response.json() == {"status": "ok", "instance_id": proxy.instance_id})


if __name__ == "__main__":
    test_removed_transport_names()
    test_required_configuration()
    test_engine_inference_and_defaults()
    test_cli_transport_override_inference()
    test_cli_provider_wiring()
    test_persistence_layout()
    asyncio.run(test_http_surface())
    print("PASS: required targets, provider wiring, cleanup, and HTTP surface")
