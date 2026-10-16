"""Offline startup contracts for HTTP transports."""

from __future__ import annotations
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxyserver.cli import main as cli_main, run_standalone_proxy
from proxyserver.config import TRANSPORT_MODES, default_config_path, load_config, resolve_transport_mode
from proxyserver.server import LLMProxyServer, make_completion_handler
from offline_common import ScriptedProvider, check


def config_file(root: Path, transport: str, engine: str | None = None) -> Path:
    path = root / f"{transport}.yaml"
    engine = engine or ("vllm" if transport in ("vime", "verl") else "sglang")
    path.write_text(f"inference_engine: {engine}\ntransport_mode: {transport}\n")
    return path


def test_required_configuration() -> None:
    with patch.dict(os.environ, {"TRANSPORT_MODE": "", "INFERENCE_ENGINE": ""}):
        check("supported transports", set(TRANSPORT_MODES) == {"direct", "slime", "vime", "verl"})
        check("default transport is direct", resolve_transport_mode() == "direct")
        check("default config exists", default_config_path().is_file())
        for engine in ("sglang", "vllm"):
            with patch.dict(os.environ, {"TRANSPORT_MODE": "verl", "INFERENCE_ENGINE": engine}):
                cfg = load_config()
                assert cfg.transport_mode == "verl" and cfg.inference_engine == engine
                assert cfg.router_url is None and cfg.return_routed_experts is False
                assert default_config_path().name == f"{engine}-verl.yaml"
        check("both VeRL HTTP templates selected through environment", True)
    with patch.object(sys, "argv", ["proxy", "--transport-mode", "verl", "--router-url", "http://verl:8000"]), \
            patch("proxyserver.cli.run_standalone_proxy") as run:
        cli_main()
        assert run.call_args.kwargs["transport_mode"] == "verl"
        assert run.call_args.kwargs["router_url"] == "http://verl:8000"
    subprocess.run([sys.executable, "-c", """
import asyncio
import sys
sys.modules['ray'] = None
from proxyserver.cli import main
from proxyserver.rollout_provider import SGLangRolloutProvider, VLLMRolloutProvider
async def run():
    for cls in (SGLangRolloutProvider, VLLMRolloutProvider):
        provider = cls('http://verl', context_length=32768)
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
        for transport, field in (("direct", "inference_engine_base_url"), ("slime", "router_url"), ("vime", "router_url"), ("verl", "router_url")):
            config = config_file(root, transport)
            # Endpoints may be supplied later by CLI flags, so YAML loads.
            check(f"{transport} config loads before overrides", load_config(config).transport_mode == transport)
            check(f"{transport} sampling-mask persistence defaults off", not load_config(config).save_rollout_sampling_masks)
            for enabled in (True, False):
                config = config_file(root, transport)
                with config.open("a") as f:
                    f.write(f"save_rollout_sampling_masks: {str(enabled).lower()}\n")
                check(f"{transport} sampling-mask persistence parses {enabled}",
                      load_config(config).save_rollout_sampling_masks is enabled)
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
        for key in ("ray_address", "ray_namespace", "verl_load_balancer"):
            config = config_file(root, "vime")
            with config.open("a") as f:
                f.write(f"{key}: old-setting\n")
            try:
                load_config(config)
            except ValueError as exc:
                assert "unsupported proxy settings" in str(exc)
            else:
                raise AssertionError("unsupported setting silently ignored")
        for engine in ("sglang", "vllm"):
            config = config_file(root, "verl", engine)
            with config.open("a") as f:
                f.write("context_length: 8192\nreturn_routed_experts: true\n")
            cfg = load_config(config)
            assert cfg.context_length == 8192 and cfg.return_routed_experts is True


def test_cli_provider_wiring() -> None:
    """CLI endpoint overrides install a provider and close it on shutdown."""
    with tempfile.TemporaryDirectory() as tmp:
        for transport, engine, provider_class, target in (
            ("direct", "sglang", "DirectRolloutProvider", {"inference_engine_base_url": ["http://engine:8000/v1"]}),
            ("slime", "sglang", "SGLangRolloutProvider", {"router_url": "http://router:32005"}),
            ("vime", "vllm", "VLLMRolloutProvider", {"router_url": "http://router:30000"}),
            ("verl", "sglang", "SGLangRolloutProvider", {"router_url": "http://verl:30000", "context_length": 8192}),
            ("verl", "vllm", "VLLMRolloutProvider", {"router_url": "http://verl:8000", "context_length": 8192}),
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
                with config.open("a") as f:
                    f.write("save_rollout_sampling_masks: true\n")
                run_standalone_proxy(config=config, api_key="test", **target)
            check(f"{transport} provider constructed once", factory.call_count == 1)
            expected_target = target.get("router_url", target.get("inference_engine_base_url"))
            target_arg = "router_url" if transport in ("slime", "vime", "verl") else "base_urls"
            check(f"{transport} CLI target reaches provider", factory.call_args.kwargs[target_arg] == expected_target)
            assert factory.call_args.kwargs["inference_engine"] == engine
            if transport == "verl":
                assert factory.call_args.kwargs["context_length"] == 8192
            kwargs = state["kwargs"]
            check(f"{transport} sampling-mask persistence reaches server", kwargs["save_rollout_sampling_masks"] is True)
            check(f"{transport} handler installed", callable(kwargs["completion_handler"]))
            check(f"{transport} deletion releases provider", kwargs["on_session_deleted"] == provider.release_session)
            check(f"{transport} sampling and fingerprints wired",
                  kwargs["get_sampling_overrides"] == provider.get_sampling_overrides
                  and kwargs["get_tokenizer_fingerprint"] == provider.tokenizer_fingerprint)
            check(f"{transport} closes server and provider", state["stopped"] and state["closed"])
            if transport in ("slime", "vime", "verl"):
                check(f"{transport} routing controls wired",
                      kwargs["get_routed_experts_config"] == provider.get_routed_experts_config
                      and kwargs["set_routed_experts_config"] == provider.set_runtime_routed_experts)


def test_persistence_layout() -> None:
    timestamp = "0000-11-22-33-44-55"
    with tempfile.TemporaryDirectory() as tmp, \
            patch("time.strftime", return_value=timestamp):
        root = Path(tmp)
        proxy = LLMProxyServer(
            completion_handler=make_completion_handler(ScriptedProvider()),
            session_dir=root / "sessions",
            error_log_dir=root / "logs",
            save_rollout_sampling_masks=True,
        )
        proxy.recorder.ensure_session("trial", model_name="test-model")
        proxy.recorder.record_completion(
            "trial", [{"role": "user", "content": "hello"}], "hi", [2], [-0.1],
            finish_reason="stop", prompt_token_ids=[1], agent_id="a",
            rollout_top_p_token_ids=[2, 3], rollout_top_p_token_offsets=[0, 2],
        )
        proxy.error_log.log("trial", "generation failed")
        session_path = root / "sessions" / timestamp / "trial.json"
        log_path = root / "logs" / timestamp / "trial.log"
        check("session JSON is directly under the timestamp directory",
              session_path.is_file() and json.loads(session_path.read_text())["session_id"] == "trial")
        turn = json.loads(session_path.read_text())["turns"][0]
        check("server forwards sampling-mask persistence to the recorder",
              turn["rollout_top_p_token_ids"] == [2, 3] and turn["rollout_top_p_token_offsets"] == [0, 2])
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
        check("health reports process liveness", response.json() == {"status": "ok"})


if __name__ == "__main__":
    test_required_configuration()
    test_cli_provider_wiring()
    test_persistence_layout()
    asyncio.run(test_http_surface())
    print("PASS: required targets, provider wiring, cleanup, and HTTP surface")
