"""Offline startup contracts for HTTP and VeRL transports."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxyserver.cli import run_standalone_proxy
from proxyserver.config import TRANSPORT_MODES, default_config_path, load_config, resolve_transport_mode
from proxyserver.server import LLMProxyServer, make_completion_handler
from offline_common import ScriptedProvider, check


def config_file(root: Path, transport: str) -> Path:
    path = root / f"{transport}.yaml"
    path.write_text(f"inference_engine: sglang\ntransport_mode: {transport}\n")
    return path


def test_required_configuration() -> None:
    with patch.dict(os.environ, {"TRANSPORT_MODE": "", "INFERENCE_ENGINE": ""}):
        check("supported transports", set(TRANSPORT_MODES) == {"direct", "slime", "verl"})
        check("default transport is direct", resolve_transport_mode() == "direct")
        check("default config exists", default_config_path().is_file())
        for engine in ("vllm", "sglang"):
            cfg = load_config(inference_engine=engine, transport_mode="verl")
            check(f"{engine} VeRL template selects Ray transport",
                  cfg.inference_engine == engine and cfg.transport_mode == "verl"
                  and cfg.ray_address == "auto" and cfg.verl_load_balancer is None)
    try:
        LLMProxyServer()
    except ValueError as e:
        check("server requires a completion handler", "completion_handler" in str(e))
    else:
        raise AssertionError("server accepted no provider handler")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for transport, field in (("direct", "inference_engine_base_url"), ("slime", "router_url"), ("verl", "verl_load_balancer")):
            config = config_file(root, transport)
            # Endpoints may be supplied later by CLI flags, so YAML loads.
            check(f"{transport} config loads before overrides", load_config(config).transport_mode == transport)
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
        for option in ({"context_length": 1024}, {"return_routed_experts": True}):
            try:
                run_standalone_proxy(config=config_file(root, "verl"), api_key="test",
                                     verl_load_balancer="trainer", **option)
            except ValueError:
                pass
            else:
                raise AssertionError("VeRL silently ignored a Slime-only option")
        config = config_file(root, "verl")
        with config.open("a") as f:
            f.write("ray_address: cluster:6379\nray_namespace: training\nverl_load_balancer: trainer\n")
        cfg = load_config(config)
        check("VeRL YAML preserves all Ray connection fields",
              (cfg.ray_address, cfg.ray_namespace, cfg.verl_load_balancer) == ("cluster:6379", "training", "trainer"))


def test_cli_provider_wiring() -> None:
    """CLI endpoint overrides install a provider and close it on shutdown."""
    with tempfile.TemporaryDirectory() as tmp:
        for transport, provider_class, target in (
            ("direct", "DirectRolloutProvider", {"inference_engine_base_url": ["http://engine:8000/v1"]}),
            ("slime", "SlimeRolloutProvider", {"router_url": "http://router:32005"}),
            ("verl", "RayRolloutProvider", {"verl_load_balancer": "trainer", "ray_address": "auto", "ray_namespace": "training"}),
            ("verl", "RayRolloutProvider", {"load_balancer": object()}),
        ):
            provider = ScriptedProvider()
            provider.base_urls = ["http://engine:8000"]
            state = {}

            async def close():
                state["closed"] = True

            provider.aclose = close
            provider.start = lambda: state.update(connected=True)
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
                run_standalone_proxy(config=config_file(Path(tmp), transport), api_key="test", **target)
            check(f"{transport} provider constructed once", factory.call_count == 1)
            if transport == "verl":
                check("VeRL connects before serving", state["connected"])
                for name, value in target.items():
                    check(f"VeRL {name} reaches provider", factory.call_args.kwargs[name] == value)
            else:
                expected_target = target.get("router_url", target.get("inference_engine_base_url"))
                target_arg = "router_url" if transport == "slime" else "base_urls"
                check(f"{transport} CLI target reaches provider", factory.call_args.kwargs[target_arg] == expected_target)
            kwargs = state["kwargs"]
            check(f"{transport} handler installed", callable(kwargs["completion_handler"]))
            check(f"{transport} deletion releases provider", kwargs["on_session_deleted"] == provider.release_session)
            check(f"{transport} sampling and fingerprints wired",
                  kwargs["get_sampling_overrides"] == provider.get_sampling_overrides
                  and kwargs["get_tokenizer_fingerprint"] == provider.tokenizer_fingerprint)
            check(f"{transport} closes server and provider", state["stopped"] and state["closed"])


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
        check("health reports process liveness", response.json() == {"status": "ok"})


if __name__ == "__main__":
    test_required_configuration()
    test_cli_provider_wiring()
    test_persistence_layout()
    asyncio.run(test_http_surface())
    print("PASS: required targets, provider wiring, cleanup, and HTTP surface")
