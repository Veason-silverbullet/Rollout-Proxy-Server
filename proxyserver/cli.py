"""Standalone recording proxy with an in-process inference provider.

Use ``--transport-mode direct`` with ``--inference-engine-base-url`` for
vLLM/SGLang endpoints. Use ``--transport-mode sglang_generate`` for the
SGLang /generate protocol or ``--transport-mode vllm_generate`` for
/inference/v1/generate, with ``--router-url`` naming the HTTP server.
Endpoint settings may also come from YAML.
The default transport is ``direct``; explicit CLI flags override config.
"""

from __future__ import annotations
import asyncio
import logging
import os
import signal
import threading
from pathlib import Path
from .config import (
    TRANSPORT_ENGINES, TRANSPORT_MODES, ProxyConfig, load_config,
    resolve_transport_engine, resolve_transport_mode
)
from .service import LLMProxyServer, make_completion_handler

logger = logging.getLogger(__name__)


def run_standalone_proxy(
    host: str | None = None,
    port: int | None = None,
    log_level: str = "INFO",
    api_key: str | None = None,
    config: str | Path | ProxyConfig | None = None,
    transport_mode: str | None = None,
    router_url: str | None = None,
    router_api_key: str | None = None,
    context_length: int | None = None,
    return_routed_experts: bool | None = None,
    inference_engine_base_url: list[str] | None = None,
    inference_engine_api_key: list[str] | None = None,
) -> None:
    """Run until interrupted, requiring a configured inference target.

    Explicit arguments override the corresponding YAML settings. The API
    key defaults to ``PROXY_API_KEY``; host and port default to 0.0.0.0:9400.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if isinstance(config, ProxyConfig):
        cfg = config
    else:
        cfg = load_config(config, transport_mode=transport_mode, override_transport=True)
    logger.info(
        "Loaded config %s (inference_engine=%s, delimiter=%r)",
        cfg.path, cfg.inference_engine, cfg.proxy_api_delimiter,
    )
    # Bind address: explicit argument/CLI flag > config > built-in default.
    if host is None:
        host = cfg.host if cfg.host is not None else "0.0.0.0"
    if port is None:
        port = cfg.port if cfg.port is not None else 9400

    api_key = api_key if api_key is not None else os.getenv("PROXY_API_KEY")
    if not api_key:
        logger.warning(
            "No api_key configured (set --api-key or PROXY_API_KEY); "
            "all completion requests will be rejected with 401"
        )

    # Transport wiring: explicit arguments override the config.
    engine_setting = None if transport_mode and cfg.inference_engine_inferred else cfg.inference_engine
    transport_mode = resolve_transport_mode(transport_mode or cfg.transport_mode)
    engine = resolve_transport_engine(engine_setting, transport_mode)
    router_url = router_url or cfg.router_url
    router_api_key = router_api_key or cfg.router_api_key
    if context_length is None:
        context_length = cfg.context_length
    if return_routed_experts is None:
        return_routed_experts = cfg.return_routed_experts
    inference_engine_base_url = inference_engine_base_url or cfg.inference_engine_base_url
    inference_engine_api_key = inference_engine_api_key or cfg.inference_engine_api_key

    # Validate before constructing clients or starting the HTTP server.
    if transport_mode in TRANSPORT_ENGINES and not (router_url and router_url.strip()):
        raise ValueError(f"transport_mode={transport_mode!r} requires router_url (--router-url)")
    if transport_mode == "direct" and not inference_engine_base_url:
        raise ValueError("transport_mode='direct' requires inference_engine_base_url (--inference-engine-base-url)")
    from .rollout_provider import DirectRolloutProvider, SGLangRolloutProvider, VLLMRolloutProvider

    sampling_overrides = cfg.sampling_overrides
    get_routed_experts_config = None
    set_routed_experts_config = None
    if transport_mode == "sglang_generate":
        provider = SGLangRolloutProvider(
            router_url=router_url,
            inference_engine=engine,
            api_key=router_api_key,
            sampling_overrides=sampling_overrides,
            context_length=context_length,
            return_routed_experts=bool(return_routed_experts),
        )
        get_routed_experts_config = provider.get_routed_experts_config
        set_routed_experts_config = provider.set_runtime_routed_experts
        transport_target = f"SGLang /generate at {router_url}"
    elif transport_mode == "vllm_generate":
        provider = VLLMRolloutProvider(
            router_url=router_url,
            inference_engine=engine,
            api_key=router_api_key,
            sampling_overrides=sampling_overrides,
            context_length=context_length,
            return_routed_experts=bool(return_routed_experts),
        )
        get_routed_experts_config = provider.get_routed_experts_config
        set_routed_experts_config = provider.set_runtime_routed_experts
        transport_target = f"vLLM /inference/v1/generate at {router_url}"
    else:
        provider = DirectRolloutProvider(
            base_urls=inference_engine_base_url,
            inference_engine=engine,
            api_keys=inference_engine_api_key,
            sampling_overrides=sampling_overrides,
        )
        transport_target = f"{provider.engine} engine(s) at {', '.join(provider.base_urls)}"

    proxy = LLMProxyServer(
        host=host,
        port=port,
        api_key=api_key,
        save_rollout_sessions=cfg.save_rollout_sessions,
        save_rollout_logprobs=cfg.save_rollout_logprobs,
        session_dir=cfg.rollout_session_dir,
        error_log_dir=cfg.log_dir,
        key_delimiter=cfg.proxy_api_delimiter,
        completion_handler=make_completion_handler(provider),
        on_session_deleted=provider.release_session,
        get_sampling_overrides=provider.get_sampling_overrides,
        set_sampling_overrides=provider.set_runtime_sampling_overrides,
        get_routed_experts_config=get_routed_experts_config,
        set_routed_experts_config=set_routed_experts_config,
        get_tokenizer_fingerprint=provider.tokenizer_fingerprint,
        save_rollout_routed_experts=cfg.save_rollout_routed_experts,
    )

    async def _run():
        shutdown = asyncio.Event()
        previous_handlers = {}

        def request_shutdown(signum, frame):
            shutdown.set()

        try:
            # Uvicorn captures and then re-raises process signals. Keep our
            # handlers installed until detached work and provider cleanup finish.
            if threading.current_thread() is threading.main_thread():
                for sig in (signal.SIGINT, signal.SIGTERM):
                    previous_handlers[sig] = signal.signal(sig, request_shutdown)
            url = await proxy.start()
            logger.info(
                "Standalone proxy started at %s (%s)\n"
                "  Health check:       %s/health\n"
                "  Session records:    %s",
                url, transport_target, url,
                proxy.session_store.run_dir if proxy.session_store is not None else "disabled",
            )
            await shutdown.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            try:
                try:
                    await proxy.stop()
                finally:
                    await provider.aclose()
            finally:
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


def main() -> None:
    """CLI entry point for running the proxy as a standalone service."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the LLM Proxy as a standalone recording service.")
    parser.add_argument(
        "--config", default=None,
        help=(
            "Path to a config YAML (default: bundled configs/"
            "{engine}-{transport}.yaml; generation transports infer their engine)"
        ),
    )
    parser.add_argument(
        "--host", default=None,
        help="Bind address (default: the config's host, else 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port number (default: the config's port, else 9400)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Shared secret for keyed routing (default: $PROXY_API_KEY)",
    )
    parser.add_argument(
        "--transport-mode", default=None, choices=TRANSPORT_MODES,
        help=(
            "How token-ID prompts reach the rollout engines; also selects the "
            "default config file (default: $TRANSPORT_MODE, else direct)"
        ),
    )
    parser.add_argument(
        "--router-url", default=None,
        help="Generate protocols: base URL of the router or HTTP server (default: the config's router_url)",
    )
    parser.add_argument(
        "--router-api-key", default=None,
        help="Generate protocols: optional bearer token for the HTTP target (default: the config's router_api_key)",
    )
    parser.add_argument(
        "--context-length", type=int, default=None,
        help=(
            "Generation transports: the engines' context window in tokens; "
            "match the deployed engine's configured limit. "
            "Pins the max_tokens clamp without router discovery "
            "(default: the config's context_length)"
        ),
    )
    parser.add_argument(
        "--return-routed-experts", default=None, action=argparse.BooleanOptionalAction,
        help=(
            "Generate protocols: ask the engines for per-token MoE expert selections "
            "and record the latest capture per agent stream (R3). Requires "
            "engines launched with the framework's routing-replay support "
            "(default: the config's return_routed_experts)"
        ),
    )
    parser.add_argument(
        "--inference-engine-base-url", default=None, nargs="+", metavar="URL",
        help=(
            "direct: OpenAI base URLs of the vLLM/SGLang engines, one per "
            "instance; required (default: the config's "
            "inference_engine_base_url)"
        ),
    )
    parser.add_argument(
        "--inference-engine-api-key", default=None, nargs="+", metavar="KEY",
        help=(
            "direct: api_keys matching the base URLs — one per URL or a single "
            "shared key (default: the config's inference_engine_api_key)"
        ),
    )
    args = parser.parse_args()

    run_standalone_proxy(
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        api_key=args.api_key,
        config=args.config,
        transport_mode=args.transport_mode,
        router_url=args.router_url,
        router_api_key=args.router_api_key,
        context_length=args.context_length,
        return_routed_experts=args.return_routed_experts,
        inference_engine_base_url=args.inference_engine_base_url,
        inference_engine_api_key=args.inference_engine_api_key,
    )


if __name__ == "__main__":
    main()
