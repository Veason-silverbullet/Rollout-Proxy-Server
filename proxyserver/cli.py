"""Standalone recording proxy with an in-process inference provider.

Use ``--transport-mode direct`` with ``--inference-engine-base-url`` for
vLLM/SGLang endpoints, or ``--transport-mode slime`` with ``--router-url``
for Slime's SGLang router. Endpoint settings may also come from YAML.
Use ``--transport-mode verl`` with a named VeRL Ray load-balancer actor.
The default transport is ``direct``; explicit CLI flags override config.
"""

from __future__ import annotations
import asyncio
import logging
import os
from pathlib import Path
from .config import ProxyConfig, load_config, resolve_transport_mode
from .server import LLMProxyServer

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
    tool_parser_factory=None,
    load_balancer=None,
    verl_load_balancer: str | None = None,
    ray_address: str | None = None,
    ray_namespace: str | None = None,
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
        try:
            cfg = load_config(config, transport_mode=transport_mode)
        except FileNotFoundError:
            if config is not None:
                raise  # an explicitly named config must exist
            cfg = None
            logger.warning("No default config file found; using built-in defaults")
    if cfg is not None:
        logger.info(
            "Loaded config %s (inference_engine=%s, delimiter=%r)",
            cfg.path, cfg.inference_engine, cfg.proxy_api_delimiter,
        )
    # Bind address: explicit argument/CLI flag > config > built-in default.
    if host is None:
        host = cfg.host if cfg is not None and cfg.host is not None else "0.0.0.0"
    if port is None:
        port = cfg.port if cfg is not None and cfg.port is not None else 9400

    api_key = api_key if api_key is not None else os.getenv("PROXY_API_KEY")
    if not api_key:
        logger.warning(
            "No api_key configured (set --api-key or PROXY_API_KEY); "
            "all completion requests will be rejected with 401"
        )

    # Transport wiring: explicit arguments override the config.
    transport_mode = resolve_transport_mode(transport_mode or (cfg.transport_mode if cfg is not None else None))
    router_url = router_url or (cfg.router_url if cfg is not None else None)
    router_api_key = router_api_key or (cfg.router_api_key if cfg is not None else None)
    if context_length is None and cfg is not None:
        context_length = cfg.context_length
    if return_routed_experts is None:
        return_routed_experts = cfg.return_routed_experts if cfg is not None else False
    inference_engine_base_url = inference_engine_base_url or (cfg.inference_engine_base_url if cfg is not None else None)
    inference_engine_api_key = inference_engine_api_key or (cfg.inference_engine_api_key if cfg is not None else None)
    verl_load_balancer = verl_load_balancer or (cfg.verl_load_balancer if cfg is not None else None)
    ray_address = ray_address or (cfg.ray_address if cfg is not None else "auto")
    ray_namespace = ray_namespace if ray_namespace is not None else (cfg.ray_namespace if cfg is not None else None)

    # Validate before constructing clients or starting the HTTP server.
    if transport_mode == "slime" and not (router_url and router_url.strip()):
        raise ValueError("transport_mode='slime' requires router_url (--router-url)")
    if transport_mode == "direct" and not inference_engine_base_url:
        raise ValueError("transport_mode='direct' requires inference_engine_base_url (--inference-engine-base-url)")
    if transport_mode == "verl" and load_balancer is None and not (verl_load_balancer and verl_load_balancer.strip()):
        raise ValueError("transport_mode='verl' requires verl_load_balancer (--verl-load-balancer) or load_balancer")
    if transport_mode == "verl" and context_length is not None:
        raise ValueError("Configure context length on the VeRL rollout actors, not on the proxy")
    if transport_mode == "verl" and return_routed_experts:
        raise ValueError("return_routed_experts is supported only by the slime transport")

    from .rollout_provider import DirectRolloutProvider, SlimeRolloutProvider
    from .server import make_completion_handler

    sampling_overrides = cfg.sampling_overrides if cfg is not None else None
    get_routed_experts_config = None
    set_routed_experts_config = None
    if transport_mode == "slime":
        provider = SlimeRolloutProvider(
            router_url=router_url,
            tool_parser_factory=tool_parser_factory,
            inference_engine=cfg.inference_engine if cfg is not None else None,
            api_key=router_api_key,
            sampling_overrides=sampling_overrides,
            context_length=context_length,
            return_routed_experts=bool(return_routed_experts),
        )
        get_routed_experts_config = provider.get_routed_experts_config
        set_routed_experts_config = provider.set_runtime_routed_experts
        transport_target = f"Slime router at {router_url}"
    elif transport_mode == "verl":
        from .rollout_provider import RayRolloutProvider

        provider = RayRolloutProvider(
            load_balancer=load_balancer,
            verl_load_balancer=verl_load_balancer,
            ray_address=ray_address,
            ray_namespace=ray_namespace,
            tool_parser_factory=tool_parser_factory,
            inference_engine=cfg.inference_engine if cfg is not None else None,
            sampling_overrides=sampling_overrides,
        )
        transport_target = f"VeRL Ray load balancer {verl_load_balancer or '(injected handle)'}"
    else:
        provider = DirectRolloutProvider(
            base_urls=inference_engine_base_url,
            tool_parser_factory=tool_parser_factory,
            inference_engine=cfg.inference_engine if cfg is not None else None,
            api_keys=inference_engine_api_key,
            sampling_overrides=sampling_overrides,
        )
        transport_target = f"{provider.engine.name} engine(s) at {', '.join(provider.base_urls)}"

    proxy = LLMProxyServer(
        host=host,
        port=port,
        api_key=api_key,
        save_rollout_sessions=cfg.save_rollout_sessions if cfg is not None else True,
        save_rollout_logprobs=cfg.save_rollout_logprobs if cfg is not None else True,
        session_dir=cfg.rollout_session_dir if cfg is not None else None,
        error_log_dir=cfg.log_dir if cfg is not None else None,
        key_delimiter=cfg.proxy_api_delimiter if cfg is not None else None,
        completion_handler=make_completion_handler(provider),
        on_session_deleted=provider.release_session,
        get_sampling_overrides=provider.get_sampling_overrides,
        set_sampling_overrides=provider.set_runtime_sampling_overrides,
        get_routed_experts_config=get_routed_experts_config,
        set_routed_experts_config=set_routed_experts_config,
        get_tokenizer_fingerprint=provider.tokenizer_fingerprint,
        save_rollout_routed_experts=cfg.save_rollout_routed_experts if cfg is not None else False,
    )

    async def _run():
        try:
            if transport_mode == "verl":
                provider.start()
            url = await proxy.start()
            logger.info(
                "Standalone proxy started at %s (%s)\n"
                "  Health check:       %s/health\n"
                "  Session records:    %s",
                url, transport_target, url,
                proxy.session_store.run_dir if proxy.session_store is not None else "disabled",
            )
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            try:
                await proxy.stop()
            finally:
                await provider.aclose()

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
            "Path to a config YAML (default: proxyserver/configs/"
            "{$INFERENCE_ENGINE or vllm}-{$TRANSPORT_MODE or direct}.yaml)"
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
        "--transport-mode", default=None, choices=["slime", "direct", "verl"],
        help=(
            "How token-ID prompts reach the rollout engines; also selects the "
            "default config file (default: $TRANSPORT_MODE, else direct)"
        ),
    )
    parser.add_argument(
        "--router-url", default=None,
        help="slime: base URL of slime's sgl-router; required (default: the config's router_url)",
    )
    parser.add_argument(
        "--router-api-key", default=None,
        help="slime: optional bearer token for the router (default: the config's router_api_key)",
    )
    parser.add_argument(
        "--context-length", type=int, default=None,
        help=(
            "slime: the engines' context window in tokens — what they were "
            "launched with (SGLang's --context-length, i.e. slime's "
            "--sglang-context-length).  Pins the max_tokens clamp instead of "
            "discovering it, which slime's sgl-router cannot answer "
            "(default: the config's context_length)"
        ),
    )
    parser.add_argument(
        "--return-routed-experts", default=None, action=argparse.BooleanOptionalAction,
        help=(
            "slime: ask the engines for per-token MoE expert selections on "
            "every /generate and record the latest capture per agent stream "
            "(R3, slime's --use-rollout-routing-replay).  Requires engines "
            "launched with enable_return_routed_experts "
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
    parser.add_argument("--verl-load-balancer", default=None,
                        help="verl: name of the trainer's Ray load-balancer actor")
    parser.add_argument("--ray-address", default=None,
                        help="verl: existing Ray cluster address (default: auto)")
    parser.add_argument("--ray-namespace", default=None,
                        help="verl: namespace containing the named load-balancer actor")
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
        verl_load_balancer=args.verl_load_balancer,
        ray_address=args.ray_address,
        ray_namespace=args.ray_namespace,
    )


if __name__ == "__main__":
    main()
