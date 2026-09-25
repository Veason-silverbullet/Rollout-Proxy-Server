"""Standalone LLM Proxy Server CLI.

Starts the proxy as an independent HTTP (+ WebSocket) service.  The mode
depends on the config's ``transport_mode``:

**relay mode** (``transport_mode: verl``, or ``slime`` / ``direct`` without
their endpoint fields)
    No framework-specific dependencies (no Ray).  Framework-side inference workers
    (:class:`~proxyserver.worker_client.InferenceWorkerClient`) connect to
    ``/ws/worker`` and service inference requests pushed by the proxy.

**local mode** (``transport_mode: slime`` with ``router_url``, or
``transport_mode: direct`` with ``inference_engine_base_url``)
    The proxy generates itself — through slime's sgl-router, or straight
    against the vLLM/SGLang engines — no relay workers.  Tokenizers and
    tool-call parsers for strict-TITO prompts are resolved from the
    ``model`` agents send, via ``tokenization/mapping.json``, so local mode
    needs ``transformers`` + ``httpx`` installed.

Configuration comes from ``proxyserver/configs/{engine}-{transport}.yaml``
— the engine is picked by ``$INFERENCE_ENGINE`` (default ``vllm``) and the
transport by ``--transport-mode`` / ``$TRANSPORT_MODE`` (default ``verl``),
or pass an explicit file with ``--config``; see :mod:`proxyserver.config`
for the fields.  Explicit CLI flags override the config.

Usage::

    # CLI — everything from configs/{engine}-{transport}.yaml
    # ({$INFERENCE_ENGINE or vllm}-{$TRANSPORT_MODE or verl}.yaml)
    python -m proxyserver.cli

    # CLI — explicit config file
    python -m proxyserver.cli --config proxyserver/configs/sglang-verl.yaml

    # CLI — slime, serving agents directly through sgl-router
    # (picks configs/sglang-slime.yaml)
    INFERENCE_ENGINE=sglang python -m proxyserver.cli \
        --transport-mode slime \
        --router-url http://<sglang-router-ip>:<port>

    # CLI — direct, serving agents straight against the engines
    # (picks configs/vllm-direct.yaml)
    python -m proxyserver.cli \
        --transport-mode direct \
        --inference-engine-base-url http://<engine-ip>:<port>/v1 \
        --inference-engine-api-key <key>
"""

from __future__ import annotations
import asyncio
import logging
import os
from pathlib import Path
from .config import ProxyConfig, load_config
from .server import LLMProxyServer

logger = logging.getLogger(__name__)


def run_standalone_proxy(
    host: str | None = None,
    port: int | None = None,
    relay_request_timeout: float = 600,
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
) -> None:
    """Run the proxy as a standalone service.

    This starts the HTTP server and blocks until interrupted.  With
    ``transport_mode: verl`` (default) the proxy runs in relay mode:
    framework-side inference workers must connect to
    ``ws://{host}:{port}/ws/worker`` to service requests.  With
    ``transport_mode: slime`` and a ``router_url``, or ``transport_mode:
    direct`` and ``inference_engine_base_url``, the proxy runs in local
    mode and generates itself — through slime's sgl-router, or straight
    against the vLLM/SGLang engines.

    Most arguments override the identically-named :class:`ProxyConfig`
    fields (see :mod:`proxyserver.config`); ``None`` defers to the config,
    then the built-in default (host ``0.0.0.0``, port ``9400``).

    Args:
        relay_request_timeout: Timeout (seconds) for relay-mode requests.
        log_level: Logging level (e.g. ``"INFO"``, ``"DEBUG"``).
        api_key: Shared secret for keyed routing (agents send
            ``{api_key}{proxy_api_delimiter}{session_id}``).  Defaults to
            the ``PROXY_API_KEY`` environment variable.
        config: A :class:`ProxyConfig`, a path to a config YAML, or ``None``
            to load the default ``configs/{engine}-{transport}.yaml``.
        transport_mode: ``"verl"`` | ``"slime"`` | ``"direct"``; when
            ``config`` is ``None`` it also selects the default config file.
        tool_parser_factory: slime/direct local mode — optional callable
            ``(tool_call_parser, tokenizer) -> tool_parser`` overriding the
            built-in parsers (e.g. verl's ``ToolParser.get_tool_parser``);
            it receives each model's ``tool_call_parser`` name from
            ``tokenization/mapping.json``.
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
    transport_mode = transport_mode or (cfg.transport_mode if cfg is not None else "verl")
    router_url = router_url or (cfg.router_url if cfg is not None else None)
    router_api_key = router_api_key or (cfg.router_api_key if cfg is not None else None)
    if context_length is None and cfg is not None:
        context_length = cfg.context_length
    if return_routed_experts is None:
        return_routed_experts = cfg.return_routed_experts if cfg is not None else False
    inference_engine_base_url = inference_engine_base_url or (cfg.inference_engine_base_url if cfg is not None else None)
    inference_engine_api_key = inference_engine_api_key or (cfg.inference_engine_api_key if cfg is not None else None)

    completion_handler = None
    on_session_deleted = None
    provider = None
    mode_label = None
    transport_target = None  # what local mode generates through, for the log
    # Routed-experts capture endpoints: only the slime transport's provider owns
    # the layers, so the callables stay None (endpoints answer 501) everywhere else.
    get_routed_experts_config = None
    set_routed_experts_config = None
    if (transport_mode == "slime" and router_url) or (transport_mode == "direct" and inference_engine_base_url):
        # Local mode: the proxy generates itself — through slime's
        # sgl-router, or straight against the vLLM/SGLang engines.
        # Tokenizers and tool parsers load lazily inside the provider, per
        # the model names requests carry.
        from .rollout_provider import DirectRolloutProvider, SlimeRolloutProvider
        from .server import make_local_completion_handler

        sampling_overrides = cfg.sampling_overrides if cfg is not None else None
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
            transport_target = f"slime sgl-router at {router_url}"
        else:
            provider = DirectRolloutProvider(
                base_urls=inference_engine_base_url,
                tool_parser_factory=tool_parser_factory,
                inference_engine=cfg.inference_engine if cfg is not None else None,
                api_keys=inference_engine_api_key,
                sampling_overrides=sampling_overrides,
            )
            transport_target = f"{provider.engine.name} engine(s) at {', '.join(provider.base_urls)}"
        completion_handler = make_local_completion_handler(provider)
        on_session_deleted = provider.release_session
        mode_label = transport_mode
    elif transport_mode == "slime":
        logger.info(
            "transport_mode 'slime' without router_url: running in relay mode — "
            "start an InferenceWorkerClient(transport_mode='slime', router_url=...) "
            "near the router"
        )
    elif transport_mode == "direct":
        logger.info(
            "transport_mode 'direct' without inference_engine_base_url: running "
            "in relay mode — start an InferenceWorkerClient(transport_mode='direct', "
            "inference_engine_base_url=...) near the engines"
        )

    proxy = LLMProxyServer(
        host=host,
        port=port,
        relay_request_timeout=relay_request_timeout,
        api_key=api_key,
        save_rollout_sessions=cfg.save_rollout_sessions if cfg is not None else True,
        save_rollout_logprobs=cfg.save_rollout_logprobs if cfg is not None else True,
        session_dir=cfg.rollout_session_dir if cfg is not None else None,
        error_log_dir=cfg.log_dir if cfg is not None else None,
        key_delimiter=cfg.proxy_api_delimiter if cfg is not None else None,
        completion_handler=completion_handler,
        on_session_deleted=on_session_deleted,
        get_sampling_overrides=provider.get_sampling_overrides if provider is not None else None,
        set_sampling_overrides=provider.set_runtime_sampling_overrides if provider is not None else None,
        get_routed_experts_config=get_routed_experts_config,
        set_routed_experts_config=set_routed_experts_config,
        get_tokenizer_fingerprint=provider.tokenizer_fingerprint if provider is not None else None,
        save_rollout_routed_experts=cfg.save_rollout_routed_experts if cfg is not None else False,
        mode_label=mode_label,
    )

    async def _run():
        url = await proxy.start()
        if completion_handler is not None:
            logger.info(
                "Standalone proxy started at %s (local mode, %s)\n"
                "  Health check:       %s/health\n"
                "  Session records:    %s",
                url, transport_target, url,
                proxy.session_store.run_dir if proxy.session_store is not None else "disabled",
            )
        else:
            logger.info(
                "Standalone proxy started at %s (relay mode, transport_mode=%s)\n"
                "  Workers connect to: ws://%s:%d/ws/worker\n"
                "  Health check:       %s/health\n"
                "  Session records:    %s",
                url, transport_mode, host, port, url,
                proxy.session_store.run_dir if proxy.session_store is not None else "disabled",
            )
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await proxy.stop()
            if provider is not None:
                await provider.aclose()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


def main() -> None:
    """CLI entry point for running the proxy as a standalone service."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the LLM Proxy as a standalone relay service.")
    parser.add_argument(
        "--config", default=None,
        help=(
            "Path to a config YAML (default: proxyserver/configs/"
            "{$INFERENCE_ENGINE or vllm}-{$TRANSPORT_MODE or verl}.yaml)"
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
        "--relay-timeout", type=float, default=600.0,
        help=(
            "Timeout (seconds) for relay requests, aligned with the OpenAI "
            "SDK's 600s client default — raise both together when agents "
            "allow longer turns.  A completion that outlives it is still "
            "recorded, and an exact retry is re-served the same turn "
            "(default: 600)"
        ),
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
        "--transport-mode", default=None, choices=["verl", "slime", "direct"],
        help=(
            "How token-ID prompts reach the rollout engines; also selects the "
            "default config file (default: $TRANSPORT_MODE, else verl)"
        ),
    )
    parser.add_argument(
        "--router-url", default=None,
        help="slime: base URL of slime's sgl-router; enables local mode (default: the config's router_url)",
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
            "instance; enables local mode (default: the config's "
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
        relay_request_timeout=args.relay_timeout,
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
