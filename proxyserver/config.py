"""Proxy configuration (``proxyserver/configs/{engine}-{transport}.yaml``).

One file per engine + transport-mode combination (slime is SGLang-only, so
there is no ``vllm-slime.yaml``), selected by ``inference_engine`` (falling
back to ``$INFERENCE_ENGINE``, then ``"vllm"``) and ``transport_mode``
(falling back to ``$TRANSPORT_MODE``, then ``"verl"``).

Only ``direct`` names engine endpoints here (``inference_engine_base_url``
/ ``inference_engine_api_key``): under ``verl`` the relay workers reach the
engines through the framework's load balancer, and under ``slime`` the proxy
needs only ``router_url`` (plus ``context_length``, which slime's router
cannot report on its own).  The model, its tokenizer, and its tool-call
parser are never configured — agents name a model per request and
``tokenization/mapping.json`` decides what that claim means
(:mod:`proxyserver.model_registry`).
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from .engines import get_adapter
from .model_registry import validate_sampling_overrides
from .server import PROXY_KEY_DELIMITER

_PACKAGE_DIR = Path(__file__).resolve().parent

CONFIG_DIR = _PACKAGE_DIR / "configs"

#: Supported transport modes (the ``transport_mode`` config field).
TRANSPORT_MODES = ("verl", "slime", "direct")

#: Environment variable consulted when ``transport_mode`` is not passed.
TRANSPORT_MODE_ENV_VAR = "TRANSPORT_MODE"


def resolve_transport_mode(transport_mode: str | None = None) -> str:
    """Resolve a transport-mode name: explicit > ``$TRANSPORT_MODE`` > ``"verl"``."""
    name = (transport_mode or os.getenv(TRANSPORT_MODE_ENV_VAR) or "verl").strip().lower()
    if name not in TRANSPORT_MODES:
        raise ValueError(f"Unknown transport_mode {name!r}; supported modes: {', '.join(TRANSPORT_MODES)}")
    return name


def default_config_path(inference_engine: str | None = None, transport_mode: str | None = None) -> Path:
    """``configs/{engine}-{transport}.yaml`` for ``inference_engine`` (or
    ``$INFERENCE_ENGINE``, or ``"vllm"``) and ``transport_mode`` (or
    ``$TRANSPORT_MODE``, or ``"verl"``).  Both names are validated exactly as the proxy does."""
    return CONFIG_DIR / f"{get_adapter(inference_engine).name}-{resolve_transport_mode(transport_mode)}.yaml"


def _resolve_dir(value: Any, default: Path) -> Path:
    """Resolve a configured directory; relative paths are package-relative."""
    if value is None:
        return default
    path = Path(str(value))
    return path if path.is_absolute() else _PACKAGE_DIR / path


def _as_str_list(value: Any) -> list[str] | None:
    """Normalize a YAML scalar-or-list field to ``list[str]`` (``None`` stays)."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return [str(value)]
    return [str(v) for v in value]


@dataclass(frozen=True)
class ProxyConfig:
    """One parsed ``configs/{engine}-{transport}.yaml``."""

    inference_engine: str
    proxy_api_delimiter: str
    host: str | None                   # proxy bind address; None → CLI/built-in default
    port: int | None                   # proxy port (0 = auto-select); None → CLI/built-in default
    rollout_session_dir: Path
    log_dir: Path
    save_rollout_sessions: bool
    #: Include ``completion_logprobs`` in the persisted session JSON.
    save_rollout_logprobs: bool = True
    transport_mode: str = "verl"       # "verl" | "slime" | "direct"
    router_url: str | None = None      # slime: base URL of slime's sgl-router
    router_api_key: str | None = None  # slime: optional bearer token for the router
    #: slime: the engines' context window in tokens, stated instead of
    #: discovered.  slime's sgl-router answers ``GET /get_server_info`` from its
    #: own RouterManager stub rather than forwarding a worker's payload, so the
    #: max_tokens clamp has no window to find unless one is configured here.
    context_length: int | None = None
    #: slime: ask the engines for per-token MoE expert selections on every
    #: ``/generate`` and record the latest capture per agent stream (R3,
    #: slime's ``--use-rollout-routing-replay``).  Requires engines launched
    #: with ``enable_return_routed_experts``.
    return_routed_experts: bool = False
    #: Include the ``routed_experts`` blobs in the persisted session JSON
    #: (they are large and every turn rewrites the file; ``GET /sessions/{id}``
    #: carries them regardless).
    save_rollout_routed_experts: bool = False
    #: direct: OpenAI base URLs of the vLLM/SGLang engines, one per instance.
    inference_engine_base_url: list[str] | None = None
    #: direct: api_keys matching ``inference_engine_base_url`` (a single key is broadcast to every base URL).
    inference_engine_api_key: list[str] | None = None
    #: Training-side sampling policy that *wins over* request-provided
    #: params (restricted to :data:`~proxyserver.model_registry.SAMPLING_PARAM_KEYS`).
    #: The restart-proof baseline under any runtime overrides the trainer
    #: pushes via ``PUT /sampling_overrides``.
    sampling_overrides: dict[str, Any] | None = None
    path: Path | None = None           # file this config was loaded from


def load_config(
    path: str | Path | None = None,
    inference_engine: str | None = None,
    transport_mode: str | None = None,
) -> ProxyConfig:
    """Load a :class:`ProxyConfig`.

    Args:
        path: Explicit config file.  When ``None``, the default
            ``configs/{engine}-{transport}.yaml`` is used, with the engine
            taken from ``inference_engine`` / ``$INFERENCE_ENGINE`` /
            ``"vllm"`` and the transport from ``transport_mode`` /
            ``$TRANSPORT_MODE`` / ``"verl"``.
        inference_engine: Engine used to pick the default file; the file's
            own ``inference_engine`` field wins once loaded.
        transport_mode: Transport used to pick the default file; the file's
            own ``transport_mode`` field wins once loaded.
    """
    config_path = Path(path) if path is not None else default_config_path(inference_engine, transport_mode)
    if not config_path.is_file():
        raise FileNotFoundError(f"Proxy config not found: {config_path}")
    data: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}

    engine = get_adapter(data.get("inference_engine") or inference_engine).name
    transport_mode = str(data.get("transport_mode") or "").strip().lower() or resolve_transport_mode(transport_mode)
    if transport_mode not in TRANSPORT_MODES:
        raise ValueError(
            f"{config_path}: unknown transport_mode {transport_mode!r}; "
            f"supported modes: {', '.join(TRANSPORT_MODES)}"
        )
    if transport_mode == "slime" and engine != "sglang":
        raise ValueError(
            f"{config_path}: transport_mode 'slime' requires inference_engine 'sglang' "
            f"(slime's sgl-router fronts SGLang servers), got {engine!r}"
        )
    host = data.get("host")
    port = data.get("port")
    router_url = data.get("router_url")
    router_api_key = data.get("router_api_key")
    context_length = data.get("context_length")
    if context_length is not None:
        if transport_mode != "slime":
            raise ValueError(
                f"{config_path}: context_length is a slime-transport field, but transport_mode "
                f"is {transport_mode!r}; under verl and direct the window comes from the engines"
            )
        try:
            context_length = int(context_length)
        except (TypeError, ValueError):
            raise ValueError(
                f"{config_path}: context_length must be a number of tokens, got {context_length!r}"
            ) from None
        if context_length <= 0:
            raise ValueError(f"{config_path}: context_length must be positive, got {context_length}")
    return_routed_experts = bool(data.get("return_routed_experts", False))
    if return_routed_experts and transport_mode != "slime":
        raise ValueError(
            f"{config_path}: return_routed_experts is a slime-transport field, but transport_mode "
            f"is {transport_mode!r}; only the slime transport speaks SGLang's /generate directly"
        )
    engine_base_urls = _as_str_list(data.get("inference_engine_base_url"))
    engine_api_keys = _as_str_list(data.get("inference_engine_api_key"))
    if engine_api_keys and not engine_base_urls:
        raise ValueError(f"{config_path}: inference_engine_api_key is set but inference_engine_base_url is not")
    if engine_base_urls and engine_api_keys and len(engine_api_keys) not in (1, len(engine_base_urls)):
        raise ValueError(
            f"{config_path}: {len(engine_api_keys)} inference_engine_api_key entries "
            f"for {len(engine_base_urls)} inference_engine_base_url entries; "
            f"provide one per base URL or a single shared key"
        )
    # Rejected rather than ignored: unknown keys are normally skipped, but a
    # config still carrying the removed field would otherwise change sampling
    # behavior silently -- the exact failure mode the override layers exist
    # to prevent.
    if "sampling_defaults" in data:
        raise ValueError(
            f"{config_path}: sampling_defaults has been removed. Pin the policy with "
            f"sampling_overrides (wins over agent-request params); request-omitted "
            f"params fall back to the model's bundled generation_config.json."
        )
    sampling_overrides = data.get("sampling_overrides")
    if sampling_overrides is not None:
        sampling_overrides = validate_sampling_overrides(sampling_overrides, f"{config_path}: sampling_overrides")
    return ProxyConfig(
        inference_engine=engine,
        proxy_api_delimiter=str(data.get("proxy_api_delimiter") or PROXY_KEY_DELIMITER),
        host=str(host) if host is not None else None,
        port=int(port) if port is not None else None,
        rollout_session_dir=_resolve_dir(data.get("rollout_session_dir"), _PACKAGE_DIR / "sessions"),
        log_dir=_resolve_dir(data.get("log_dir"), _PACKAGE_DIR / "logs"),
        save_rollout_sessions=bool(data.get("save_rollout_sessions", True)),
        save_rollout_logprobs=bool(data.get("save_rollout_logprobs", True)),
        save_rollout_routed_experts=bool(data.get("save_rollout_routed_experts", False)),
        transport_mode=transport_mode,
        router_url=str(router_url) if router_url else None,
        router_api_key=str(router_api_key) if router_api_key else None,
        context_length=context_length,
        return_routed_experts=return_routed_experts,
        inference_engine_base_url=engine_base_urls,
        inference_engine_api_key=engine_api_keys,
        sampling_overrides=sampling_overrides,
        path=config_path,
    )
