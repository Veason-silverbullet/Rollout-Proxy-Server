"""Direct inference transport and public rollout-provider imports.

DirectRolloutProvider owns endpoint placement, discovery, and per-stream
bindings for vLLM /v1/completions and SGLang /generate. The native generation
providers live in rollout_sglang and rollout_vllm; the shared strict-TITO
flow lives in rollout_base. Public provider imports remain available here.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Callable
import httpx
from .engines import DEFAULT_MAX_TOKENS, SGLANG, EngineError
from .model_registry import SAMPLING_PARAM_KEYS, UnknownModelError, resolve_profile
from .rollout_base import BaseRolloutProvider, _clamp_to_context_window, _finite_logprob, engine_http_limits
from .rollout_sglang import SGLangRolloutProvider, _SGLangContextWindow, _sglang_generate
from .rollout_vllm import VLLMRolloutProvider

__all__ = [
    "BaseRolloutProvider", "DirectRolloutProvider", "SGLangRolloutProvider",
    "VLLMRolloutProvider", "engine_http_limits", "normalize_engine_endpoints",
]

logger = logging.getLogger(__name__)


def normalize_engine_endpoints(
    base_urls: list[str] | str,
    api_keys: list[str] | str | None,
) -> tuple[list[str], list[str | None]]:
    """Normalize and validate the direct transport's endpoint wiring.

    Returns ``(roots, keys)``: server roots (whitespace and a trailing
    ``/v1`` stripped) and exactly one api_key per root — a single key is
    broadcast to every URL, ``None`` means unauthenticated.
    """
    if isinstance(base_urls, str):
        base_urls = [base_urls]
    roots = [str(url).strip().rstrip("/").removesuffix("/v1") for url in base_urls]
    if not roots:
        raise ValueError("direct transport needs at least one inference_engine_base_url")
    blank = [i for i, root in enumerate(roots) if not root]
    if blank:
        raise ValueError(
            f"direct transport got blank inference_engine_base_url "
            f"entr{'y' if len(blank) == 1 else 'ies'} at position(s) {blank}"
        )
    if api_keys is None or isinstance(api_keys, str):
        keys: list[str | None] = [api_keys] * len(roots)
    elif len(api_keys) == 1 and len(roots) > 1:
        keys = list(api_keys) * len(roots)
    elif len(api_keys) != len(roots):
        raise ValueError(
            f"direct transport got {len(api_keys)} api_keys for {len(roots)} "
            f"base URLs; provide one per URL or a single shared key"
        )
    else:
        keys = list(api_keys)
    return roots, keys


class DirectRolloutProvider(BaseRolloutProvider):
    """direct transport: the vLLM / SGLang engines over plain HTTP, strictly TITO.

    Token-ID prompts go to vLLM's ``/v1/completions`` (with
    ``return_tokens_as_token_ids``) or SGLang's native ``/generate``. Each
    stream is bound round-robin to one endpoint for prefix-cache reuse.
    Each endpoint's served model and context window are discovered lazily
    (vLLM ``/v1/models``; SGLang ``/get_model_info`` and ``/get_server_info``),
    and the served model must map to the request's tokenizer profile.
    """

    def __init__(
        self,
        base_urls: list[str] | str,
        inference_engine: str | None = None,
        api_keys: list[str] | str | None = None,
        request_timeout: float = 900.0,
        tokenizer_loader: Callable[[str], Any] | None = None,
        sampling_overrides: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            base_urls: OpenAI base URLs of the engines, one per running
                instance, e.g. ``["http://10.0.1.5:8000/v1"]`` (a trailing
                ``/v1`` is optional — routes are resolved from the root).
            inference_engine: ``"vllm"`` or ``"sglang"``. Defaults to
                ``$INFERENCE_ENGINE``, then ``"vllm"``.
            api_keys: Bearer tokens matching ``base_urls`` — one per URL, or
                a single key broadcast to all, or ``None`` for
                unauthenticated engines.
            request_timeout: Per-request HTTP timeout (seconds).
            tokenizer_loader: Optional tokenizer-loading override (see
                :class:`BaseRolloutProvider`).
        """
        roots, api_keys = normalize_engine_endpoints(base_urls, api_keys)

        super().__init__(inference_engine, tokenizer_loader, sampling_overrides=sampling_overrides)

        self.base_urls = roots
        self._clients = [
            httpx.AsyncClient(
                base_url=root,
                headers={"Authorization": f"Bearer {key}"} if key else None,
                timeout=request_timeout,
                limits=engine_http_limits(),
            )
            for root, key in zip(roots, api_keys)
        ]
        self._session_endpoints: dict[str, int] = {}
        self._rr_index = 0
        self._model_ids: list[str | None] = [None] * len(roots)
        self._model_lens: list[int | None] = [None] * len(roots)
        self._model_lock = asyncio.Lock()
        self._context_windows = [_SGLangContextWindow(self._model_lock) for _ in roots]
        logger.info(
            "Direct rollout provider routing to %d %s endpoint(s): %s",
            len(roots), self.engine, ", ".join(roots),
        )

    def release_session(self, session_id: str) -> None:
        """Free the session's model pin, token stream, and sticky endpoint binding."""
        self._session_endpoints.pop(session_id, None)
        super().release_session(session_id)

    def _acquire_endpoint(self, session_id: str | None) -> int:
        """Resolve the endpoint index for a session — sticky, round-robin on first use."""
        if session_id is not None and session_id in self._session_endpoints:
            return self._session_endpoints[session_id]
        index = self._rr_index % len(self._clients)
        self._rr_index = index + 1
        if session_id is not None:
            self._session_endpoints[session_id] = index
        return index

    async def _vllm_model_id(self, index: int) -> str:
        """The served model id of endpoint ``index``, from ``GET /v1/models``
        (fetched once and cached; the engine serves exactly one model).
        The same response carries ``max_model_len``, cached alongside for
        the context-window clamp in :meth:`_call_engine`."""
        if self._model_ids[index] is None:
            async with self._model_lock:
                if self._model_ids[index] is None:
                    resp = await self._clients[index].get("/v1/models")
                    resp.raise_for_status()
                    models = resp.json().get("data") or []
                    if not models:
                        raise EngineError(f"{self.base_urls[index]}: /v1/models reported no served model")
                    max_len = models[0].get("max_model_len")
                    self._model_lens[index] = int(max_len) if max_len else None
                    self._model_ids[index] = models[0]["id"]
        return self._model_ids[index]

    async def _sglang_model_id(self, index: int) -> str:
        """Discover and cache a direct SGLang endpoint's model identity.

        The served alias follows the same mapping contract as vLLM's model
        ID. Older SGLang versions expose only model_path, which the shared
        validator accepts as a mapped full path or checkpoint basename.
        Missing or malformed identity must fail before generation, even
        when context-window discovery could proceed without metadata.
        """
        if self._model_ids[index] is None:
            async with self._model_lock:
                if self._model_ids[index] is None:
                    resp = await self._clients[index].get("/get_model_info")
                    resp.raise_for_status()
                    info = resp.json()
                    model = (info.get("served_model_name") or info.get("model_path")) if isinstance(info, dict) else None
                    if not isinstance(model, str) or not model.strip():
                        raise EngineError(
                            f"{self.base_urls[index]}: /get_model_info reported no valid "
                            "served_model_name or model_path; refusing unverified token IDs"
                        )
                    self._model_ids[index] = model.strip()
        return self._model_ids[index]

    async def _sglang_context_length(self, index: int) -> int | None:
        """Endpoint ``index``'s discovered window; None disables clamping."""
        return await self._context_windows[index].get(self._clients[index], self.base_urls[index])

    def _validate_served_model(self, index: int, served_model: str, session_id: str | None) -> None:
        """Require an explicit profile mapping before sending token IDs.

        Model aliases and checkpoint paths are supported through mapping.json;
        a registered full name takes precedence over its basename.
        """
        claimed = self._session_models.get(session_id) if session_id is not None else None
        if claimed is None:
            raise EngineError(f"Session {session_id}: cannot verify the backend without a pinned model")
        for name in (served_model, served_model.rstrip("/").rsplit("/", 1)[-1]):
            try:
                served_profile = resolve_profile(name)
                break
            except UnknownModelError:
                continue
        else:
            raise EngineError(
                f"{self.base_urls[index]} serves unrecognized model {served_model!r}; "
                "add its name to tokenization/mapping.json with the verified tokenizer profile before sending token IDs"
            )
        claimed_profile = resolve_profile(claimed)
        if served_profile.name != claimed_profile.name:
            raise EngineError(
                f"Session {session_id} requests model {claimed!r} (profile {claimed_profile.name}) "
                f"but endpoint {self.base_urls[index]} serves {served_model!r} "
                f"(profile {served_profile.name}); refusing incompatible token IDs"
            )

    async def _call_engine(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        session_id: str | None,
        routed_experts_prior: tuple[int, int] | None = None,
    ) -> tuple[list[int], list[float], str | None, dict[str, Any]]:
        index = self._acquire_endpoint(session_id)
        http = self._clients[index]
        if self.engine == SGLANG:
            served_model = await self._sglang_model_id(index)
            self._validate_served_model(index, served_model, session_id)
            _clamp_to_context_window(
                sampling_params, len(prompt_ids),
                await self._sglang_context_length(index), self.base_urls[index], session_id,
            )
            return await _sglang_generate(http, prompt_ids, sampling_params, session_id)

        served_model = await self._vllm_model_id(index)
        self._validate_served_model(index, served_model, session_id)

        _clamp_to_context_window(
            sampling_params, len(prompt_ids),
            self._model_lens[index], self.base_urls[index], session_id,
        )
        max_tokens = sampling_params.get("max_tokens", DEFAULT_MAX_TOKENS)

        payload: dict[str, Any] = {
            "model": served_model,
            "prompt": prompt_ids,  # token IDs in
            "max_tokens": max_tokens,
            # vLLM's own dialect: logprobs is an int of extra top-logprobs
            # per position; 0 returns just the sampled token's logprob.
            "logprobs": 0 if sampling_params.get("logprobs") else None,
            "return_tokens_as_token_ids": True,
        }
        # top_k/min_p/repetition_penalty are vLLM extensions /v1/completions accepts.
        for key in (*SAMPLING_PARAM_KEYS, "stop", "seed", "stop_token_ids"):
            if sampling_params.get(key) is not None:
                payload[key] = sampling_params[key]

        resp = await http.post("/v1/completions", json=payload)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]

        logprobs_blob = choice.get("logprobs") or {}
        tokens = logprobs_blob.get("tokens") or []
        token_logprobs = logprobs_blob.get("token_logprobs") or []
        if (
            not isinstance(tokens, list) or not isinstance(token_logprobs, list)
            or len(tokens) != len(token_logprobs)
        ):
            raise EngineError(f"Session {session_id}: vLLM returned misaligned tokens and token_logprobs")
        token_ids = []
        log_probs = []
        # Validate each raw pair so filtering cannot shift positions.
        for index, (token, lp) in enumerate(zip(tokens, token_logprobs)):
            token_id = token.removeprefix("token_id:") if isinstance(token, str) else ""
            if (
                not isinstance(token, str) or not token.startswith("token_id:")
                or not token_id.isascii() or not token_id.isdecimal()
                or isinstance(lp, bool) or not isinstance(lp, (int, float))
            ):
                raise EngineError(f"Session {session_id}: vLLM returned an invalid token/logprob pair at position {index}")
            token_ids.append(int(token_id))
            log_probs.append(_finite_logprob(lp, session_id))
        # The OpenAI-compatible response reports no weight version.
        return token_ids, log_probs, choice.get("finish_reason"), {}

    async def aclose(self) -> None:
        """Close the underlying HTTP clients."""
        for client in self._clients:
            await client.aclose()
