"""Resolve request model names through ``tokenization/mapping.json``.

ModelRegistry lazily caches a tokenizer, parsers, and TokenStreamManager per
profile. Model names mapped to the same profile share that bundle.
"""

from __future__ import annotations
import json
import logging
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Callable
from .protocol import finite
from .token_stream import TokenStreamManager
from .tokenization import fingerprint as _fingerprint
from .tokenization.base import TOKENIZATION_DIR, ModelProfile, get_profile
from .tokenization.tool_parser import BaseToolParser, NullToolParser, get_tool_parser

logger = logging.getLogger(__name__)

#: Maps each served model name to its profile module (basename under :data:`TOKENIZATION_DIR`).
MODEL_MAPPING_PATH = TOKENIZATION_DIR / "mapping.json"


class UnknownModelError(ValueError):
    """No profile is mapped to the requested model name.

    Maps to HTTP 404: completion responses use the ``error`` field;
    the tokenizer-fingerprint endpoint uses ``detail``.
    """


@cache
def load_model_mapping() -> dict[str, str]:
    """Parse ``tokenization/mapping.json`` into ``{model_name: profile_name}``.

    Cached for this process; restart the proxy to apply mapping changes.
    ModelRegistry validates the mapped profiles when it is constructed.
    """
    try:
        mapping = json.loads(MODEL_MAPPING_PATH.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"Model mapping not found: {MODEL_MAPPING_PATH}") from None
    if not isinstance(mapping, dict):
        raise ValueError(f"{MODEL_MAPPING_PATH}: expected an object mapping each model name to a profile module name")
    profiles: dict[str, str] = {}
    for model, profile_name in mapping.items():
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise ValueError(
                f"{MODEL_MAPPING_PATH}: entry for model {model!r} must name a "
                f"profile module under {TOKENIZATION_DIR}, got {profile_name!r}"
            )
        profiles[str(model)] = profile_name.strip()
    return profiles


def resolve_profile(model_name: Any) -> ModelProfile:
    """Resolve ``model_name`` to its :class:`ModelProfile` via ``mapping.json``."""
    name = str(model_name or "").strip()
    if not name:
        raise ValueError("model_name is empty; cannot resolve a profile")
    mapping = load_model_mapping()
    profile_name = mapping.get(name)
    if profile_name is None:
        raise UnknownModelError(
            f"No profile mapped to model {name!r} in {MODEL_MAPPING_PATH}; "
            f"known models: {', '.join(sorted(mapping)) or '(none)'}"
        )
    profile = get_profile(profile_name)  # ValueError if the module/PROFILE is bad
    if not Path(profile.tokenizer_path).is_dir():
        raise FileNotFoundError(
            f"{MODEL_MAPPING_PATH} maps model {name!r} to profile {profile_name!r} "
            f"(tokenizer {profile.tokenizer_path}), but that tokenizer directory "
            f"does not exist"
        )
    return profile


def build_tool_parser(
    tool_call_parser: str | None,
    tokenizer: Any,
    reasoning: Any = None,
) -> BaseToolParser:
    """Resolve the tool parser named ``tool_call_parser`` on ``tokenizer``.

    A falsy name yields a :class:`~proxyserver.tokenization.tool_parser.NullToolParser`,
    which still cleans agent-visible content.
    """
    if not tool_call_parser:
        return NullToolParser(tokenizer, reasoning=reasoning)
    return get_tool_parser(tool_call_parser, tokenizer, reasoning=reasoning)


#: Serializes the first ``transformers`` import: its lazy module fails
#: concurrent first imports from worker threads with ``cannot import name
#: 'AutoTokenizer'``. Once imported, different tokenizers load concurrently.
_TRANSFORMERS_IMPORT_LOCK = threading.Lock()


def _auto_tokenizer() -> Any:
    """``transformers.AutoTokenizer``, importing ``transformers`` under the lock."""
    with _TRANSFORMERS_IMPORT_LOCK:
        from transformers import AutoTokenizer
    return AutoTokenizer


def load_local_tokenizer(tokenizer_path: str) -> Any:
    """Default tokenizer loader, shared with the e2e harness's verifier."""
    return _auto_tokenizer().from_pretrained(tokenizer_path, trust_remote_code=True)


#: Distribution-shaping keys read from ``generation_config.json`` and allowed in
#: overrides; ``max_tokens`` and ``stop`` stay request-owned.
SAMPLING_PARAM_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
)


#: Rollouts always sample: zero and smaller positive temperatures use this floor.
#: Align this with vLLM which raises positive temperatures below 0.01 to 0.01.
MIN_TEMPERATURE = 0.01


def apply_temperature_floor(params: dict[str, Any]) -> dict[str, Any]:
    """Copy validated sampling params, flooring temperature when present."""
    result = dict(params)
    temperature = result.get("temperature")
    if temperature is not None and temperature < MIN_TEMPERATURE:
        result["temperature"] = MIN_TEMPERATURE
    return result


def validate_sampling_overrides(overrides: Any, where: str) -> dict[str, Any]:
    """Validate a ``sampling_overrides`` mapping: only :data:`SAMPLING_PARAM_KEYS`
    with finite values in range (``top_k=-1`` disables the cutoff). Returns
    the overrides in :data:`SAMPLING_PARAM_KEYS` order.
    """
    if not isinstance(overrides, dict):
        raise ValueError(
            f"{where}: sampling overrides must be a mapping of sampling "
            f"params (e.g. {{temperature: 1.0}}), got {type(overrides).__name__}"
        )
    bad = sorted(set(overrides) - set(SAMPLING_PARAM_KEYS), key=str)
    if bad:
        raise ValueError(
            f"{where}: sampling overrides may only carry the distribution keys "
            f"({', '.join(SAMPLING_PARAM_KEYS)}); got {', '.join(map(str, bad))}. "
            f"max_tokens and stop stay request-owned."
        )
    for key, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: sampling override {key!r} must be a number, got {value!r}")
        if not finite(value):
            raise ValueError(f"{where}: {key} must be finite")
        if key == "top_k":
            valid = type(value) is int and (value == -1 or value >= 1)
            expected = "an integer equal to -1 or greater than zero"
        elif key == "top_p":
            valid, expected = 0 < value <= 1, "in (0, 1]"
        elif key == "min_p":
            valid, expected = 0 <= value <= 1, "in [0, 1]"
        elif key == "temperature":
            valid, expected = value >= 0, "nonnegative"
        elif key == "repetition_penalty":
            valid, expected = value > 0, "greater than zero"
        else:
            valid, expected = -2 <= value <= 2, "in [-2, 2]"
        if not valid:
            raise ValueError(f"{where}: {key} must be {expected}")
    return {key: overrides[key] for key in SAMPLING_PARAM_KEYS if key in overrides}


def load_sampling_defaults(tokenizer_path: str) -> dict[str, Any]:
    """The publisher's sampling defaults from ``{tokenizer_path}/generation_config.json``,
    filling in request-omitted params; ``{}`` if missing or unreadable."""
    path = Path(tokenizer_path) / "generation_config.json"
    try:
        config = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning("Cannot read sampling defaults from %s: %s", path, e)
        return {}
    if not isinstance(config, dict):
        logger.warning("%s: expected a JSON object, got %s", path, type(config).__name__)
        return {}
    return {key: config[key] for key in SAMPLING_PARAM_KEYS if config.get(key) is not None}


@dataclass
class ResolvedModel:
    """Everything a rollout provider needs to serve one model family."""

    tokenizer: Any
    tool_parser: Any | None
    token_streams: TokenStreamManager
    sampling_defaults: dict[str, Any] = field(default_factory=dict)
    reasoning: Any = None
    stop_token_ids: tuple[int, ...] = ()
    reasoning_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None
    profile: str = ""


class ModelRegistry:
    """Lazily loads and caches :class:`ResolvedModel` bundles per profile.

    Thread-safe: concurrent first requests for one profile share a single
    tokenizer load, or its failure; different profiles load in parallel.
    """

    def __init__(self, tokenizer_loader: Callable[[str], Any] | None = None) -> None:
        """
        Args:
            tokenizer_loader: Optional ``(tokenizer_dir) -> tokenizer`` override
                (the offline tests inject fake tokenizers here).
        """
        # Validate every mapped profile eagerly; importing one loads no tokenizer.
        mapping = load_model_mapping()
        for model, profile_name in mapping.items():
            try:
                get_profile(profile_name)
            except ValueError as e:
                raise ValueError(f"{MODEL_MAPPING_PATH}: model {model!r}: {e}") from None
        if tokenizer_loader is None:
            # Import transformers before any worker thread loads a tokenizer.
            _auto_tokenizer()
        self._tokenizer_loader = tokenizer_loader or load_local_tokenizer
        self._bundles: dict[str, ResolvedModel] = {}  # by profile name
        self._loading: dict[str, Future] = {}  # in-flight loads by profile name
        self._fingerprints: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def resolve(self, model_name: str) -> ResolvedModel:
        """The (cached) bundle serving ``model_name``."""
        profile = resolve_profile(model_name)

        with self._lock:
            bundle = self._bundles.get(profile.name)
            if bundle is not None:
                return bundle
            loading = self._loading.get(profile.name)
            owner = loading is None
            if owner:
                loading = self._loading[profile.name] = Future()
        if not owner:
            return loading.result()

        # Build outside the lock; a failed load installs nothing, its waiters
        # receive the error, and the next request retries.
        try:
            bundle = self._build_bundle(profile, model_name)
        except BaseException as e:
            with self._lock:
                del self._loading[profile.name]
            loading.set_exception(e)
            raise
        with self._lock:
            self._bundles[profile.name] = bundle
            del self._loading[profile.name]
        loading.set_result(bundle)
        return bundle

    def _build_bundle(self, profile: ModelProfile, model_name: str) -> ResolvedModel:
        """Load the profile's tokenizer and build its bundle (not yet cached)."""
        tokenizer = self._tokenizer_loader(profile.tokenizer_path)
        parser_name = profile.tool_call_parser.NAME if profile.tool_call_parser is not None else None
        bundle = ResolvedModel(
            tokenizer=tokenizer,
            tool_parser=build_tool_parser(parser_name, tokenizer, reasoning=profile.reasoning),
            token_streams=TokenStreamManager(
                tokenizer,
                generation_prompt=profile.generation_prompt,
                assistant_turn_end=profile.assistant_turn_end,
                continuation_headers=profile.continuation_headers,
                assistant_turn_endings=profile.assistant_turn_endings,
                prepare_delta_messages=profile.prepare_delta_messages,
                default_reasoning_effort=profile.default_reasoning_effort,
            ),
            sampling_defaults=load_sampling_defaults(profile.tokenizer_path),
            reasoning=profile.reasoning,
            stop_token_ids=profile.stop_token_ids,
            reasoning_efforts=profile.reasoning_efforts,
            default_reasoning_effort=profile.default_reasoning_effort,
            profile=profile.name,
        )
        logger.info(
            "Loaded profile %s (tokenizer=%s, tool_call_parser=%s, "
            "generation_prompt=%r, assistant_turn_end=%r) for model %s",
            profile.name, profile.tokenizer_dir, parser_name or "disabled",
            profile.generation_prompt, profile.assistant_turn_end, model_name,
        )
        return bundle

    def fingerprint_payload(self, model_name: str) -> dict[str, Any]:
        """The tokenizer-identity fingerprint of the profile serving
        ``model_name`` (see :mod:`proxyserver.tokenization.fingerprint`),
        computed on the *runtime* tokenizer — the exact object every
        ``apply_chat_template``/``encode`` of this profile's sessions runs on
        — and cached per profile.

        Backs ``GET /tokenizer_fingerprint``; the trainer compares the
        ``fingerprint`` field against its checkpoint tokenizer's and the
        ``template_pin`` against an operator-pinned hash before any trial runs.
        """
        profile = resolve_profile(model_name)
        with self._lock:
            cached = self._fingerprints.get(profile.name)
        if cached is None:
            bundle = self.resolve(model_name)
            payload = _fingerprint.tokenizer_fingerprint(bundle.tokenizer)
            payload["profile"] = profile.name
            payload["generation_prompt"] = profile.generation_prompt
            payload["assistant_turn_end"] = profile.assistant_turn_end
            if profile.continuation_headers:
                payload["continuation_headers"] = list(profile.continuation_headers)
            if profile.assistant_turn_endings:
                payload["assistant_turn_endings"] = list(profile.assistant_turn_endings)
            payload["template_pin"] = _fingerprint.template_pin(
                payload["template_sha256"], profile.generation_prompt, profile.assistant_turn_end,
                continuation_headers=profile.continuation_headers,
                assistant_turn_endings=profile.assistant_turn_endings,
            )
            with self._lock:
                cached = self._fingerprints.setdefault(profile.name, payload)
        return {"model": str(model_name), **cached}

    def drop_session(self, session_id: str) -> None:
        """Forget a session's token stream in every loaded bundle."""
        with self._lock:
            bundles = list(self._bundles.values())
        for bundle in bundles:
            bundle.token_streams.drop(session_id)

    def has_session(self, session_id: str) -> bool:
        """Whether any loaded bundle still holds state for ``session_id``."""
        with self._lock:
            bundles = list(self._bundles.values())
        return any(bundle.token_streams.has_session(session_id) for bundle in bundles)
