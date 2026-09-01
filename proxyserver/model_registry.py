"""Model-name → :class:`~proxyserver.tokenization.base.ModelProfile` resolution,
driven by the agent's OpenAI request.

The proxy never configures tokenizer *paths* or tool-call formats.  Agents name
a model in the ``model`` field of their OpenAI ``/v1/chat/completions`` request,
and the proxy decides what that name means: ``tokenization/mapping.json`` maps
each served model name to a **profile** module under ``proxyserver/tokenization/``::

    {"Qwen3.5-35B-A3B": "qwen3_5", "Llama-3.1-8B-Instruct": "llama3_1", ...}

The profile (see :mod:`proxyserver.tokenization.base`) declares everything
format-specific about that model family — its bundled tokenizer directory, its
strict-TITO chat markers, its tool-call parser, and its reasoning format — in
one place, stated rather than inferred.

:class:`ModelRegistry` is the runtime half: it lazily loads and caches, per
profile, the tokenizer, the tool parser built on it, the reasoning parser, and
the strict-TITO :class:`~proxyserver.token_stream.TokenStreamManager` (built
with the profile's stated markers) — one bundle shared by every session of
every model name the profile serves.  Rollout providers resolve through it on
each session's first request and pin the session to that model (see
:mod:`proxyserver.rollout_provider`).

Usage::

    from proxyserver.model_registry import ModelRegistry

    registry = ModelRegistry()  # or ModelRegistry(tool_parser_factory=...)
    resolved = registry.resolve("Qwen3.5-35B-A3B")
    resolved.tokenizer      # HF AutoTokenizer (or the injected loader's)
    resolved.tool_parser    # the profile's tool parser built on that tokenizer
    resolved.token_streams  # TokenStreamManager (profile markers) for this family
    resolved.reasoning      # the profile's reasoning parser
"""

from __future__ import annotations
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from .token_stream import TokenStreamManager
from .tokenization import fingerprint as _fingerprint
from .tokenization.base import ModelProfile, get_profile
from .tokenization.tool_parser import NullToolParser, get_tool_parser

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent

#: Directory holding the bundled tokenizers and the profile modules.
TOKENIZATION_DIR = _PACKAGE_DIR / "tokenization"

#: Maps each served model name to its profile module (basename under
#: :data:`TOKENIZATION_DIR`, e.g. ``"qwen3_5"`` -> ``tokenization/qwen3_5.py``).
MODEL_MAPPING_PATH = TOKENIZATION_DIR / "mapping.json"


class UnknownModelError(ValueError):
    """No profile is mapped to the requested model name.

    Maps to an OpenAI-style 404 (``model_not_found``) at the HTTP layer.
    """


@dataclass(frozen=True)
class ModelSpec:
    """A compatibility view of a model's profile — its tokenizer directory and
    tool-call parser *name*.

    Superseded by :class:`~proxyserver.tokenization.base.ModelProfile`; kept
    for the e2e harness, which resolves a tokenizer path and parser name this
    way.  New code should use :func:`resolve_profile`.
    """

    #: Absolute path of the model's tokenizer directory.
    tokenizer_path: str
    #: Tool-call parser name (e.g. ``"qwen3_coder"``), or ``None`` if disabled.
    tool_call_parser: str | None


#: Parsed-mapping cache: ``((mtime_ns, size), {model_name: profile_name})``.
#: The mapping is consulted on every request (model validation, session
#: resolution), so it must not be re-read each time; keying on the file's stat
#: keeps mid-run edits working — an edited file is re-parsed exactly once.
_mapping_cache: tuple[tuple[int, int], dict[str, str]] | None = None
_mapping_lock = threading.Lock()


def load_model_mapping() -> dict[str, str]:
    """Parse ``tokenization/mapping.json`` into ``{model_name: profile_name}``.

    Cached on the file's ``(mtime, size)``: the hot path costs one ``stat``,
    and an edited file is re-parsed on its next use.  Profile modules are named
    but not imported here — that happens per model in :func:`resolve_profile`.
    """
    global _mapping_cache
    try:
        stat = MODEL_MAPPING_PATH.stat()
    except FileNotFoundError:
        raise FileNotFoundError(f"Model mapping not found: {MODEL_MAPPING_PATH}") from None
    key = (stat.st_mtime_ns, stat.st_size)
    with _mapping_lock:
        if _mapping_cache is not None and _mapping_cache[0] == key:
            return _mapping_cache[1]
    mapping = json.loads(MODEL_MAPPING_PATH.read_text())
    if not isinstance(mapping, dict):
        raise ValueError(
            f"{MODEL_MAPPING_PATH}: expected an object mapping each model name to "
            f'a profile module name (e.g. {{"Qwen3.5-9B": "qwen3_5"}})'
        )
    profiles: dict[str, str] = {}
    for model, profile_name in mapping.items():
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise ValueError(
                f"{MODEL_MAPPING_PATH}: entry for model {model!r} must name a "
                f"profile module under {TOKENIZATION_DIR} (e.g. \"qwen3_5\"), "
                f"got {profile_name!r}"
            )
        profiles[str(model)] = profile_name.strip()
    with _mapping_lock:
        _mapping_cache = (key, profiles)
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


def resolve_model_spec(model_name: Any) -> ModelSpec:
    """A :class:`ModelSpec` view of ``model_name``'s profile (compat).

    Prefer :func:`resolve_profile`.  Same exception contract as it.
    """
    profile = resolve_profile(model_name)
    parser = profile.tool_call_parser
    return ModelSpec(
        tokenizer_path=profile.tokenizer_path,
        tool_call_parser=parser.NAME if parser is not None else None,
    )


class _FrameworkToolParser:
    """Adapts a training framework's tool parser to the proxy's contract.

    A framework parser (e.g. verl's ``ToolParser``) implements
    ``extract_tool_calls(token_ids)``; the built-in parsers also accept the
    request's ``tools``, so they can type each argument by its declared
    schema.  Normalizing the difference here — the one place a foreign parser
    enters the system — keeps every call site on a single contract instead of
    type-testing the parser it holds.
    """

    def __init__(self, parser: Any) -> None:
        self._parser = parser

    async def extract_tool_calls(self, token_ids: list[int], tools: list[dict] | None = None):
        # The framework's parser has no use for the schemas; drop them.
        return await self._parser.extract_tool_calls(token_ids)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parser, name)


def build_tool_parser(
    tool_call_parser: str | None,
    tokenizer: Any,
    tool_parser_factory: Callable[[str, Any], Any] | None = None,
    reasoning: Any = None,
) -> Any:
    """Resolve the tool parser named ``tool_call_parser`` on ``tokenizer``.

    Returns a :class:`~proxyserver.tokenization.tool_parser.NullToolParser` when
    ``tool_call_parser`` is falsy — the profile disables tool-call extraction,
    but the agent-visible content is still cleaned exactly as the real parsers
    clean it.  ``reasoning`` is the profile's reasoning parser, threaded into
    the built-in parsers so they strip reasoning the model's own way (a
    framework factory owns its parser's reasoning, so it is not threaded
    there).  A training framework may own the parser via ``tool_parser_factory``
    (e.g. verl's ``ToolParser.get_tool_parser``).
    """
    if not tool_call_parser:
        return NullToolParser(tokenizer, reasoning=reasoning)
    if tool_parser_factory is None:
        return get_tool_parser(tool_call_parser, tokenizer, reasoning=reasoning)
    try:
        return _FrameworkToolParser(tool_parser_factory(tool_call_parser, tokenizer))
    except Exception as e:
        raise ValueError(
            f"tool_parser_factory rejected tool_call_parser "
            f"{tool_call_parser!r} ({type(e).__name__}: {e}); the parser names "
            f"a profile declares must be ones the training framework's factory accepts"
        ) from e


def load_local_tokenizer(tokenizer_path: str) -> Any:
    """Default tokenizer loader: ``AutoTokenizer`` from the bundled directory.

    Public so the e2e harness loads its verifier tokenizers through exactly
    this code path — a kwarg added here must reach the tests too, or their
    independent TITO verification checks against a different tokenizer than
    the providers use.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


#: The distribution-shaping sampling knobs: honored from a bundled
#: ``generation_config.json`` (the model publisher's recommended decoding
#: settings), and the only keys ``sampling_overrides`` may carry —
#: ``max_tokens`` and ``stop`` stay request-owned, since agent logic depends
#: on them.  Everything else in a ``generation_config.json`` (``do_sample``,
#: token ids, ...) is generation *mechanics*, not sampling policy, and is
#: deliberately ignored.
SAMPLING_PARAM_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
)


def validate_sampling_overrides(overrides: Any, where: str) -> dict[str, Any]:
    """Validate a ``sampling_overrides`` mapping and return it normalized.

    Overrides outrank the params agents put on their requests, so they are
    held to a strict contract: only the distribution keys in
    :data:`SAMPLING_PARAM_KEYS`, with numeric values.
    ``max_tokens`` and ``stop`` are rejected — agent logic owns those, and an
    override there would break tool-call and turn-boundary behavior rather
    than just reshape the sampling distribution.

    Returns the overrides re-keyed in :data:`SAMPLING_PARAM_KEYS` order, so
    two equal policies always compare and print identically.
    """
    if not isinstance(overrides, dict):
        raise ValueError(
            f"{where}: sampling overrides must be a mapping of sampling "
            f"params (e.g. {{temperature: 1.0}}), got {type(overrides).__name__}"
        )
    bad = sorted(set(overrides) - set(SAMPLING_PARAM_KEYS))
    if bad:
        raise ValueError(
            f"{where}: sampling overrides may only carry the distribution keys "
            f"({', '.join(SAMPLING_PARAM_KEYS)}); got {', '.join(map(str, bad))}. "
            f"max_tokens and stop stay request-owned."
        )
    for key, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: sampling override {key!r} must be a number, got {value!r}")
    return {key: overrides[key] for key in SAMPLING_PARAM_KEYS if key in overrides}


def load_sampling_defaults(tokenizer_path: str) -> dict[str, Any]:
    """The sampling defaults published in ``{tokenizer_path}/generation_config.json``.

    RL rollouts must not sample from the raw full distribution just because
    an agent's OpenAI request named no sampling params: model publishers ship
    the decoding settings the model was validated with (e.g. Qwen3.5:
    ``top_p=0.95, top_k=20``), and sampling outside them is exactly what
    sends long generations into repetition loops.  These defaults fill in
    for request-omitted params in
    :meth:`proxyserver.rollout_provider.BaseRolloutProvider.generate`.

    Missing or unreadable file → ``{}`` (the engines' own defaults apply).
    """
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
    #: Strict-TITO token streams of every session pinned to a model this
    #: profile serves (built with the profile's stated chat markers).
    token_streams: TokenStreamManager
    #: Publisher-recommended sampling defaults from the bundled
    #: ``generation_config.json`` (see :func:`load_sampling_defaults`);
    #: fill in for request-omitted sampling params.
    sampling_defaults: dict[str, Any] = field(default_factory=dict)
    #: The profile's reasoning parser (how the model delimits chain-of-thought).
    reasoning: Any = None


class ModelRegistry:
    """Lazily loads and caches :class:`ResolvedModel` bundles per profile.

    All model names mapped to the same profile share one bundle, so the
    tokenizer is loaded once no matter which of its model names sessions use.

    Thread-safety: :meth:`resolve` may be called from worker threads (the
    providers resolve off the event loop, next to tokenization); the cache is
    guarded by a lock, so a first load never runs twice.
    """

    def __init__(
        self,
        tool_parser_factory: Callable[[str, Any], Any] | None = None,
        tokenizer_loader: Callable[[str], Any] | None = None,
    ) -> None:
        """
        Args:
            tool_parser_factory: Optional callable
                ``(tool_call_parser, tokenizer) -> tool_parser`` supplied by
                the training framework; it receives each profile's tool parser
                *name* (its ``NAME``).  Without it the profile's built-in
                parser class serves.
            tokenizer_loader: Callable ``(tokenizer_dir) -> tokenizer``
                overriding the default ``AutoTokenizer.from_pretrained`` —
                the injection seam the offline tests use for their fake
                tokenizers.
        """
        # Validate every mapped profile eagerly: a bad mapping or a broken
        # profile module fails here, not on the first request minutes into a
        # rollout.  (Importing a profile is cheap — no tokenizer loads.)
        mapping = load_model_mapping()  # malformed/missing mapping fails here
        for model, profile_name in mapping.items():
            try:
                get_profile(profile_name)
            except ValueError as e:
                raise ValueError(f"{MODEL_MAPPING_PATH}: model {model!r}: {e}") from None
        self._tool_parser_factory = tool_parser_factory
        self._tokenizer_loader = tokenizer_loader or load_local_tokenizer
        if tokenizer_loader is None:
            from transformers import AutoTokenizer  # noqa: F401
        #: Keyed by profile name; every model name of a profile shares one bundle.
        self._bundles: dict[str, ResolvedModel] = {}
        #: Tokenizer-identity fingerprints, keyed by profile name (computing
        #: one hashes the full vocab — cheap, but not per-request cheap).
        self._fingerprints: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def resolve(self, model_name: str) -> ResolvedModel:
        """The (cached) bundle serving ``model_name``."""
        profile = resolve_profile(model_name)

        with self._lock:
            bundle = self._bundles.get(profile.name)
            if bundle is not None:
                return bundle

        # Load the tokenizer outside the lock (it reads from disk, or the
        # injected loader may do I/O); a racing resolve for the same profile is
        # caught by the re-check below.
        tokenizer = self._tokenizer_loader(profile.tokenizer_path)

        with self._lock:
            bundle = self._bundles.get(profile.name)
            if bundle is not None:  # another thread won the race for this profile
                return bundle
            parser_name = profile.tool_call_parser.NAME if profile.tool_call_parser is not None else None
            bundle = ResolvedModel(
                tokenizer=tokenizer,
                tool_parser=build_tool_parser(
                    parser_name, tokenizer, self._tool_parser_factory, reasoning=profile.reasoning
                ),
                token_streams=TokenStreamManager(
                    tokenizer,
                    generation_prompt=profile.generation_prompt,
                    assistant_turn_end=profile.assistant_turn_end,
                ),
                sampling_defaults=load_sampling_defaults(profile.tokenizer_path),
                reasoning=profile.reasoning,
            )
            self._bundles[profile.name] = bundle
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
        ``fingerprint`` field against its ``--hf-checkpoint``'s and the
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
            payload["template_pin"] = _fingerprint.template_pin(
                payload["template_sha256"], profile.generation_prompt, profile.assistant_turn_end
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
