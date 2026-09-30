"""Shared fixtures and verification for direct and Slime integration tests.

Loads live endpoints from test/test_engines.yaml, overridable by environment,
and bundled tokenizers from the production registry. Includes independent
native HTTP clients for the wire-contract test and the multi-agent gauntlet.
The default transport used to resolve config is direct.
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import httpx
import openai
import yaml
from openai import AsyncOpenAI
# Make the repo-root `proxyserver` package importable when the tests run as
# plain scripts (python test/test-*.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proxyserver.config import load_config  # noqa: E402
from proxyserver.engines import SGLANG, VLLM, get_adapter  # noqa: E402
from proxyserver.model_registry import build_tool_parser, load_local_tokenizer, resolve_profile  # noqa: E402
from proxyserver.rollout_provider import engine_http_limits  # noqa: E402
from proxyserver.server import LLMProxyServer, PROXY_KEY_DELIMITER  # noqa: E402
from proxyserver.token_stream import (  # noqa: E402
    TokenStreamManager,
    as_token_ids,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("proxy-test")


ENGINES_YAML = Path(__file__).resolve().parent / "test_engines.yaml"


@dataclass(frozen=True)
class EngineEndpoint:
    """One running inference engine, as listed in ``test/test_engines.yaml``."""

    engine: str      # normalized: "vllm" | "sglang"
    base_url: str    # OpenAI-compatible root, ends in /v1
    api_key: str
    model: str

    @property
    def root_url(self) -> str:
        """Server root, for the engines' non-OpenAI routes (SGLang's /generate)."""
        return self.base_url.removesuffix("/v1")


def _fixture_endpoints(engine: str) -> list[EngineEndpoint]:
    """All endpoints of ``engine`` from its section of ``test/test_engines.yaml``.

    ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` are lists (one entry per running
    engine instance; a single api_key is broadcast to every base_url), so one
    section yields one :class:`EngineEndpoint` per entry.  Returns ``[]`` when
    the file or the engine's section is absent, so the env vars alone can
    drive the tests.
    """
    if not ENGINES_YAML.is_file():
        return []
    section = (yaml.safe_load(ENGINES_YAML.read_text()) or {}).get(engine) or {}
    base_urls = section.get("OPENAI_BASE_URL") or []
    api_keys = section.get("OPENAI_API_KEY") or []
    if not isinstance(base_urls, list):
        base_urls = [base_urls]
    if not isinstance(api_keys, list):
        api_keys = [api_keys]
    if len(api_keys) == 1 and len(base_urls) > 1:
        api_keys = api_keys * len(base_urls)
    model = section.get("MODEL_NAME")
    if base_urls and (len(api_keys) != len(base_urls) or not model):
        raise RuntimeError(
            f"{ENGINES_YAML}: section {engine!r} needs MODEL_NAME and one "
            f"OPENAI_API_KEY per OPENAI_BASE_URL (or a single shared key)"
        )
    return [
        EngineEndpoint(engine, str(base_url), str(api_key), str(model))
        for base_url, api_key in zip(base_urls, api_keys)
    ]


# Engine under test — the same name the proxy takes for `inference_engine`,
# so an unsupported value fails here exactly as it would in production.
ENGINE = get_adapter(os.getenv("INFERENCE_ENGINE")).name


def live_test_engines() -> tuple[str, ...]:
    """Run both engines by default, or just the engine selected by environment.

    Generic endpoint overrides describe one engine, defaulting to vLLM.
    Never send both protocols to that one overridden endpoint.
    """
    if any(os.getenv(key) for key in (
        "INFERENCE_ENGINE", "OPENAI_BASE_URL", "OPENAI_API_KEY", "MODEL_NAME",
    )):
        return (get_adapter(os.getenv("INFERENCE_ENGINE")).name,)
    return (VLLM, SGLANG)


def load_engine_endpoints(engine: str) -> list[EngineEndpoint]:
    """Endpoints for ``engine``: env vars win, then ``test/test_engines.yaml``.

    Without env overrides, every endpoint listed in the fixture file is
    returned; an override collapses the list to that single endpoint (missing
    fields fall back to the file's first entry).
    """
    engine = get_adapter(engine).name
    rows = _fixture_endpoints(engine)
    env_base_url = os.getenv("OPENAI_BASE_URL")
    env_api_key = os.getenv("OPENAI_API_KEY")
    env_model = os.getenv("MODEL_NAME")
    if not (env_base_url or env_api_key or env_model):
        if not rows:
            raise RuntimeError(
                f"No endpoint for engine {engine!r}: add a section to {ENGINES_YAML} "
                f"or set $OPENAI_BASE_URL / $OPENAI_API_KEY / $MODEL_NAME"
            )
        return rows
    first = rows[0] if rows else None
    base_url = env_base_url or (first.base_url if first else None)
    api_key = env_api_key or (first.api_key if first else None)
    model = env_model or (first.model if first else None)
    if not (base_url and api_key and model):
        raise RuntimeError(
            f"No endpoint for engine {engine!r}: add a section to {ENGINES_YAML} "
            f"or set $OPENAI_BASE_URL / $OPENAI_API_KEY / $MODEL_NAME"
        )
    return [EngineEndpoint(engine, base_url, api_key, model)]


ENDPOINTS = load_engine_endpoints(ENGINE)
ENDPOINT = ENDPOINTS[0]
MODEL_NAME = ENDPOINT.model

# Delimiter for keyed routing, from the engine's config (`proxy_api_delimiter`);
# the tests construct the proxy with the same delimiter.
try:
    KEY_DELIMITER = load_config(inference_engine=ENGINE).proxy_api_delimiter
except FileNotFoundError:
    KEY_DELIMITER = PROXY_KEY_DELIMITER

# Shared secret for keyed routing
# ("{REAL_API_KEY}{delimiter}{session_id}{delimiter}{agent_id}"): the tests
# construct the proxy with this key.
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "test-proxy-real-key")
# agent_id for the single-mock-agent-per-session helpers below (truncation,
# streaming, model-validation, tamper, bad-key checks).
DEFAULT_AGENT_ID = "agent"
ALL_AGENT_PROMPTS = [
    ("agent_0", "What is 17 * 23? Answer with just the number.",
     "Now add 100 to that result. Answer with just the number."),
    ("agent_1", "Name the capital of France in one word.",
     "And the capital of Japan, in one word?"),
    ("agent_2", "Write a haiku about the ocean.",
     "Now translate that haiku into French."),
    ("agent_3", "List three prime numbers greater than 50.",
     "Which of those three is the largest?"),
    ("agent_4", "Who are you?",
     "What can you do?"),
    # The five below stress token boundaries in the echoed history:
    # unicode/emoji, code, JSON escapes, CJK, and exact-sequence continuation.
    ("agent_5", "Repeat exactly: naïve café — 你好, 世界! 🌊🚀",
     "Repeat it once more, unchanged."),
    ("agent_6", "Write a Python one-liner that reverses a string.",
     "Rewrite it as a full function with type hints."),
    ("agent_7", 'Output this JSON verbatim: {"a": 1, "b": [true, null]}',
     'Now add a key "c" whose value is the two-line string "line1\\nline2".'),
    ("agent_8", "用中文写一句关于秋天的诗。",
     "把它翻译成英文。"),
    ("agent_9", "Count from 1 to 30, separated by commas.",
     "Continue counting from 31 to 60."),
    ("agent_10", "What is 2 to the power of 16? Just the number.",
     "And 2 to the power of 20? Just the number."),
    ("agent_11", "Name three colors of the rainbow.",
     "Name the remaining four."),
    ("agent_12", "Spell the word 'strawberry' letter by letter, separated by dashes.",
     "How many times does the letter r appear in it?"),
    ("agent_13", "Give me a two-sentence bedtime story about a robot.",
     "Retell it from the robot's point of view."),
    ("agent_14", "What is the chemical formula of water?",
     "And of carbon dioxide?"),
    ("agent_15", "Sort these numbers ascending: 42, 7, 19, 3, 88.",
     "Now give them in descending order."),
    ("agent_16", "Write a SQL query that selects all rows from a table named users.",
     "Modify it to return only the 10 newest rows by created_at."),
    ("agent_17", "Translate 'good morning' into Spanish, French, and German.",
     "Now into Italian, Portuguese, and Dutch."),
    ("agent_18", "State the Pythagorean theorem in one sentence.",
     "Apply it: legs 3 and 4, what is the hypotenuse?"),
    ("agent_19", "List the first five Fibonacci numbers.",
     "Continue the sequence for five more."),
    ("agent_20", "Write a limerick about a cat who codes.",
     "Rewrite it so the cat is a dog."),
    ("agent_21", "What year did the first human walk on the moon?",
     "Who was it?"),
    ("agent_22", "Give a regex that matches an email address.",
     "Explain each part of that regex briefly."),
    ("agent_23", "Summarize the plot of Romeo and Juliet in one sentence.",
     "Now in exactly five words."),
    ("agent_24", "Convert 100 degrees Fahrenheit to Celsius, one decimal place.",
     "And to Kelvin, one decimal place."),
    ("agent_25", "Write a haiku that contains the word 'byte'.",
     "Replace 'byte' with 'bit' and adjust the syllables if needed."),
    # More token-boundary stress: mixed scripts, math symbols, markdown,
    # escapes, base64-ish strings, and long exact repetition.
    ("agent_26", "Repeat exactly: Ω ≈ 3.14— µ±σ №42 «quoted» …done",
     "Repeat it again, then append the word END."),
    ("agent_27", "Output a markdown table with two rows: name|age, Bob|30, Ann|25.",
     "Add a third row: Cid|41."),
    ("agent_28", "Repeat exactly: SGVsbG8sIFdvcmxkIQ==",
     "Repeat it twice on one line separated by a single space."),
    ("agent_29", "日本語で自己紹介を一文で書いてください。",
     "その文を韓国語に翻訳してください。"),
    ("agent_30", 'Print this with escapes intact: line1\\n\\ttab "quoted" back\\\\slash',
     "Print it again, unchanged."),
    ("agent_31", "Write the word 'echo' exactly 20 times, space separated.",
     "Now exactly 5 more times, comma separated."),
    ("agent_32", "What is 12! (twelve factorial)? Just the number.",
     "Divide it by 12 to get 11!. Just the number."),
    ("agent_33", "Name the four largest planets of the solar system.",
     "Order them by distance from the sun."),
    ("agent_34", "Write a CSS rule that centers a div horizontally.",
     "Extend it to center vertically as well."),
    ("agent_35", "Give the first line of a famous English novel and name the book.",
     "Now do the same for a famous poem."),
    ("agent_36", "Convert 3 hours 45 minutes to minutes. Just the number.",
     "And to seconds. Just the number."),
    ("agent_37", "Write a YAML snippet with keys name, version, and a list deps of two items.",
     "Convert that snippet to JSON."),
    ("agent_38", "State Newton's second law as a formula.",
     "Solve it for acceleration."),
    ("agent_39", "Give three synonyms of 'fast'.",
     "Now three antonyms."),
    ("agent_40", "Write a bash one-liner that counts files in the current directory.",
     "Modify it to count only .txt files recursively."),
    ("agent_41", "Spell 'Mississippi' backwards.",
     "How many letters does it have? Just the number."),
    # More token-boundary stress: RTL text, Cyrillic/Greek/Thai/Korean,
    # LaTeX, URLs with query strings, HTML entities, and typographic quotes.
    ("agent_42", "Repeat exactly: مرحبا بالعالم — שלום עולם",
     "Repeat it once more, then append DONE."),
    ("agent_43", "Repeat exactly: Привет мир! Γειά σου κόσμε! สวัสดีชาวโลก",
     "Repeat only the Greek part."),
    ("agent_44", "안녕하세요를 사용하는 한국어 인사말 문장을 하나 쓰세요.",
     "그 문장을 영어로 번역하세요."),
    ("agent_45", "Write the quadratic formula in LaTeX.",
     "Now write the Pythagorean theorem in LaTeX."),
    ("agent_46", "Repeat exactly: https://example.com/search?q=a+b&lang=zh-CN&page=2#results",
     "Repeat it with page=3 instead."),
    ("agent_47", 'Repeat exactly: &lt;div class="x"&gt; &amp;nbsp; &lt;/div&gt;',
     "Repeat it once more, unchanged."),
    ("agent_48", "Repeat exactly: “curly quotes” and ‘single’ — em-dash… ellipsis",
     "Repeat it, replacing the em-dash with a colon."),
    ("agent_49", "Count down from 20 to 1, separated by semicolons.",
     "Now count up from 1 to 20, separated by spaces."),
]


def live_agent_prompts() -> list[tuple[str, str, str]]:
    mode = os.getenv("LIVE_TEST_MODE", "full")
    if mode == "full":
        return list(ALL_AGENT_PROMPTS)
    if mode == "smoke":
        # Unicode, code, JSON, CJK, symbols, escaping, RTL, and continuation.
        selected = {5, 6, 7, 8, 26, 30, 42, 49}
        return [spec for i, spec in enumerate(ALL_AGENT_PROMPTS) if i in selected]
    raise ValueError(f"Unknown LIVE_TEST_MODE {mode!r}; expected 'full' or 'smoke'")


AGENT_PROMPTS = live_agent_prompts()
NUM_AGENTS = len(AGENT_PROMPTS)


def parse_token_ids(tokens: list[str]) -> list[int]:
    """Parse vLLM ``return_tokens_as_token_ids`` strings ("token_id:<id>")."""
    return [
        int(tok.split(":", 1)[1])
        for tok in tokens
        if tok.startswith("token_id:")
    ]


# ---------------------------------------------------------------------------
# Native HTTP clients for independent engine-contract checks
# ---------------------------------------------------------------------------


# Connection-pool limits for the engine clients — imported from production
# (keep-alive race avoidance + pool caps, see engine_http_limits's docstring)
# so the harness exercises exactly the connection behavior the providers use.
ENGINE_HTTP_LIMITS = engine_http_limits()


@dataclass
class TokenOutput:
    """Token IDs, aligned logprobs, and the native finish reason."""

    token_ids: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    stop_reason: str | None = None


class VLLMEngineClient:
    """Independent vLLM /v1/completions wire-contract client."""

    engine = VLLM

    def __init__(self, endpoint: EngineEndpoint):
        self._endpoint = endpoint
        self._http = httpx.AsyncClient(
            base_url=endpoint.base_url,
            headers={"Authorization": f"Bearer {endpoint.api_key}"},
            timeout=900,
            limits=ENGINE_HTTP_LIMITS,
        )

    async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> TokenOutput:
        payload: dict[str, Any] = {
            "model": self._endpoint.model,
            "prompt": prompt_ids,  # token-IDs in
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 1.0),
            "max_tokens": sampling_params.get("max_tokens", 2048),
            # Zero requests only the sampled token's logprob.
            "logprobs": 0 if sampling_params.get("logprobs") else None,
            "return_tokens_as_token_ids": True,
        }
        for key in ("stop", "repetition_penalty"):
            if sampling_params.get(key) is not None:
                payload[key] = sampling_params[key]

        resp = await self._http.post("/completions", json=payload)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]

        finish_reason = choice.get("finish_reason")
        return TokenOutput(
            token_ids=parse_token_ids(choice["logprobs"]["tokens"]),
            log_probs=list(choice["logprobs"]["token_logprobs"]),
            stop_reason=finish_reason,
        )

    async def aclose(self) -> None:
        await self._http.aclose()


class SGLangEngineClient:
    """Independent SGLang /generate wire-contract client."""

    engine = SGLANG

    def __init__(self, endpoint: EngineEndpoint):
        self._http = httpx.AsyncClient(
            base_url=endpoint.root_url,
            headers={"Authorization": f"Bearer {endpoint.api_key}"},
            timeout=900,
            limits=ENGINE_HTTP_LIMITS,
        )

    async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any]) -> TokenOutput:
        params = dict(sampling_params)
        max_new_tokens = params.pop("max_new_tokens", None) or params.pop("max_tokens", 2048)
        return_logprob = bool(params.pop("logprobs", False))
        params.pop("prompt_logprobs", None)
        params["max_new_tokens"] = max_new_tokens
        params = {k: v for k, v in params.items() if v is not None}

        resp = await self._http.post("/generate", json={
            "input_ids": prompt_ids,
            "sampling_params": params,
            "return_logprob": return_logprob,
        })
        resp.raise_for_status()
        output = resp.json()

        meta_info = output.get("meta_info", {})
        finish_reason = meta_info.get("finish_reason")
        stop_reason = finish_reason["type"] if finish_reason else None

        if return_logprob:
            triples = meta_info.get("output_token_logprobs") or []
            token_ids = list(output.get("output_ids", [tid for _, tid, _ in triples]))
            output_token_logprobs = meta_info.get("output_token_logprobs") or []
            if output_token_logprobs and len(output_token_logprobs) == len(token_ids):
                log_probs = [float(lp) for lp, _, _ in output_token_logprobs]
            else:
                # Invalid native output must fail the contract check.
                logger.error(
                    "SGLang logprob/token length mismatch (%d vs %d)",
                    len(output_token_logprobs), len(token_ids),
                )
                token_ids, log_probs = [], []
        else:
            token_ids, log_probs = list(output["output_ids"]), []

        return TokenOutput(token_ids=token_ids, log_probs=log_probs, stop_reason=stop_reason)

    async def aclose(self) -> None:
        await self._http.aclose()


def make_engine_client(endpoint: EngineEndpoint):
    """Build the native HTTP client matching ``endpoint.engine``."""
    if endpoint.engine == SGLANG:
        return SGLangEngineClient(endpoint)
    return VLLMEngineClient(endpoint)


#: Bundled tokenizers already loaded, keyed by tokenizer directory — the
#: verifier and the contract test all share one instance.
_TOKENIZER_CACHE: dict[str, Any] = {}


def load_bundled_tokenizer(model: str) -> Any:
    """The bundled tokenizer serving ``model``, loaded once per directory.

    Resolves ``model`` exactly as the proxy does (``tokenization/mapping.json``
    via :func:`proxyserver.model_registry.resolve_profile`).  The harness
    never asks the engine to tokenize — no ``/tokenize`` / ``/detokenize``
    round-trip — it tokenizes locally with the same bundled tokenizers the
    proxy uses in production.
    """
    path = resolve_profile(model).tokenizer_path
    tokenizer = _TOKENIZER_CACHE.get(path)
    if tokenizer is None:
        tokenizer = load_local_tokenizer(path)  # the registry's own loader
        _TOKENIZER_CACHE[path] = tokenizer
    return tokenizer


def make_verifier(model: str) -> TokenStreamManager:
    """Independent token-delta verifier on the bundled tokenizer."""
    profile = resolve_profile(model)
    return TokenStreamManager(
        load_bundled_tokenizer(model), generation_prompt=profile.generation_prompt,
        assistant_turn_end=profile.assistant_turn_end,
    )


def expected_agent_content(model: str, token_ids: list[int]) -> str:
    """The agent-visible content production derives from sampled token IDs.

    Mirrors the server-side pipeline exactly
    (``BaseToolParser.extract_tool_calls``): decode without special tokens,
    strip reasoning according to the model profile, parse tool-call markup out of the
    content.  The recorded ``completion_text`` is deliberately the *raw*
    decode (training data keeps everything, ``<think>`` and EOS included),
    so it can differ from the agent's reply when reasoning or special tokens
    are present.
    """
    tokenizer = load_bundled_tokenizer(model)
    profile = resolve_profile(model)
    parser_name = profile.tool_call_parser.NAME if profile.tool_call_parser is not None else None
    parser = build_tool_parser(parser_name, tokenizer, reasoning=profile.reasoning)
    content, _calls = parser.parse(
        profile.reasoning.strip(tokenizer.decode(token_ids, skip_special_tokens=True))
    )
    return content


async def assert_length_truncation(
    proxy_url: str,
    http: httpx.AsyncClient,
    proxy_api_key: str,
    *,
    max_tokens: int = 16,
) -> None:
    """A response cut off at ``max_tokens`` must be reported as ``"length"``.

    Native HTTP engines report truncation explicitly; the proxy falls back
    to the effective token budget when a finish reason is absent or unknown.
    A budget-exhausting response must arrive here as
    ``finish_reason == "length"``, in the OpenAI response *and* in the
    recorded turn the trainer will read.
    """
    sid = "trial_truncate"
    client = AsyncOpenAI(
        base_url=f"{proxy_url}/v1",
        api_key=f"{proxy_api_key}{KEY_DELIMITER}{sid}{KEY_DELIMITER}{DEFAULT_AGENT_ID}",
        timeout=900,
        max_retries=0,
    )
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Count from 1 to 500, separated by commas."}],
            temperature=0.6,
            max_tokens=max_tokens,
        )
        choice = completion.choices[0]
        assert choice.finish_reason == "length", (
            f"truncated response reported finish_reason={choice.finish_reason!r}, "
            f"expected 'length' (engine={ENGINE})"
        )
        assert completion.usage.completion_tokens == max_tokens, (
            f"expected exactly {max_tokens} completion tokens, "
            f"got {completion.usage.completion_tokens}"
        )
    finally:
        await client.close()

    resp = await http.get(f"{proxy_url}/sessions/{sid}")
    assert resp.status_code == 200, f"session fetch failed: {resp.text}"
    turn = resp.json()["turns"][0]
    assert turn["finish_reason"] == "length", (
        f"recorded finish_reason={turn['finish_reason']!r}, expected 'length' — "
        f"the trainer cannot mask truncated rollouts it is not told about"
    )
    assert len(turn["completion_token_ids"]) == max_tokens
    assert len(turn["completion_logprobs"]) == max_tokens

    logger.info("Truncation correctly reported as finish_reason='length' (engine=%s)", ENGINE)
    await http.delete(f"{proxy_url}/sessions/{sid}")


async def assert_streaming_supported(
    proxy_url: str,
    http: httpx.AsyncClient,
    proxy_api_key: str,
) -> None:
    """``stream: true`` must be honored identically with both transports.

    Strict TITO forces the proxy to generate the whole turn before replying,
    so streaming is a *reply format*, not incremental generation: the client
    must receive valid ``chat.completion.chunk`` SSE events whose assembled
    text equals the recorded turn, ending with a finish_reason chunk.
    """
    sid = "trial_stream"
    client = AsyncOpenAI(
        base_url=f"{proxy_url}/v1",
        api_key=f"{proxy_api_key}{KEY_DELIMITER}{sid}{KEY_DELIMITER}{DEFAULT_AGENT_ID}",
        timeout=900,
        max_retries=0,
    )
    try:
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Name any three animals, comma separated."}],
            temperature=0.6,
            max_tokens=4096,
            stream=True,
        )
        text_parts: list[str] = []
        finish_reason = None
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta and choice.delta.content:
                text_parts.append(choice.delta.content)
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        reply = "".join(text_parts)
        assert reply.strip(), "streamed completion carried no content"
        assert finish_reason in ("stop", "length"), (
            f"streamed finish_reason={finish_reason!r}, expected 'stop' or 'length'"
        )
    finally:
        await client.close()

    # The streamed text must be exactly what the recorded turn's token IDs
    # clean to — the stream is a replay of the completed, already-recorded
    # response (the record itself keeps the raw decode).
    resp = await http.get(f"{proxy_url}/sessions/{sid}")
    assert resp.status_code == 200, f"session fetch failed: {resp.text}"
    turn = resp.json()["turns"][0]
    assert reply == expected_agent_content(MODEL_NAME, turn["completion_token_ids"]), (
        "streamed text differs from the recorded turn's cleaned content"
    )
    assert len(turn["completion_token_ids"]) > 0
    assert len(turn["completion_logprobs"]) == len(turn["completion_token_ids"])

    logger.info("Streaming honored as SSE replay (%d chars, finish=%s)", len(reply), finish_reason)
    await http.delete(f"{proxy_url}/sessions/{sid}")


async def assert_model_validation(
    proxy_url: str,
    http: httpx.AsyncClient,
    proxy_api_key: str,
) -> None:
    """The agent's ``model`` field is a claim the proxy decides on.

    The model comes from the request — never from proxy config: a model with
    no entry in ``tokenization/mapping.json`` must be rejected
    with an OpenAI 404, a request naming no model (``model=""``) with a 400,
    and — after a session's first turn pins its model — naming a *different*
    (even mapped) model must be rejected with a 400, like any other TITO
    violation.  The pinned model must be recorded on the session.
    """
    sid = "trial_model_claim"
    client = AsyncOpenAI(
        base_url=f"{proxy_url}/v1",
        api_key=f"{proxy_api_key}{KEY_DELIMITER}{sid}{KEY_DELIMITER}{DEFAULT_AGENT_ID}",
        timeout=900,
        max_retries=0,
    )
    other_mapped = "Qwen3.5-4B" if MODEL_NAME != "Qwen3.5-4B" else "Qwen3.5-9B"
    try:
        # 1. Unknown model -> 404, before any inference.
        try:
            await client.chat.completions.create(
                model="model-with-no-tokenizer",
                messages=[{"role": "user", "content": "Say OK."}],
                max_tokens=8,
            )
            raise AssertionError("a model with no tokenizer in the mapping was accepted")
        except openai.NotFoundError:
            pass

        # 2. Missing model -> 400: the agent decides the model, so it must
        #    actually say which one.
        try:
            await client.chat.completions.create(
                model="",
                messages=[{"role": "user", "content": "Say OK."}],
                max_tokens=8,
            )
            raise AssertionError("a request naming no model was accepted")
        except openai.BadRequestError:
            pass

        # 3. The first turn pins the session's model...
        first = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say OK."}],
            temperature=0.6,
            max_tokens=64,
        )
        assert first.model == MODEL_NAME, (
            f"response echoed model {first.model!r}, expected {MODEL_NAME!r}"
        )
        # ...and a later turn naming another model — even a mapped one — is
        # rejected: switching models mid-session would switch tokenizers
        # mid-stream.
        followup = [
            {"role": "user", "content": "Say OK."},
            {"role": "assistant", "content": first.choices[0].message.content or ""},
            {"role": "user", "content": "Say OK again."},
        ]
        try:
            await client.chat.completions.create(
                model=other_mapped,
                messages=followup,
                temperature=0.6,
                max_tokens=64,
            )
            raise AssertionError("a mid-session model switch was accepted")
        except openai.BadRequestError as e:
            assert "pinned" in str(e), f"unexpected rejection: {e}"
    finally:
        await client.close()

    resp = await http.get(f"{proxy_url}/sessions/{sid}")
    assert resp.status_code == 200, f"session fetch failed: {resp.text}"
    assert resp.json()["model_name"] == MODEL_NAME, "pinned model not recorded on the session"

    logger.info(
        "Model claims decided by the proxy: unknown model -> 404, missing "
        "model -> 400, mid-session switch -> 400, %s pinned and recorded",
        MODEL_NAME,
    )
    await http.delete(f"{proxy_url}/sessions/{sid}")


def _expected_generation_prompt(verifier: TokenStreamManager) -> list[int]:
    """The generation prompt the trainer appends to every render.

    Renders happen with ``add_generation_prompt=False`` and the bare
    assistant header from the model profile is appended instead of the
    template's own generation prompt (which may pre-fill reasoning).
    The harness tokenizes the stated header so the expectation stays independent of the
    manager's internals."""
    return as_token_ids(
        verifier.tokenizer.encode(verifier.generation_prompt, add_special_tokens=False),
        what="generation prompt",
    )


def _expected_turn_gap(verifier: TokenStreamManager, stream: list[int]) -> list[int]:
    """Close the previous assistant message the way the chat template does.

    The profile states the assistant ending. For Qwen, the engine samples
    ``<|im_end|>`` and the trainer supplies its trailing newline; Llama's
    sampled ``<|eot_id|>`` needs no suffix. A truncated turn may need the
    whole ending. Reimplemented here rather than called off the manager, so the
    harness stays an independent check on it."""
    end = as_token_ids(
        verifier.tokenizer.encode(verifier.assistant_turn_end, add_special_tokens=False),
        what="assistant turn end",
    )
    for have in range(len(end), 0, -1):
        if stream[-have:] == end[:have]:
            return end[have:]
    return end


def _expected_gap_delta(
    verifier: TokenStreamManager, message: str, stream_before: list[int]
) -> list[int]:
    """Tokenize one inter-turn user message the way the trainer does.

    ``stream_before`` is the session's token stream as of the end of the
    previous turn — it decides how much of the assistant turn ending the
    delta must carry (see :func:`_expected_turn_gap`).
    """
    # as_token_ids: transformers>=5 returns a BatchEncoding here, the
    # harness compares against plain token-ID lists.
    delta = as_token_ids(
        verifier.tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            add_generation_prompt=False,
            tokenize=True,
        ),
        what="inter-turn delta",
    )
    sys_ids = verifier._get_sys_ids()
    if sys_ids and delta[: len(sys_ids)] == sys_ids:
        delta = delta[len(sys_ids):]
    return _expected_turn_gap(verifier, stream_before) + delta + _expected_generation_prompt(verifier)


async def run_agent(
    proxy_url: str,
    http: httpx.AsyncClient,
    agent_id: str,
    prompt_turn1: str,
    prompt_turn2: str,
    *,
    max_tokens: int,
    verifier: TokenStreamManager,
    proxy_api_key: str,
) -> dict[str, Any]:
    """Simulate one agent: hold a 2-turn conversation through the proxy,
    then verify the proxy's session record (including strict TITO and
    delta storage: each turn records only the messages / prompt tokens the
    request appended to the session's token stream, and the inter-turn
    delta must match the trainer's tokenization scheme).

    Keyed routing — there is **no registration call**: the agent hits the
    plain ``/v1`` base_url with ``api_key = {key}{delimiter}{session_id}{delimiter}{agent_id}``
    and the proxy opens the session lazily on the first completion request."""
    session_id = f"trial_{agent_id}"

    # The agent only sees an OpenAI-compatible base_url + api_key.  SDK
    # retries are safe in production — an exact retry of a request whose
    # first attempt completed server-side is re-served the cached turn —
    # but stay off here so every recorded turn maps 1:1 to a scripted
    # request and the verification below stays deterministic.
    agent_client = AsyncOpenAI(
        base_url=f"{proxy_url}/v1",
        api_key=f"{proxy_api_key}{KEY_DELIMITER}{session_id}{KEY_DELIMITER}{agent_id}",
        timeout=900,
        max_retries=0,
    )
    try:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt_turn1}]
        replies: list[str] = []
        for turn, next_prompt in ((1, prompt_turn2), (2, None)):
            completion = await agent_client.chat.completions.create(
                model=MODEL_NAME,  # the proxy validates this claim and pins it to the session
                messages=messages,
                temperature=0.6,
                max_tokens=max_tokens,
            )
            choice = completion.choices[0]
            reply = choice.message.content or ""
            assert reply.strip(), f"[{agent_id}] empty completion on turn {turn}"
            assert completion.usage.completion_tokens > 0, f"[{agent_id}] no completion tokens reported on turn {turn}"
            assert completion.usage.prompt_tokens > 0, f"[{agent_id}] no prompt tokens reported on turn {turn}"
            # Native finish reasons remain authoritative when a context
            # clamp reduces the effective budget below the requested one.
            assert choice.finish_reason in ("stop", "length"), (
                f"[{agent_id}] turn {turn}: unexpected finish_reason={choice.finish_reason!r}"
            )
            # The response echoes the effective model the proxy resolved.
            assert completion.model == MODEL_NAME, (
                f"[{agent_id}] turn {turn}: response model={completion.model!r}, "
                f"expected {MODEL_NAME!r}"
            )
            n_completion = completion.usage.completion_tokens
            assert n_completion <= max_tokens, (
                f"[{agent_id}] turn {turn}: {n_completion} completion tokens "
                f"exceed the requested budget {max_tokens}"
            )
            replies.append(reply)
            logger.info(
                "[%s] turn %d ok (%d prompt + %d completion tokens, finish=%s): %.60r",
                agent_id, turn, completion.usage.prompt_tokens,
                completion.usage.completion_tokens, choice.finish_reason, reply,
            )
            if next_prompt is not None:
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": next_prompt})
    finally:
        await agent_client.close()

    # Verify what the proxy recorded for this session.
    resp = await http.post(f"{proxy_url}/sessions/{session_id}/complete")
    assert resp.status_code == 200
    resp = await http.get(f"{proxy_url}/sessions/{session_id}")
    assert resp.status_code == 200, f"[{agent_id}] session fetch failed: {resp.text}"
    record = resp.json()

    assert record["session_id"] == session_id
    assert record["completed"] is True
    assert record["model_name"] == MODEL_NAME, (
        f"[{agent_id}] recorded model_name={record.get('model_name')!r}, "
        f"expected {MODEL_NAME!r} — the trainer needs to know which tokenizer "
        f"built this session's prompts"
    )
    assert len(record["turns"]) == 2, f"[{agent_id}] expected 2 recorded turns, got {len(record['turns'])}"
    stream: list[int] = []  # authoritative token stream, rebuilt turn by turn
    for i, turn_record in enumerate(record["turns"]):
        # The record keeps the RAW decode (training data: reasoning + EOS
        # included), faithful to the recorded token IDs...
        raw = verifier.tokenizer.decode(turn_record["completion_token_ids"], skip_special_tokens=False)
        assert turn_record["completion_text"] == raw, (
            f"[{agent_id}] turn {i + 1}: recorded completion_text is not the "
            f"raw decode of the recorded completion_token_ids"
        )
        # ...while the agent's reply must be the cleaned content the
        # production parser pipeline derives from those same IDs.
        expected_reply = expected_agent_content(MODEL_NAME, turn_record["completion_token_ids"])
        assert replies[i] == expected_reply, (
            f"[{agent_id}] turn {i + 1}: agent reply does not match the "
            f"content derived from the recorded token IDs "
            f"(reply={replies[i][:80]!r}..., expected={expected_reply[:80]!r}...)"
        )
        n_tokens = len(turn_record["completion_token_ids"])
        n_logprobs = len(turn_record["completion_logprobs"])
        assert n_tokens > 0, f"[{agent_id}] no token IDs recorded on turn {i + 1}"
        assert n_tokens == n_logprobs, (f"[{agent_id}] token/logprob length mismatch: {n_tokens} vs {n_logprobs}")

        # ---- strict TITO / delta-storage checks --------------------------
        # Turns store only what the request appended to the session's token
        # stream: turn 1 opens the conversation (full render), turn 2 must
        # be recorded as a pure delta on it.
        assert turn_record["new_conversation"] is (i == 0), (
            f"[{agent_id}] turn {i + 1}: new_conversation="
            f"{turn_record['new_conversation']!r}, expected {i == 0}"
        )
        prompt_delta = turn_record["prompt_token_ids"]
        assert len(prompt_delta) > 0, (f"[{agent_id}] turn {i + 1} recorded no prompt tokens")
        stream += prompt_delta + turn_record["completion_token_ids"]

    turn1, turn2 = record["turns"]

    # Turn 1 stores the full opening request; turn 2 stores only the delta:
    # the new user message — the assistant echo is not stored, being turn
    # 1's recorded completion (multi-turn check).
    assert [m["role"] for m in turn1["request_messages"]] == ["user"]
    assert [m["role"] for m in turn2["request_messages"]] == ["user"], (
        f"[{agent_id}] turn 2 message delta should be [new user message], "
        f"got roles {[m.get('role') for m in turn2['request_messages']]}"
    )
    assert turn2["request_messages"][0]["content"] == prompt_turn2

    # Turn 1's full render must match the local tokenizer's rendering of
    # the opening request.
    expected_first = as_token_ids(
        verifier.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_turn1}],
            add_generation_prompt=False,
            tokenize=True,
        ),
        what="turn-1 prompt",
    ) + _expected_generation_prompt(verifier)
    assert turn1["prompt_token_ids"] == expected_first, (
        f"[{agent_id}] turn 1 prompt mismatch: {len(turn1['prompt_token_ids'])} "
        f"tokens recorded vs {len(expected_first)} expected"
    )

    # Turn 2's recorded delta must equal the tokenization of just the new
    # user message + generation prompt (the trainer's reconstruction
    # scheme), preceded by whatever of the assistant turn ending the sampled
    # completion did not itself carry — the assistant echo contributes no
    # re-tokenized tokens.
    expected_delta = _expected_gap_delta(
        verifier, prompt_turn2, turn1["prompt_token_ids"] + turn1["completion_token_ids"],
    )
    assert turn2["prompt_token_ids"] == expected_delta, (
        f"[{agent_id}] inter-turn delta mismatch: "
        f"{len(turn2['prompt_token_ids'])} tokens recorded vs "
        f"{len(expected_delta)} expected"
    )

    logger.info(
        "[%s] session record verified (%d turns, strict TITO, %d-token stream)",
        agent_id, len(record["turns"]), len(stream),
    )
    return record


def dump_records(records: list[dict[str, Any]], subdir: str) -> None:
    """Dump each agent's SessionRecord to test/proxy/{subdir}/session_record-{i}.json."""
    dump_dir = Path(__file__).resolve().parent / "proxy" / subdir
    dump_dir.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(records):
        dump_path = dump_dir / f"session_record-{i}.json"
        dump_path.write_text(json.dumps(record, indent=4, ensure_ascii=False))
        logger.info("[agent-%d] SessionRecord dumped to %s", i, dump_path)


async def assert_bad_key_rejected(proxy_url: str, real_api_key: str) -> None:
    """Keyed-routing auth check: a wrong real key or a bare real key with no
    session_id appended must be rejected with 401 before any inference is
    attempted; a key naming no agent_id or an empty session_id must be
    refused by the parser."""
    bad_keys = [
        f"wrong_key{KEY_DELIMITER}trial_bad{KEY_DELIMITER}{DEFAULT_AGENT_ID}",  # wrong real key
        real_api_key,                                     # bare real key, no session_id
    ]
    for bad_key in bad_keys:
        client = AsyncOpenAI(base_url=f"{proxy_url}/v1", api_key=bad_key, max_retries=0)
        try:
            try:
                await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": "Say OK."}],
                    max_tokens=8,
                )
                raise AssertionError(f"keyed route accepted a bad api_key: {bad_key!r}")
            except openai.AuthenticationError:
                pass
        finally:
            await client.close()

    simple_key = "simplekey"
    async def unused_handler(*args):
        raise AssertionError("key parsing must not generate")

    parse = LLMProxyServer(api_key=simple_key, save_rollout_sessions=False,
                           completion_handler=unused_handler).parse_session_key
    assert parse(f"{simple_key}{KEY_DELIMITER}trial_bad") is None, "a key naming no agent_id must be refused"
    assert parse(f"{simple_key}{KEY_DELIMITER}") is None, "an empty session id must be refused"

    logger.info("Keyed routing correctly rejected %d malformed/unauthorized api_keys "
                "plus 2 structural key-parsing cases", len(bad_keys))


async def assert_tamper_rejected(
    proxy_url: str,
    http: httpx.AsyncClient,
    proxy_api_key: str,
    error_log_dir: str | Path,
) -> None:
    """TITO enforcement check: tampered history must be rejected with a 400
    (identically with either transport), never silently re-tokenized."""
    sid = "trial_tamper"
    client = AsyncOpenAI(
        base_url=f"{proxy_url}/v1",
        api_key=f"{proxy_api_key}{KEY_DELIMITER}{sid}{KEY_DELIMITER}{DEFAULT_AGENT_ID}",
        max_retries=0,
    )
    try:
        first = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say OK."}],
            temperature=0.6,
            max_tokens=64,
        )
        tampered = [
            {"role": "user", "content": "Say OK. (tampered)"},
            {"role": "assistant", "content": first.choices[0].message.content or ""},
            {"role": "user", "content": "Say OK again."},
        ]
        try:
            await client.chat.completions.create(
                model=MODEL_NAME,
                messages=tampered,
                temperature=0.6,
                max_tokens=64,
            )
            raise AssertionError("tampered history was accepted — TITO not enforced")
        except openai.BadRequestError as e:
            assert "modified" in str(e), f"unexpected rejection: {e}"
            logger.info("Tampered history correctly rejected: %s", e)
    finally:
        await client.close()

    # The caller supplies an isolated directory for this proxy instance.
    # A previous run's log cannot satisfy the assertion.
    log_files = list(Path(error_log_dir).glob(f"*/{sid}.log"))
    assert len(log_files) == 1, f"expected one current error log for session {sid}: {log_files}"
    log_content = log_files[0].read_text()
    assert "modified" in log_content, f"error log missing the rejection: {log_content!r}"
    logger.info("Session error log written: %s", log_files[0])

    await http.delete(f"{proxy_url}/sessions/{sid}")


async def run_gauntlet(
    proxy_url: str,
    http: httpx.AsyncClient,
    *,
    verifier: TokenStreamManager,
    max_tokens: int,
    dump_dir: str,
    error_log_dir: str | Path,
    proxy_api_key: str = PROXY_API_KEY,
    after_rollouts: Any = None,
    drop_check: Any = None,
) -> list[dict[str, Any]]:
    """Drive concurrent agents, verify exact tokens, then check API behavior.

    Both transports run this gauntlet. ``after_rollouts`` optionally checks
    placement; ``drop_check`` checks provider cleanup after session deletion.
    """
    records = await asyncio.gather(*(
        run_agent(proxy_url, http, *spec,
                  max_tokens=max_tokens, verifier=verifier,
                  proxy_api_key=proxy_api_key)
        for spec in AGENT_PROMPTS
    ))

    assert len(records) == NUM_AGENTS
    session_ids = {r["session_id"] for r in records}
    assert len(session_ids) == NUM_AGENTS, "sessions not isolated"
    if after_rollouts is not None:
        after_rollouts(records)

    dump_records(records, dump_dir)

    sid = records[0]["session_id"]
    await http.delete(f"{proxy_url}/sessions/{sid}")
    if drop_check is not None:
        drop_check(sid)

    await assert_tamper_rejected(proxy_url, http, proxy_api_key, error_log_dir)
    await assert_bad_key_rejected(proxy_url, proxy_api_key)
    await assert_model_validation(proxy_url, http, proxy_api_key)
    await assert_length_truncation(proxy_url, http, proxy_api_key)
    await assert_streaming_supported(proxy_url, http, proxy_api_key)

    return records
