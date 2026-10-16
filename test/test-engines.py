"""Offline tests of engine adapters and the shared generation pipeline.

A scripted provider exercises token integrity, retry caching, model pinning,
and tool parsing. Fake HTTP endpoints exercise the actual direct and Slime
providers, including payloads, context limits, and endpoint placement.
Run: python test/test-engines.py
"""

from __future__ import annotations
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxyserver.engines import (ABORT, LENGTH, SGLANG, STOP, VLLM,  # noqa: E402
                                 EngineAbort, EngineError, get_adapter)
from proxyserver.model_registry import UnknownModelError  # noqa: E402
from proxyserver.rollout_provider import BaseRolloutProvider, DirectRolloutProvider, SGLangRolloutProvider
from proxyserver.server import LLMProxyServer, make_completion_handler
from proxyserver.token_stream import GENERATION_PROMPT, TokenStreamError  # noqa: E402

MAX_TOKENS = 8

# Any model name from tokenization/mapping.json: the providers resolve the
# request's model through the real mapping, and the fake tokenizer is
# injected underneath via tokenizer_loader.
MODEL = "Qwen3.5-9B"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class TokenOutput:
    """Scripted engine output."""

    token_ids: list[int] = field(default_factory=list)
    log_probs: list[float] | None = None
    stop_reason: str | None = None


class FakeServer:
    """Records generation parameters and returns a scripted result."""

    def __init__(self, output: TokenOutput):
        self.output = output
        self.calls: list[dict] = []


class CannedProvider(BaseRolloutProvider):
    def __init__(self, server=None, **kwargs):
        super().__init__(**kwargs)
        self.server = server

    async def _call_engine(self, prompt_ids, sampling_params, session_id):
        self.server.calls.append(dict(sampling_params))
        output = self.server.output
        return output.token_ids, output.log_probs or [], output.stop_reason, {}


class FakeTokenizer:
    """Enough of a HF tokenizer for TokenStreamManager and decoding."""

    TEMPLATE_IDS = [1, 2, 3]

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=True):
        return list(self.TEMPLATE_IDS)

    def encode(self, text, add_special_tokens=False):
        # TokenStreamManager tokenizes its generation prompt through this
        # (see token_stream.GENERATION_PROMPT); a tokenizer without it fails
        # every build_prompt.
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(f"<{t}>" for t in token_ids)


#: The prompt the engines must receive: the template's render plus the
#: generation prompt TokenStreamManager appends itself (the template is
#: rendered with add_generation_prompt=False — see token_stream.GENERATION_PROMPT).
PROMPT_IDS = FakeTokenizer.TEMPLATE_IDS + [ord(ch) for ch in GENERATION_PROMPT]


@dataclass
class _ParsedCall:
    name: str
    arguments: str


class FakeToolParser:
    """Always reports one tool call, so we can watch finish_reason survive."""

    async def extract_tool_calls(self, token_ids):
        return "thinking out loud", [_ParsedCall(name="get_weather", arguments='{"city":"SF"}')]


def _fake_tokenizer_loader(tokenizer_path: str) -> FakeTokenizer:
    return FakeTokenizer()


def _tool_parser_kwargs(tool_parser) -> dict:
    """Provider kwargs installing a canned tool parser (parsers are built
    per tokenizer through the factory, using the parser class's ``NAME``
    from each model's profile)."""
    if tool_parser is None:
        return {}
    return {"tool_parser_factory": lambda name, tok: tool_parser}


def token_streams(provider):
    """The strict-TITO stream manager backing MODEL's tokenizer directory."""
    return provider.models.resolve(MODEL).token_streams


def make_provider(engine: str, output: TokenOutput, tool_parser=None) -> tuple[CannedProvider, FakeServer]:
    server = FakeServer(output)
    provider = CannedProvider(
        server=server,
        inference_engine=engine,
        tokenizer_loader=_fake_tokenizer_loader,
        **_tool_parser_kwargs(tool_parser),
    )
    return provider, server


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class FakeSlimeRouter:
    """Stands in for slime's sgl-router: replays one canned SGLang
    ``/generate`` response and records what the provider sent."""

    def __init__(self, response: dict):
        self.response = response
        self.requests: list[dict] = []
        self.headers: list[dict | None] = []

    async def post(self, path: str, json: dict | None = None, headers: dict | None = None) -> _FakeHTTPResponse:
        assert path == "/generate", f"unexpected router path: {path}"
        self.requests.append(json or {})
        # The session-affinity key the router uses under consistent_hashing.
        self.headers.append(headers)
        return _FakeHTTPResponse(self.response)


def make_slime_provider(response: dict, tool_parser=None) -> tuple[SGLangRolloutProvider, FakeSlimeRouter]:
    provider = SGLangRolloutProvider(
        router_url="http://fake-router:30000",
        tokenizer_loader=_fake_tokenizer_loader,
        **_tool_parser_kwargs(tool_parser),
    )
    router = FakeSlimeRouter(response)
    provider._http = router
    return provider, router


class FakeEngineHTTP:
    """Stands in for one engine endpoint behind :class:`DirectRolloutProvider`:
    replays one canned response and records what the provider sent."""

    def __init__(
        self, response: dict, model_id: str = MODEL,
        max_model_len: int | None = None, context_length: int | None = None,
    ):
        self.response = response
        self.model_id = model_id
        self.max_model_len = max_model_len
        self.context_length = context_length
        self.posts: list[tuple[str, dict]] = []
        self.model_fetches = 0
        self.server_info_fetches = 0

    async def post(self, path: str, json: dict | None = None, headers: dict | None = None) -> _FakeHTTPResponse:
        self.posts.append((path, json or {}))
        return _FakeHTTPResponse(self.response)

    async def get(self, path: str) -> _FakeHTTPResponse:
        if path == "/get_model_info":
            self.model_fetches += 1
            return _FakeHTTPResponse({"model_path": self.model_id})
        if path == "/get_server_info":
            self.server_info_fetches += 1
            info: dict = {}
            if self.context_length is not None:
                info["context_length"] = self.context_length
            return _FakeHTTPResponse(info)
        assert path == "/v1/models", f"unexpected GET {path}"
        self.model_fetches += 1
        served: dict = {"id": self.model_id}
        if self.max_model_len is not None:
            served["max_model_len"] = self.max_model_len
        return _FakeHTTPResponse({"data": [served]})


def make_direct_provider(
    engine: str, responses: list[dict], tool_parser=None,
    max_model_len: int | None = None, context_length: int | None = None,
) -> tuple[DirectRolloutProvider, list[FakeEngineHTTP]]:
    provider = DirectRolloutProvider(
        base_urls=[f"http://fake-engine-{i}:8000/v1" for i in range(len(responses))],
        inference_engine=engine,
        tokenizer_loader=_fake_tokenizer_loader,
        **_tool_parser_kwargs(tool_parser),
    )
    fakes = [
        FakeEngineHTTP(response, max_model_len=max_model_len, context_length=context_length)
        for response in responses
    ]
    provider._clients = fakes
    return provider, fakes


def _vllm_response(token_ids: list[int], finish_reason: str, logprobs: list[float] | None = None) -> dict:
    """A vLLM ``/v1/completions`` response with ``return_tokens_as_token_ids``."""
    if logprobs is None:
        logprobs = [-0.1] * len(token_ids)
    return {
        "choices": [{
            "finish_reason": finish_reason,
            "logprobs": {
                "tokens": [f"token_id:{tid}" for tid in token_ids],
                "token_logprobs": list(logprobs),
            },
        }],
    }


def _sglang_response(token_ids: list[int], finish_type: str, logprobs: list[float] | None = None) -> dict:
    """An SGLang ``/generate`` response the way the router returns it."""
    if logprobs is None:
        logprobs = [-0.1] * len(token_ids)
    return {
        "output_ids": list(token_ids),
        "meta_info": {
            "finish_reason": {"type": finish_type},
            "output_token_logprobs": [(lp, tid, "") for lp, tid in zip(logprobs, token_ids)],
        },
    }


async def generate(provider: CannedProvider, session_id: str = "s1"):
    return await provider.generate(
        messages=[{"role": "user", "content": "hi"}],
        session_id=session_id,
        model=MODEL,
        max_tokens=MAX_TOKENS,
    )


# ---------------------------------------------------------------------------
# 1. Engine selection
# ---------------------------------------------------------------------------


def test_engine_selection(monkeyenv) -> None:
    assert get_adapter("vllm").name == VLLM
    assert get_adapter("sglang").name == SGLANG
    assert get_adapter("  SGLang ").name == SGLANG, "engine name must be case/space insensitive"

    # Env fallback, then the "vllm" default.
    monkeyenv("INFERENCE_ENGINE", "sglang")
    assert get_adapter(None).name == SGLANG
    assert get_adapter("vllm").name == VLLM, "explicit argument must win over the env var"
    monkeyenv("INFERENCE_ENGINE", None)
    assert get_adapter(None).name == VLLM

    for bad in ("tgi", "vLLM2", ""):
        try:
            get_adapter(bad or "nope")
        except ValueError as e:
            assert "supported engines" in str(e)
        else:
            raise AssertionError(f"expected ValueError for engine {bad!r}")

    # A bad name must fail at construction, not on the first request.
    try:
        CannedProvider(server=None, inference_engine="tgi")
    except ValueError:
        pass
    else:
        raise AssertionError("provider accepted an unsupported inference_engine")
    print("  ok  engine selection: explicit > $INFERENCE_ENGINE > vllm; bad names rejected eagerly")


# ---------------------------------------------------------------------------
# 2. Shared finish-reason normalization
# ---------------------------------------------------------------------------


def test_stop_reason_normalization() -> None:
    for engine in (VLLM, SGLANG):
        adapter = get_adapter(engine)
        assert adapter.finish_reason("stop", num_tokens=8, max_tokens=8) == STOP
        assert adapter.finish_reason("length", num_tokens=8, max_tokens=8) == LENGTH
        assert adapter.finish_reason("length", num_tokens=3, max_tokens=8) == LENGTH
        assert adapter.finish_reason("abort", num_tokens=3, max_tokens=8) == ABORT
        assert adapter.finish_reason(None, num_tokens=3, max_tokens=8) == STOP
        assert adapter.finish_reason(None, num_tokens=8, max_tokens=8) == LENGTH
    print("  ok  native finish reasons are authoritative on both engines")


# ---------------------------------------------------------------------------
# 3. The provider surfaces a normalized finish_reason
# ---------------------------------------------------------------------------


async def test_provider_finish_reason() -> None:
    cases = [
        (VLLM, "stop", 3, STOP),
        (VLLM, "length", MAX_TOKENS, LENGTH),
        (SGLANG, "stop", 3, STOP),
        (SGLANG, "stop", MAX_TOKENS, STOP),
        (SGLANG, "length", 3, LENGTH),
    ]
    for engine, stop_reason, n_tokens, expected in cases:
        output = TokenOutput(token_ids=list(range(100, 100 + n_tokens)),
                             log_probs=[-0.1] * n_tokens, stop_reason=stop_reason)
        provider, _ = make_provider(engine, output)
        _, token_ids, log_probs, finish_reason, text, _ = await generate(provider)
        assert finish_reason == expected, f"{engine}/{stop_reason}/{n_tokens} -> {finish_reason}, want {expected}"
        assert len(token_ids) == n_tokens and len(log_probs) == n_tokens
        assert text == "".join(f"<{t}>" for t in token_ids)
    print(f"  ok  provider normalizes finish_reason across {len(cases)} engine/stop_reason cases")


# ---------------------------------------------------------------------------
# 4. Sampling params stay vLLM-styled until transport-specific translation
# ---------------------------------------------------------------------------


async def test_sampling_params_are_engine_neutral() -> None:
    for engine in (VLLM, SGLANG):
        output = TokenOutput(token_ids=[7], log_probs=[-0.5], stop_reason="stop")
        provider, server = make_provider(engine, output)
        await generate(provider)
        sent = server.calls[0]
        # All SGLang HTTP paths translate "max_tokens"/"logprobs" inside _sglang_generate.
        # The shared generation pipeline must keep the vLLM-style names.
        assert sent["max_tokens"] == MAX_TOKENS, sent
        assert sent["logprobs"] is True, sent
        assert "max_new_tokens" not in sent and "return_logprob" not in sent, sent
    print("  ok  sampling params identical for both engines before transport-specific translation")


# ---------------------------------------------------------------------------
# 5. Aborted / empty completions never enter the token stream
# ---------------------------------------------------------------------------


async def test_request_owned_sampling_params() -> None:
    for transport, engine in (("direct", VLLM), ("direct", SGLANG), ("slime", SGLANG)):
        response = _vllm_response([7], "stop") if engine == VLLM else _sglang_response([7], "stop")
        if transport == "slime":
            provider, fake = make_slime_provider(response)
        else:
            provider, (fake,) = make_direct_provider(engine, [response])
        for index, (seed, stop_ids) in enumerate(((None, None), (0, []), (42, [99]))):
            await provider.generate(
                messages=[{"role": "user", "content": "hi"}], session_id=f"sampling-{index}",
                model=MODEL, max_tokens=MAX_TOKENS, seed=seed, stop_token_ids=stop_ids,
            )
            body = fake.requests[-1] if transport == "slime" else fake.posts[-1][1]
            sent = body if engine == VLLM else body["sampling_params"]
            seed_key = "seed" if engine == VLLM else "sampling_seed"
            if seed is None:
                assert seed_key not in sent and "stop_token_ids" not in sent, sent
            else:
                assert sent[seed_key] == seed and sent["stop_token_ids"] == stop_ids, sent
            if engine == SGLANG:
                assert "seed" not in sent, sent
    print("  ok  request seeds and stop token IDs reach direct and Slime native payloads")


async def test_bad_completions_rejected() -> None:
    # Abort, reported in each engine's own vocabulary.
    for engine, stop_reason in ((VLLM, "abort"), (SGLANG, "abort")):
        provider, _ = make_provider(engine, TokenOutput(token_ids=[1, 2], log_probs=[-0.1, -0.2],
                                                        stop_reason=stop_reason))
        try:
            await generate(provider)
        except EngineAbort:
            pass
        else:
            raise AssertionError(f"{engine}: aborted completion was accepted")
        assert "s1" not in token_streams(provider)._sessions, f"{engine}: aborted turn polluted the token stream"

    # The SGLang HTTP transport empties token_ids when the
    # logprob payload does not
    # line up with the output ids; recording that would corrupt the trial.
    provider, _ = make_provider(SGLANG, TokenOutput(token_ids=[], log_probs=[], stop_reason="stop"))
    try:
        await generate(provider)
    except EngineError as e:
        assert not isinstance(e, EngineAbort)
    else:
        raise AssertionError("empty completion was accepted")
    assert "s1" not in token_streams(provider)._sessions, "empty turn polluted the token stream"

    # A healthy turn does commit, so the guards above are not vacuous.
    provider, _ = make_provider(SGLANG, TokenOutput(token_ids=[9], log_probs=[-0.1], stop_reason="stop"))
    await generate(provider)
    assert token_streams(provider)._sessions["s1"].stream == PROMPT_IDS + [9]
    print("  ok  aborted/empty completions raise and leave the token stream untouched")


async def test_invalid_values_preserve_record() -> None:
    provider, server = make_provider(VLLM, TokenOutput([7], [-0.1], "stop"))
    with TemporaryDirectory() as root:
        proxy = LLMProxyServer(
            api_key="test", key_delimiter="-", completion_handler=make_completion_handler(provider),
            session_dir=Path(root) / "sessions", error_log_dir=Path(root) / "logs",
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
            messages = [{"role": "user", "content": "hi"}]

            async def chat():
                return await client.post("/v1/chat/completions", headers={"Authorization": "Bearer test-trial-agent"},
                                         json={"model": MODEL, "messages": messages})

            first = await chat()
            assert first.status_code == 200, first.text
            record = (await client.get("/sessions/trial")).json()
            original = list(token_streams(provider)._sessions["trial-agent"].stream)
            snapshot = next((Path(root) / "sessions").rglob("trial.json"))
            saved = snapshot.read_text()
            messages += [{"role": "assistant", "content": first.json()["choices"][0]["message"]["content"]},
                         {"role": "user", "content": "continue"}]
            cases = [([value], [-0.1]) for value in (True, -1, 7.5, 7.0, "7", [7])]
            cases += [([7], [value]) for value in (float("nan"), float("inf"), -float("inf"),
                                                   True, "-0.1", None, [-0.1], 10 ** 1000)]
            for tokens, probs in cases:
                server.output = TokenOutput(tokens, probs, "stop")
                response = await chat()
                assert response.status_code == 500, (tokens, probs, response.text)
                fetched = await client.get("/sessions/trial")
                assert fetched.status_code == 200 and fetched.json() == record
                assert snapshot.read_text() == saved, "invalid output changed the disk record"
                assert token_streams(provider)._sessions["trial-agent"].stream == original

            server.output = TokenOutput([8], [0], "stop")
            recovered = await chat()
            assert recovered.status_code == 200, recovered.text
            turns = (await client.get("/sessions/trial")).json()["turns"]
            assert len(turns) == 2 and turns[-1]["completion_token_ids"] == [8]
            assert type(turns[-1]["completion_logprobs"][0]) is float
            saved_turns = json.loads(snapshot.read_text())["turns"]
            assert len(saved_turns) == 2 and saved_turns[-1]["completion_logprobs"] == [0.0]
            assert len(server.calls) == len(cases) + 2, "retry reused an invalid completion"
    print("  ok  invalid engine values leave the stream, session API and disk record intact; corrected retry succeeds")


# ---------------------------------------------------------------------------
# 6. Tool calls survive finish_reason unless the response was truncated
# ---------------------------------------------------------------------------


async def test_tool_calls_survive_finish_reason() -> None:
    async def finish_reason_of(engine: str, stop_reason: str, n_tokens: int) -> str:
        output = TokenOutput(token_ids=list(range(100, 100 + n_tokens)),
                             log_probs=[-0.1] * n_tokens, stop_reason=stop_reason)
        provider, _ = make_provider(engine, output, tool_parser=FakeToolParser())
        _, token_ids, _, engine_finish_reason, text, _ = await provider.generate(
            messages=[{"role": "user", "content": "weather?"}],
            session_id="tool-sess", model=MODEL, max_tokens=MAX_TOKENS,
        )
        parsed = await provider.parse_completion("tool-sess", token_ids, text, engine_finish_reason)
        assert parsed.tool_calls, "tool call vanished from the parsed result"
        replay = await provider.parse_completion("tool-sess", token_ids, text, engine_finish_reason)
        assert replay == parsed
        replay.tool_calls[0]["function"]["arguments"] = "mutated by caller"
        replay = await provider.parse_completion("tool-sess", token_ids, text, engine_finish_reason)
        assert replay == parsed, "caller mutated the cached parsed result"
        return parsed.finish_reason

    # The regression this guards: under SGLang, stop_reason == "stop" used to
    # overwrite the tool parser's "tool_calls" verdict, so agents that branch on
    # finish_reason == "tool_calls" silently stopped calling tools.
    assert await finish_reason_of(SGLANG, "stop", 3) == "tool_calls"
    assert await finish_reason_of(VLLM, "stop", 3) == "tool_calls"

    # Truncation does override it — a cut-off tool call is not a tool call.
    assert await finish_reason_of(SGLANG, "length", 3) == LENGTH
    assert await finish_reason_of(VLLM, "length", MAX_TOKENS) == LENGTH
    print("  ok  tool_calls survives on both engines; truncation still overrides it")


# ---------------------------------------------------------------------------
# 6b. A call-free completion still gets agent-clean content
# ---------------------------------------------------------------------------


async def test_call_free_completion_still_cleans_content() -> None:
    # Call-free completions must still clean generate()'s raw decode
    # (skip_special_tokens=False), keeping "<|im_end|>" and pre-filled
    # reasoning out of message.content. Without a factory the provider
    # builds MODEL's mapping-named parser (qwen3_coder) on the fake tokenizer;
    # the completion contains no <tool_call> block, so the parser must clean
    # the content and extract nothing.
    class SpecialTokenTokenizer(FakeTokenizer):
        def decode(self, token_ids, skip_special_tokens=False):
            text = "planning it out\n</think>\n\nvisible answer"
            return text if skip_special_tokens else text + "<|im_end|>"

    server = FakeServer(TokenOutput(token_ids=[9], log_probs=[-0.1], stop_reason="stop"))
    provider = CannedProvider(
        server=server,
        inference_engine=SGLANG,
        tokenizer_loader=lambda path: SpecialTokenTokenizer(),
    )
    _, token_ids, _, finish_reason, text, _ = await provider.generate(
        messages=[{"role": "user", "content": "hi"}],
        session_id="clean-sess", model=MODEL, max_tokens=MAX_TOKENS,
    )
    assert "<|im_end|>" in text, "generate() must keep the raw decode for the session record"

    parsed = await provider.parse_completion(
        "clean-sess", token_ids, text, finish_reason,
    )
    assert parsed.tool_calls is None and parsed.finish_reason == STOP
    assert parsed.content == "visible answer", f"agent-visible content not cleaned: {parsed.content!r}"
    assert parsed.reasoning_content == "planning it out"
    print("  ok  call-free completions still get special tokens and reasoning stripped from content")


# ---------------------------------------------------------------------------
# 7. Strict TITO is unaffected by the engine
# ---------------------------------------------------------------------------


async def test_tito_still_enforced() -> None:
    for engine in (VLLM, SGLANG):
        provider, _ = make_provider(engine, TokenOutput(token_ids=[9], log_probs=[-0.1], stop_reason="stop" if engine == VLLM else "stop"))
        try:
            await provider.generate(messages=[{"role": "assistant", "content": "injected"}],
                                    session_id="fresh", model=MODEL, max_tokens=MAX_TOKENS)
        except TokenStreamError:
            pass
        else:
            raise AssertionError(f"{engine}: assistant content on a fresh session was re-tokenized")
    print("  ok  strict TITO enforcement unchanged on both engines")


# ---------------------------------------------------------------------------
# 7b. Lost-response retries are re-served, not rejected
# ---------------------------------------------------------------------------


async def test_duplicate_request_reserved() -> None:
    """An exact repeat of the previous request — an agent (or its SDK's
    automatic retry) resending a turn whose response was generated and
    committed but lost in transit — must be re-served from the cached turn:
    same tokens, same logprobs, no second engine call.  That covers the
    session's *opening* request too, which carries no assistant message; only
    a **different** assistant-free request opens a fresh conversation.  An
    edited repeat is still a strict-TITO violation."""
    for engine, stop_reason in ((VLLM, "stop"), (SGLANG, "stop")):
        output = TokenOutput(token_ids=[5, 6, 7], log_probs=[-0.5, -0.6, -0.7], stop_reason=stop_reason)
        provider, server = make_provider(engine, output)
        await generate(provider)
        turn2 = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<5><6><7>"},
            {"role": "user", "content": "more"},
        ]
        second = await provider.generate(messages=turn2, session_id="s1", model=MODEL, max_tokens=MAX_TOKENS)
        engine_calls = len(server.calls)

        # The lost-response retry: byte-identical request -> cached turn.
        replay = await provider.generate(messages=turn2, session_id="s1", model=MODEL, max_tokens=MAX_TOKENS)
        assert replay == second, f"{engine}: re-served turn differs from the original"
        assert len(server.calls) == engine_calls, f"{engine}: duplicate request hit the engine again"

        # An edited repeat of the same length is still a violation.
        tampered = turn2[:-1] + [{"role": "user", "content": "more (edited)"}]
        try:
            await provider.generate(messages=tampered, session_id="s1", model=MODEL, max_tokens=MAX_TOKENS)
        except TokenStreamError:
            pass
        else:
            raise AssertionError(f"{engine}: edited same-length request was accepted")

        # The stream is intact: the next real turn still extends it.
        turn3 = turn2 + [
            {"role": "assistant", "content": "<5><6><7>"},
            {"role": "user", "content": "again"},
        ]
        await provider.generate(messages=turn3, session_id="s1", model=MODEL, max_tokens=MAX_TOKENS)
        assert len(server.calls) == engine_calls + 1

        # The opening request carries no assistant message, and its retry is
        # the most expensive turn to lose: a byte-identical repeat must be
        # re-served, not sampled a second time.
        provider, server = make_provider(engine, output)
        first = await generate(provider, session_id="x")
        retried = await generate(provider, session_id="x")
        assert retried == first, f"{engine}: re-served opening turn differs from the original"
        assert len(server.calls) == 1, (
            f"{engine}: verbatim retry of the opening request re-sampled it "
            f"({len(server.calls)} engine calls)"
        )

        # A *different* assistant-free request still opens a fresh
        # conversation on the same session, and does reach the engine.
        await provider.generate(
            messages=[{"role": "user", "content": "unrelated new task"}],
            session_id="x", model=MODEL, max_tokens=MAX_TOKENS,
        )
        assert len(server.calls) == 2, f"{engine}: a new conversation should reach the engine"
    print("  ok  lost-response retries re-served from the cached turn (no engine call),")
    print("      opening requests included; edited repeats still rejected, and a")
    print("      different assistant-free request still opens a fresh conversation")


class _ErrorHTTPResponse:
    """A response whose ``raise_for_status`` raises a real
    ``httpx.HTTPStatusError`` carrying the given status and body."""

    def __init__(self, status: int, detail: str):
        request = httpx.Request("POST", "http://fake-engine-0:8000/v1/completions")
        self._response = httpx.Response(status, request=request, text=detail)

    def raise_for_status(self) -> None:
        self._response.raise_for_status()

    def json(self) -> dict:
        raise AssertionError("body of an error response must not be parsed")


class ScriptedStatusEngineHTTP(FakeEngineHTTP):
    """FakeEngineHTTP whose posts fail with scripted HTTP statuses first
    (consumed one per post), then succeed with the canned response."""

    def __init__(self, response: dict, statuses: list[int], detail: str = ""):
        super().__init__(response)
        self.statuses = statuses
        self.detail = detail

    async def post(self, path: str, json: dict | None = None, headers: dict | None = None):
        self.posts.append((path, json or {}))
        if self.statuses:
            return _ErrorHTTPResponse(self.statuses.pop(0), self.detail)
        return _FakeHTTPResponse(self.response)


async def test_rejected_request_failed_fast() -> None:
    """The mirror of the lost-response re-serve, for failure: a request the
    engine rejects with a deterministic 4xx gets MAX_REJECTED_GENERATIONS
    live attempts; further exact repeats are failed fast with the stored
    rejection, no engine call.  Without this, a rejection that takes longer
    to arrive than the agent's own timeout is never seen by the agent — it
    retries the identical request forever, each retry burning a full doomed
    generation (vLLM emitting NaN logprobs on one long-context stream, 400
    "Out of range float values are not JSON compliant: nan", wedged four
    rollouts exactly so).  5xx and transient 4xx never poison the turn, and
    the session's next successful turn clears the verdict."""
    from proxyserver.rollout_provider import MAX_REJECTED_GENERATIONS

    nan_detail = ('{"error":{"message":"Out of range float values are not '
                  'JSON compliant: nan","type":"BadRequestError","param":null,"code":400}}')

    async def expect_status(coro, status: int) -> httpx.HTTPStatusError:
        try:
            await coro
        except httpx.HTTPStatusError as e:
            assert e.response.status_code == status, \
                f"expected HTTP {status}, got {e.response.status_code}"
            return e
        raise AssertionError(f"expected HTTP {status} to propagate to the caller")

    def make_provider_with_statuses(statuses: list[int]):
        provider = DirectRolloutProvider(
            base_urls=["http://fake-engine-0:8000/v1"],
            inference_engine=VLLM,
            tokenizer_loader=_fake_tokenizer_loader,
        )
        fake = ScriptedStatusEngineHTTP(_vllm_response([5, 6, 7], "stop"), statuses, detail=nan_detail)
        provider._clients = [fake]
        return provider, fake

    # Deterministic 400: two live attempts, then fail-fast.
    provider, fake = make_provider_with_statuses([400] * MAX_REJECTED_GENERATIONS)
    await expect_status(generate(provider, session_id="s1"), 400)
    assert len(fake.posts) == 1
    await expect_status(generate(provider, session_id="s1"), 400)
    assert len(fake.posts) == MAX_REJECTED_GENERATIONS, "the first retry deserves a live attempt"
    for _ in range(2):  # every further exact repeat: same verdict, no generation
        err = await expect_status(generate(provider, session_id="s1"), 400)
        assert len(fake.posts) == MAX_REJECTED_GENERATIONS, \
            "an exact repeat of a rejected request burned another generation"
        assert "JSON compliant" in err.response.text, "fail-fast lost the engine's rejection detail"

    # A *different* request is no repeat: it generates live, and its success
    # clears the session's verdict.
    ok = await provider.generate(
        messages=[{"role": "user", "content": "another task"}],
        session_id="s1", model=MODEL, max_tokens=MAX_TOKENS,
    )
    assert ok[1] == [5, 6, 7], "the session's next request must generate normally"
    assert "s1" not in provider._rejected_turns, "a successful turn must clear the verdict"

    # 5xx and transient 4xx are not verdicts on the request: identical
    # retries keep reaching the engine, and the one after them succeeds.
    for status in (502, 429):
        provider, fake = make_provider_with_statuses([status] * 3)
        for attempt in range(3):
            await expect_status(generate(provider, session_id="s2"), status)
            assert len(fake.posts) == attempt + 1, \
                f"HTTP {status} must not poison the turn (attempt {attempt + 1} skipped the engine)"
        ok = await generate(provider, session_id="s2")
        assert ok[1] == [5, 6, 7]

    # release_session drops the verdict with the rest of the session state.
    provider, fake = make_provider_with_statuses([400] * MAX_REJECTED_GENERATIONS)
    for _ in range(MAX_REJECTED_GENERATIONS):
        await expect_status(generate(provider, session_id="s3"), 400)
    assert "s3" in provider._rejected_turns
    provider.release_session("s3")
    assert "s3" not in provider._rejected_turns
    print("  ok  a deterministic engine 4xx fails exact retries fast after "
          f"{MAX_REJECTED_GENERATIONS} live attempts")
    print("      (rejection re-raised, no re-generation); 5xx/transient 4xx never poison")
    print("      the turn, and a successful or released session clears the verdict")


# ---------------------------------------------------------------------------
# 8. The slime transport: same contract through sgl-router HTTP
# ---------------------------------------------------------------------------


async def test_slime_provider() -> None:
    import os

    # 1. Wire format: token IDs in, SGLang-native sampling params — the
    #    provider itself does the max_tokens/logprobs translation, since
    #    slime's router is transparent.
    provider, router = make_slime_provider(_sglang_response([7, 8], "stop", [-0.1, -0.2]))
    _, token_ids, log_probs, finish_reason, _, _ = await generate(provider)
    sent = router.requests[0]
    assert sent["input_ids"] == PROMPT_IDS, sent
    assert sent["return_logprob"] is True, sent
    params = sent["sampling_params"]
    assert params["max_new_tokens"] == MAX_TOKENS, params
    assert "max_tokens" not in params and "logprobs" not in params, params
    assert token_ids == [7, 8] and log_probs == [-0.1, -0.2] and finish_reason == STOP

    # 2. stop_reason is authoritative (the sglang adapter), exactly as when
    #    direct SGLang: EOS on the last permitted token
    #    stays a "stop", truncation arrives as "length" verbatim.
    provider, _ = make_slime_provider(_sglang_response(list(range(100, 100 + MAX_TOKENS)), "stop"))
    assert (await generate(provider))[3] == STOP
    provider, _ = make_slime_provider(_sglang_response([100, 101, 102], "length"))
    assert (await generate(provider))[3] == LENGTH
    provider, _ = make_slime_provider(_sglang_response([100], "abort"))
    try:
        await generate(provider)
    except EngineAbort:
        pass
    else:
        raise AssertionError("slime: aborted completion was accepted")

    # 3. A misaligned logprob payload is refused before it can poison the
    #    session's token stream.
    provider, _ = make_slime_provider({
        "output_ids": [5, 6],
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [(-0.1, 5, "")],  # one entry short
        },
    })
    try:
        await generate(provider)
    except EngineError as e:
        assert not isinstance(e, EngineAbort)
    else:
        raise AssertionError("slime: misaligned logprob payload was accepted")
    assert "s1" not in token_streams(provider)._sessions, "mismatched turn polluted the token stream"

    # 4. slime is SGLang-only, and never falls back to $INFERENCE_ENGINE —
    #    a vLLM-flavored environment must not silently misconfigure it.
    saved = os.environ.get("INFERENCE_ENGINE")
    os.environ["INFERENCE_ENGINE"] = "vllm"
    try:
        provider = SGLangRolloutProvider("http://fake-router:30000")
        assert provider.engine.name == SGLANG
        try:
            SGLangRolloutProvider("http://fake-router:30000", inference_engine="vllm")
        except ValueError as e:
            assert "sglang" in str(e)
        else:
            raise AssertionError("SGLangRolloutProvider accepted inference_engine='vllm'")
    finally:
        if saved is None:
            os.environ.pop("INFERENCE_ENGINE", None)
        else:
            os.environ["INFERENCE_ENGINE"] = saved

    print("  ok  slime transport: SGLang-native wire format, authoritative stop_reason,")
    print("      misaligned logprobs refused, SGLang-only enforced")


# ---------------------------------------------------------------------------
# 9. The direct transport: the engines' own HTTP APIs, both backends
# ---------------------------------------------------------------------------


async def test_http_alignment_failure_preserves_stream() -> None:
    # The old vLLM filters dropped an invalid token and a null logprob at
    # different positions, yielding equal lengths but the wrong pairing.
    vllm_cases = [
        (["token_id:5", "text", "token_id:6"], [-0.1, -0.2, None]),
        (["token_id:5", "text", "token_id:6"], [-0.1, None, -0.3]),
        (["token_id:5", "token_id:6"], [-0.1]),
        (["token_id:5", "token_id:bad"], [-0.1, -0.2]),
        (["token_id:5", None], [-0.1, -0.2]),
        (["token_id:5", "token_id:6"], [-0.1, None]),
        (["token_id:5", "token_id:6"], [-0.1, "-0.2"]),
        (["token_id:5", "token_id:6"], [-0.1, True]),
        (["token_id:5", "token_id:-1"], [-0.1, -0.2]),
    ]
    cases = []
    for tokens, probs in vllm_cases:
        payload = _vllm_response([5, 6], "stop")
        payload["choices"][0]["logprobs"] = {"tokens": tokens, "token_logprobs": probs}
        cases.append((VLLM, payload))
    for ids in ([6, 5], [5, 7]):
        payload = _sglang_response([5, 6], "stop", [-0.1, -0.2])
        payload["output_ids"] = ids
        cases.append((SGLANG, payload))
        cases.append(("slime", payload))
    for value in (float("nan"), float("inf"), -float("inf"), 10 ** 1000):
        cases.append((VLLM, _vllm_response([5, 6], "stop", [-0.1, value])))
        for engine in (SGLANG, "slime"):
            cases.append((engine, _sglang_response([5, 6], "stop", [-0.1, value])))

    for engine, bad_payload in cases:
        good_payload = (_vllm_response if engine == VLLM else _sglang_response)([7, 8], "stop", [-0.1, -0.2])
        if engine == "slime":
            provider, fake = make_slime_provider(good_payload)
        else:
            provider, (fake,) = make_direct_provider(engine, [good_payload])
        first = await generate(provider)
        streams = token_streams(provider)
        committed = streams.committed_turn("s1", first[1])
        messages = [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": first[4]},
                    {"role": "user", "content": "continue"}]
        fake.response = bad_payload
        try:
            await provider.generate(messages, "s1", model=MODEL, max_tokens=MAX_TOKENS)
        except EngineError:
            pass
        else:
            raise AssertionError(f"{engine}: malformed token/logprob alignment was accepted: {bad_payload}")
        assert streams.committed_turn("s1", first[1]) is committed, "invalid output replaced the committed turn"
        assert streams.cached_turn("s1", [{"role": "user", "content": "hi"}])[:3] == first[:3]

        # A corrected engine response can retry the failed continuation and
        # still extends the original sampled stream exactly.
        fake.response = good_payload
        recovered = await provider.generate(messages, "s1", model=MODEL, max_tokens=MAX_TOKENS)
        prefix = first[0] + first[1]
        assert recovered[0][:len(prefix)] == prefix
        assert recovered[1:3] == ([7, 8], [-0.1, -0.2])
    print("  ok  HTTP transports reject invalid token/logprob pairs without committing; corrected retries preserve TITO")


async def test_direct_model_compatibility() -> None:
    for engine in (VLLM, SGLANG):
        await _check_direct_model_compatibility(engine)


async def _check_direct_model_compatibility(engine: str) -> None:
    response = _vllm_response([7], "stop") if engine == VLLM else _sglang_response([7], "stop")
    for served in ("Llama-3.1-8B-Instruct", "/weights/GLM-5.3-Flash", "Qwen3.6-27B", "checkpoint-123"):
        provider, (fake,) = make_direct_provider(engine, [response])
        fake.model_id = served
        for _ in range(2):  # cached discovery must never bypass the check
            try:
                await generate(provider)
            except EngineError as e:
                assert served in str(e), str(e)
            else:
                raise AssertionError(f"incompatible backend {served!r} accepted")
            assert not fake.posts, "incompatible token IDs reached the engine"
            assert not provider.models.has_session("s1"), "failed validation committed tokens"
        assert fake.model_fetches == 1

    for served in (MODEL, f"/weights/{MODEL}", f"Qwen/{MODEL}", "rl-exp-model", "Qwen3.5-35B-A3B"):
        provider, (fake,) = make_direct_provider(engine, [response])
        fake.model_id = served
        assert (await generate(provider))[1] == [7]
        if engine == VLLM:
            assert fake.posts[0][1]["model"] == served, "the backend's actual name must stay on the wire"
        # A second stream reuses discovery but must still be checked against
        # its own model pin, not the first stream's successful validation.
        try:
            await provider.generate(
                messages=[{"role": "user", "content": "hi"}],
                session_id="incompatible_claim", model="Llama-3.1-8B-Instruct",
            )
        except EngineError:
            pass
        else:
            raise AssertionError("cached discovery bypassed the new stream's model check")
        assert len(fake.posts) == 1 and fake.model_fetches == 1

    # Every selected endpoint is checked, even when another endpoint passed.
    provider, fakes = make_direct_provider(engine, [response] * 2)
    fakes[1].model_id = "Llama-3.1-8B-Instruct"
    await generate(provider, "good")
    proxy = LLMProxyServer(
        host="127.0.0.1", port=0, api_key="test",
        completion_handler=make_completion_handler(provider), save_rollout_sessions=False,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions", headers={"Authorization": "Bearer test-trial-agent"},
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 500, response.text
        record = await client.get("/sessions/trial")
        assert record.status_code == 404 or not record.json()["turns"], record.text
    assert len(fakes[0].posts) == 1 and not fakes[1].posts
    assert not provider.models.has_session("trial-agent")
    print(f"  ok  direct {engine} rejects mismatched/unknown profiles before generation; mapped aliases and paths work")


async def test_direct_sglang_model_discovery() -> None:
    class ModelInfoHTTP(FakeEngineHTTP):
        def __init__(self, info):
            super().__init__(_sglang_response([7], "stop"))
            self.info = info
            self.fail = False

        async def get(self, path):
            if path != "/get_model_info":
                return await super().get(path)
            self.model_fetches += 1
            if self.fail:
                raise httpx.ConnectError("model discovery unavailable")
            return _FakeHTTPResponse(self.info)

    # Served aliases take precedence over checkpoint paths, just as the
    # explicit model IDs returned by vLLM do. Older servers omit the alias.
    for info, accepted in (
        ({"served_model_name": "rl-exp-model", "model_path": "/weights/checkpoint-123"}, True),
        ({"model_path": f"/weights/{MODEL}"}, True),
        ({"served_model_name": None, "model_path": MODEL}, True),
        ({"served_model_name": "Llama-3.1-8B-Instruct", "model_path": MODEL}, False),
        ({"served_model_name": "unknown-alias", "model_path": MODEL}, False),
        ({}, False),
        ({"model_path": " "}, False),
        ({"model_path": 123}, False),
        ({"served_model_name": [MODEL], "model_path": MODEL}, False),
        ([], False),
        (None, False),
    ):
        provider, _ = make_direct_provider(SGLANG, [_sglang_response([7], "stop")])
        fake = ModelInfoHTTP(info)
        provider._clients = [fake]
        try:
            result = await generate(provider)
        except EngineError:
            assert not accepted, info
            assert not fake.posts and not provider.models.has_session("s1")
        else:
            assert accepted and result[1] == [7], info

    # Neither a transport failure nor a malformed response may cache an
    # unverified identity or prevent discovery recovering on the next call.
    provider, _ = make_direct_provider(SGLANG, [_sglang_response([7], "stop")])
    fake = ModelInfoHTTP({})
    provider._clients = [fake]
    fake.fail = True
    for failure in (httpx.ConnectError, EngineError):
        try:
            await generate(provider)
        except failure:
            pass
        else:
            raise AssertionError("invalid model discovery allowed generation")
        assert not fake.posts and provider._model_ids == [None]
        fake.fail = False
    fake.info = {"model_path": MODEL}
    assert (await generate(provider))[1] == [7]
    assert fake.model_fetches == 3 and len(fake.posts) == 1
    print("  ok  direct SGLang model discovery validates metadata and recovers after failed discovery")


async def test_direct_provider() -> None:
    # 1. vLLM wire format: token-ID prompt to /v1/completions with
    #    return_tokens_as_token_ids, logprobs=0 (vLLM's own dialect), and the
    #    model id discovered from /v1/models once and cached.
    provider, (fake,) = make_direct_provider(VLLM, [_vllm_response([7, 8], "stop", [-0.1, -0.2])])
    _, token_ids, log_probs, finish_reason, _, _ = await generate(provider)
    path, sent = fake.posts[0]
    assert path == "/v1/completions", path
    assert sent["prompt"] == PROMPT_IDS, sent
    assert sent["return_tokens_as_token_ids"] is True, sent
    assert sent["logprobs"] == 0, sent
    assert sent["max_tokens"] == MAX_TOKENS, sent
    assert sent["model"] == MODEL, sent
    assert token_ids == [7, 8] and log_probs == [-0.1, -0.2] and finish_reason == STOP
    await provider.generate(
        messages=[{"role": "user", "content": "hi"},
                  {"role": "assistant", "content": "<7><8>"},
                  {"role": "user", "content": "again"}],
        session_id="s1", model=MODEL, max_tokens=MAX_TOKENS,
    )
    assert fake.model_fetches == 1, "served-model id must be fetched once and cached"

    # 2. vLLM's native "length" and "stop" reasons are authoritative, including
    #    a stop on the final permitted token. "abort" is refused before commit.
    provider, _ = make_direct_provider(VLLM, [_vllm_response([100, 101, 102], "length")])
    assert (await generate(provider))[3] == LENGTH
    provider, _ = make_direct_provider(VLLM, [_vllm_response(list(range(100, 100 + MAX_TOKENS)), "stop")])
    assert (await generate(provider))[3] == STOP, "EOS on the final token remains a natural stop"
    provider, _ = make_direct_provider(VLLM, [_vllm_response([100], "abort")])
    try:
        await generate(provider)
    except EngineAbort:
        pass
    else:
        raise AssertionError("direct/vllm: aborted completion was accepted")
    assert "s1" not in token_streams(provider)._sessions, "aborted turn polluted the token stream"

    # 2b. The context clamp changes the request cap; native length is preserved.
    budget = 3
    provider, (fake,) = make_direct_provider(
        VLLM,
        [_vllm_response(list(range(200, 200 + budget)), "length")],
        max_model_len=len(PROMPT_IDS) + budget,
    )
    _, token_ids, _, finish_reason, _, _ = await generate(provider)   # asks MAX_TOKENS
    _, sent = fake.posts[0]
    assert sent["max_tokens"] == budget, f"max_tokens must be clamped to the window: {sent}"
    assert len(token_ids) == budget, token_ids
    assert finish_reason == LENGTH, "a turn truncated by the context clamp must report 'length', not 'stop'"

    #     ...while an un-clamped turn that stops early still reports "stop".
    provider, (fake,) = make_direct_provider(VLLM, [_vllm_response([200, 201], "stop")], max_model_len=len(PROMPT_IDS) + 1000)
    assert (await generate(provider))[3] == STOP, "an unclamped short completion is a natural stop"
    assert fake.posts[0][1]["max_tokens"] == MAX_TOKENS, "a window with room to spare must not clamp"

    #     A prompt that alone fills the window cannot generate at all.
    provider, _ = make_direct_provider(
        VLLM, [_vllm_response([200], "stop")], max_model_len=len(PROMPT_IDS),
    )
    try:
        await generate(provider)
    except EngineError:
        pass
    else:
        raise AssertionError("direct/vllm: a prompt filling the context window was accepted")

    # 3. SGLang direct: the same native /generate dialect as the slime
    #    transport, stop_reason authoritative, misaligned logprobs refused.
    provider, (fake,) = make_direct_provider(SGLANG, [_sglang_response([7, 8], "stop", [-0.1, -0.2])])
    _, token_ids, log_probs, finish_reason, _, _ = await generate(provider)
    path, sent = fake.posts[0]
    assert path == "/generate", path
    assert sent["input_ids"] == PROMPT_IDS, sent
    assert sent["return_logprob"] is True, sent
    assert sent["sampling_params"]["max_new_tokens"] == MAX_TOKENS, sent
    assert token_ids == [7, 8] and log_probs == [-0.1, -0.2] and finish_reason == STOP
    provider, _ = make_direct_provider(SGLANG, [_sglang_response(list(range(100, 100 + MAX_TOKENS)), "stop")])
    assert (await generate(provider))[3] == STOP, "EOS on the last permitted token must stay a 'stop' under SGLang"
    provider, _ = make_direct_provider(SGLANG, [{
        "output_ids": [5, 6],
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [(-0.1, 5, "")],  # one entry short
        },
    }])
    try:
        await generate(provider)
    except EngineError as e:
        assert not isinstance(e, EngineAbort)
    else:
        raise AssertionError("direct/sglang: misaligned logprob payload was accepted")

    # 3b. The same context-window clamp as vLLM (2b), the window discovered
    #     from /get_server_info instead: SGLang admission-checks
    #     prompt + max_new_tokens <= context_len, so an un-clamped request is
    #     rejected up-front even when room to generate remains. Both engines'
    #     native finish reasons are authoritative; this fixture reports
    #     finish_reason='length' for a truncated turn at the clamped budget.
    budget = 3
    provider, (fake,) = make_direct_provider(
        SGLANG,
        [_sglang_response(list(range(200, 200 + budget)), "length")],
        context_length=len(PROMPT_IDS) + budget,
    )
    _, token_ids, _, finish_reason, _, _ = await generate(provider)   # asks MAX_TOKENS
    _, sent = fake.posts[0]
    assert sent["sampling_params"]["max_new_tokens"] == budget, f"max_new_tokens must be clamped to the window: {sent}"
    assert finish_reason == LENGTH

    # Discovery shapes, caching, failures and full-window refusal are shared
    # across direct/Slime in test-sglang-transport.py. Keep the provider's
    # normalized length result above and its per-endpoint isolation below.

    #     The window is per endpoint: two endpoints, two windows.
    provider, fakes = make_direct_provider(
        SGLANG,
        [_sglang_response(list(range(200, 200 + budget)), "length"),
         _sglang_response([7, 8], "stop")],
    )
    fakes[0].context_length = len(PROMPT_IDS) + budget
    fakes[1].context_length = len(PROMPT_IDS) + 1000
    await generate(provider, session_id="sess_small")   # round-robin -> endpoint 0
    await generate(provider, session_id="sess_big")     # -> endpoint 1
    assert fakes[0].posts[0][1]["sampling_params"]["max_new_tokens"] == budget
    assert fakes[1].posts[0][1]["sampling_params"]["max_new_tokens"] == MAX_TOKENS

    # 4. Endpoint routing: sessions bind round-robin on first use and stay
    #    sticky (prefix cache); deleting a session frees the binding.
    responses = [_vllm_response([7], "stop"), _vllm_response([8], "stop")]
    provider, fakes = make_direct_provider(VLLM, responses)
    await generate(provider, session_id="sess_a")
    await generate(provider, session_id="sess_b")
    assert provider._session_endpoints == {"sess_a": 0, "sess_b": 1}
    await provider.generate(
        messages=[{"role": "user", "content": "hi"},
                  {"role": "assistant", "content": "<7>"},
                  {"role": "user", "content": "again"}],
        session_id="sess_a", model=MODEL, max_tokens=MAX_TOKENS,
    )
    assert len(fakes[0].posts) == 2 and len(fakes[1].posts) == 1, "session sess_a was not sticky to its endpoint"
    provider.release_session("sess_a")
    assert "sess_a" not in provider._session_endpoints
    assert not provider.models.has_session("sess_a")

    # 5. Endpoint validation fails at construction, not on the first request:
    #    a single api_key broadcasts; mismatched lists, empty/blank/
    #    whitespace-only base URLs are all rejected eagerly.
    provider = DirectRolloutProvider(
        ["http://a:8000/v1", " http://b:8000/v1/ "],
        inference_engine=VLLM, api_keys=["shared-key"],
    )
    assert len(provider._clients) == 2
    assert provider.base_urls == ["http://a:8000", "http://b:8000"], provider.base_urls
    for bad_urls in ([], [""], ["   "], "", ["http://a:8000/v1", ""]):
        try:
            DirectRolloutProvider(bad_urls, inference_engine=VLLM)
        except ValueError:
            pass
        else:
            raise AssertionError(f"base_urls={bad_urls!r} was accepted")
    try:
        DirectRolloutProvider(
            ["http://a:8000/v1", "http://b:8000/v1", "http://c:8000/v1"],
            inference_engine=VLLM, api_keys=["k1", "k2"],
        )
    except ValueError as e:
        assert "api_keys" in str(e)
    else:
        raise AssertionError("mismatched api_keys list was accepted")

    print("  ok  direct transport: vLLM /v1/completions and SGLang /generate wire formats,")
    print("      context-window clamp on both engines (/v1/models and /get_server_info),")
    print("      raw stop_reason vocabularies normalized, sticky endpoint bindings,")
    print("      misaligned logprobs refused, endpoint wiring validated eagerly")


# ---------------------------------------------------------------------------
# 10. Model resolution: request-driven tokenizer selection, pinned per session
# ---------------------------------------------------------------------------


async def test_model_resolution() -> None:
    loads: list[str] = []

    def counting_loader(tokenizer_path: str) -> FakeTokenizer:
        loads.append(tokenizer_path)
        return FakeTokenizer()

    server = FakeServer(TokenOutput(token_ids=[9], log_probs=[-0.1], stop_reason="stop"))
    provider = CannedProvider(
        server=server,
        inference_engine=SGLANG,
        tokenizer_loader=counting_loader,
    )

    async def gen(session_id, model=None, messages=None):
        return await provider.generate(
            messages=messages or [{"role": "user", "content": "hi"}],
            session_id=session_id, model=model, max_tokens=MAX_TOKENS,
        )

    # 1. A model with no tokenizer in the mapping is rejected before anything
    #    reaches the engine or the token stream.
    try:
        await gen("s1", model="not-a-model")
    except UnknownModelError as e:
        assert "not-a-model" in str(e)
    else:
        raise AssertionError("unknown model was accepted")
    assert not provider.models.has_session("s1")
    assert not server.calls, "unknown model still reached the engine"

    # 2. A session's first request must name a model.
    try:
        await gen("s1")
    except TokenStreamError as e:
        assert "model" in str(e)
    else:
        raise AssertionError("first request without a model was accepted")

    # 3. The first request pins the session; later turns may omit the model
    #    but cannot name another (switching tokenizers mid-stream).
    await gen("s1", model="Qwen3.5-9B")
    followup = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "<9>"},
                {"role": "user", "content": "again"}]
    try:
        await gen("s1", model="Qwen3.5-27B", messages=followup)
    except TokenStreamError as e:
        assert "pinned" in str(e)
    else:
        raise AssertionError("mid-session model switch was accepted")
    await gen("s1", messages=followup)  # omitting the model reuses the pin

    # 4. Every model name of one tokenizer directory shares one loaded
    #    tokenizer — the mapping is resolved per name, the load happens once.
    await gen("s2", model="Qwen3.5-27B")
    assert loads and len(loads) == 1, f"tokenizer loaded {len(loads)} times, expected once"

    # 5. release_session drops the pin with the stream; a fresh request for
    #    the same id must name a model again.
    provider.release_session("s1")
    assert not provider.models.has_session("s1")
    assert "s1" not in provider._session_models

    print("  ok  model resolution: unknown models rejected, sessions pinned to their")
    print("      first model, mid-session switches refused, tokenizers shared per")
    print("      mapping directory, pins released with the session")


# ---------------------------------------------------------------------------


async def test_release_mid_flight_skips_commit() -> None:
    """Direct provider callers can release a session during generation.

    The turn must return without committing or re-creating the released
    stream, which would otherwise occupy an LRU slot until eviction.
    HTTP DELETE uses a separate lifecycle: cancel and drain active work,
    then release the streams.
    """
    provider, _server = make_provider(VLLM, TokenOutput(token_ids=[70, 71], log_probs=[-0.1, -0.2], stop_reason="stop"))

    real_call = provider._call_engine

    async def call_and_release(prompt_ids, sampling_params, session_id):
        output = await real_call(prompt_ids, sampling_params, session_id)
        # Simulate a direct caller releasing the stream before commit.
        provider.release_session(session_id)
        return output

    provider._call_engine = call_and_release
    _, token_ids, _, _, _, _ = await generate(provider, session_id="doomed")
    assert token_ids == [70, 71], "released-mid-flight turn did not return to the caller"
    assert "doomed" not in token_streams(provider)._sessions, \
        "commit re-created the token stream of a released session"
    assert "doomed" not in provider._session_models, \
        "the released session's model pin came back"

    # A session NOT released mid-flight still commits through the same path.
    provider._call_engine = real_call
    await generate(provider, session_id="alive")
    assert "alive" in token_streams(provider)._sessions, \
        "an unreleased session no longer commits its stream"
    print("  ok  a session released mid-generation returns its turn without re-creating the token stream")


def main() -> None:
    import os

    saved: dict[str, str | None] = {}

    def monkeyenv(key: str, value: str | None) -> None:
        saved.setdefault(key, os.environ.get(key))
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    monkeyenv("INFERENCE_ENGINE", None)
    print("engine routing tests")
    try:
        test_engine_selection(monkeyenv)
        test_stop_reason_normalization()
        asyncio.run(test_provider_finish_reason())
        asyncio.run(test_sampling_params_are_engine_neutral())
        asyncio.run(test_request_owned_sampling_params())
        asyncio.run(test_bad_completions_rejected())
        asyncio.run(test_invalid_values_preserve_record())
        asyncio.run(test_tool_calls_survive_finish_reason())
        asyncio.run(test_call_free_completion_still_cleans_content())
        asyncio.run(test_tito_still_enforced())
        asyncio.run(test_duplicate_request_reserved())
        asyncio.run(test_rejected_request_failed_fast())
        asyncio.run(test_release_mid_flight_skips_commit())
        asyncio.run(test_slime_provider())
        asyncio.run(test_direct_provider())
        asyncio.run(test_direct_model_compatibility())
        asyncio.run(test_direct_sglang_model_discovery())
        asyncio.run(test_http_alignment_failure_preserves_stream())
        asyncio.run(test_model_resolution())
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("\n" + "=" * 70)
    print("PASS: inference_engine routing — vllm and sglang stop_reason vocabularies")
    print("      normalized, tool calls preserved, aborted/empty completions rejected,")
    print("      sampling params translated by the HTTP transports, TITO intact,")
    print("      and the slime (sgl-router HTTP) and direct (engine HTTP) transports")
    print("      honor the same contract.")
    print("=" * 70)


if __name__ == "__main__":
    main()
