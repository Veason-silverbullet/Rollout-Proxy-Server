"""Offline test of the SGLang ``/generate`` transport (``_sglang_generate``).

Three behaviors that only this layer can get wrong, none of them visible to
the strict-TITO tests above it (they drive ``ScriptedProvider``, which replaces
the transport):

1. **Recovering the sampled token ids.**  SGLang returns ``output_ids`` only
   when the engine was started tokens-only (``--skip-tokenizer-init``).  With
   the tokenizer initialized it returns decoded ``text`` instead and the ids
   survive only inside the logprob triples.  Under the slime transport the
   *training framework* owns the engine launch and does not pass that flag, so
   a proxy that reads only ``output_ids`` fails on the first turn of every
   rollout — with a logprob/token length mismatch, which reads like corruption
   rather than a launch-flag mismatch.

2. **Session affinity.**  The stream id rides as ``X-SMG-Routing-Key`` so a
   consistent-hashing router keeps an agent's turns on the replica holding its
   prefix.  Without it a 100-turn agentic stream re-prefills its whole context
   on every hop.

3. **Weight version.**  Carried out of ``meta_info`` so a trainer can prove a
   rollout did not straddle a weight update.

Run:
    python test/test-sglang-transport.py
"""

from __future__ import annotations
import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver.rollout_provider import DirectRolloutProvider, SlimeRolloutProvider, _engine_meta, _sglang_generate  # noqa: E402
from proxyserver.engines import EngineError
from proxyserver.config import load_config  # noqa: E402
from offline_common import FakeTokenizer, check  # noqa: E402

SLIME_YAML = "inference_engine: sglang\ntransport_mode: slime\n"


def load_yaml_config(body: str):
    """Parse ``body`` as a config file, from a temp path so the error messages
    (which name the file) are exercised too."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(body)
        path = f.name
    try:
        return load_config(path)
    finally:
        Path(path).unlink()


PROMPT = [1, 2, 3]
PARAMS = {"max_tokens": 64, "logprobs": True, "temperature": 0.7, "top_p": None}
TOKENS = [900, 901, 902]
LOGPROBS = [-0.1, -0.2, -0.3]


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeHttp:
    """Captures the request and replays a canned SGLang response."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, json: dict[str, Any], headers: Any = None) -> FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers})
        return FakeResponse(self._payload)


def sglang_response(
    *,
    include_output_ids: bool,
    tokens: list[int] = TOKENS,
    logprobs: list[float] = LOGPROBS,
    finish_type: str = "stop",
    weight_version: str | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "finish_reason": {"type": finish_type},
        "output_token_logprobs": [(lp, tid, "") for lp, tid in zip(logprobs, tokens)],
    }
    if weight_version is not None:
        meta["weight_version"] = weight_version
    payload: dict[str, Any] = {"text": "".join(f"<{t}>" for t in tokens), "meta_info": meta}
    if include_output_ids:
        payload["output_ids"] = list(tokens)
    return payload


def call(payload: dict[str, Any], session_id: str | None = "trial_042-agent_001"):
    http = FakeHttp(payload)
    result = asyncio.run(_sglang_generate(http, PROMPT, dict(PARAMS), session_id))
    return result, http


def test_tokens_only_engine() -> None:
    print("\nTokens-only engine (--skip-tokenizer-init): ids come from output_ids")
    (token_ids, log_probs, stop_reason, meta), http = call(
        sglang_response(include_output_ids=True)
    )
    check("sampled ids recovered", token_ids == TOKENS)
    check("logprobs aligned", log_probs == LOGPROBS)
    check("stop reason passed through in SGLang's vocabulary", stop_reason == "stop")
    check("no engine metadata when the engine reports none", meta == {})
    body = http.calls[0]["json"]
    check("token-ID prompt sent as input_ids", body["input_ids"] == PROMPT)
    check("max_tokens translated to max_new_tokens", body["sampling_params"]["max_new_tokens"] == 64)
    check("logprobs translated to return_logprob", body["return_logprob"] is True)
    check("None-valued sampling knobs are dropped", "top_p" not in body["sampling_params"])
    check("resolved knobs ride through", body["sampling_params"]["temperature"] == 0.7)


def test_tokenizer_initialized_engine() -> None:
    print("\nTokenizer-initialized engine (no output_ids): ids come from the logprob triples")
    # This is what slime's own rollout reads (sglang_rollout.py), and the only
    # place the ids exist when the engine decodes text itself.
    (token_ids, log_probs, stop_reason, _), _ = call(
        sglang_response(include_output_ids=False)
    )
    check("sampled ids recovered from output_token_logprobs", token_ids == TOKENS)
    check("logprobs still aligned", log_probs == LOGPROBS)
    check("the turn is usable, not an EngineError", bool(token_ids) and stop_reason == "stop")


def test_empty_output_ids_key() -> None:
    print("\nAn engine that returns an empty output_ids still falls back")
    payload = sglang_response(include_output_ids=True)
    payload["output_ids"] = []
    (token_ids, log_probs, _, _), _ = call(payload)
    check("ids recovered despite the empty key", token_ids == TOKENS)
    check("logprobs aligned", log_probs == LOGPROBS)


def test_genuine_mismatch_still_refused() -> None:
    print("\nA genuinely misaligned payload is still refused, not papered over")
    # The fallback must not mask real corruption: output_ids present but of a
    # different length than the logprobs means the engine contradicted itself.
    payload = sglang_response(include_output_ids=True)
    payload["output_ids"] = [900, 901]  # one short of the three logprob entries
    (token_ids, log_probs, stop_reason, _), _ = call(payload)
    check("empty result, which generate() turns into an EngineError", token_ids == [])
    check("no logprobs recorded either", log_probs == [])
    check("stop reason still reported for the log", stop_reason == "stop")


def test_truncation_reported() -> None:
    print("\nTruncation is reported in SGLang's own vocabulary")
    (_, _, stop_reason, _), _ = call(
        sglang_response(include_output_ids=False, finish_type="length")
    )
    check("finish_reason.type passed through verbatim", stop_reason == "length")


def test_routing_key_header() -> None:
    print("\nThe stream id rides as the router's session-affinity key")
    _, http = call(sglang_response(include_output_ids=True), session_id="trial_042-agent_001")
    headers = http.calls[0]["headers"]
    check("X-SMG-Routing-Key carries the stream id",
          headers == {"X-SMG-Routing-Key": "trial_042-agent_001"})

    # Keyed on the stream, not the rollout: agents of one trial hold
    # independent token streams with no shared prefix.
    _, other = call(sglang_response(include_output_ids=True), session_id="trial_042-agent_002")
    check("a sibling agent of the same rollout gets its own key",
          other.calls[0]["headers"]["X-SMG-Routing-Key"] != headers["X-SMG-Routing-Key"])

    _, anon = call(sglang_response(include_output_ids=True), session_id=None)
    check("no key sent for a session-less request", anon.calls[0]["headers"] is None)


def test_weight_version_carried() -> None:
    print("\nThe engine's weight version is carried out of meta_info")
    (_, _, _, meta), _ = call(
        sglang_response(include_output_ids=True, weight_version="step-7")
    )
    check("weight_version extracted", meta == {"weight_version": "step-7"})

    check("a numeric version is normalized to str",
          _engine_meta({"weight_version": 12}) == {"weight_version": "12"})
    check("version 0 is kept, not dropped as falsy",
          _engine_meta({"weight_version": 0}) == {"weight_version": "0"})
    check("nothing reported -> empty meta", _engine_meta({}) == {})
    check("unrelated meta_info fields are not carried",
          _engine_meta({"cached_tokens": 12, "prompt_tokens": 40}) == {})


class FakeRouterHttp(FakeHttp):
    """FakeHttp plus the router's ``GET /get_server_info``."""

    def __init__(self, payload: dict[str, Any], server_info: Any = None, fail_info: bool = False) -> None:
        super().__init__(payload)
        self._server_info = server_info
        self._fail_info = fail_info
        self.info_calls = 0

    async def get(self, url: str) -> FakeResponse:
        self.info_calls += 1
        if self._fail_info:
            raise RuntimeError("router: no such endpoint")
        return FakeResponse(self._server_info)


async def provider_with_http(http, mode="slime", context_length=None):
    loader = lambda path: FakeTokenizer()
    if mode == "slime":
        provider = SlimeRolloutProvider("http://router.test", tokenizer_loader=loader,
                                       context_length=context_length)
        await provider.aclose()
        provider._http = http
    else:
        provider = DirectRolloutProvider("http://engine.test", inference_engine="sglang",
                                        tokenizer_loader=loader)
        await provider.aclose()
        provider._clients = [http]
    return provider


def provider_call(http: FakeRouterHttp, turns: int = 1, context_length: int | None = None, *, mode="slime"):
    """Run discovery and generation on one loop for either SGLang transport."""
    async def drive():
        provider = await provider_with_http(http, mode, context_length)
        for _ in range(turns):
            await provider._call_engine(PROMPT, dict(PARAMS), "trial_042-agent_001")

    asyncio.run(drive())
    return http


def test_context_window_clamp() -> None:
    print("\nBoth SGLang providers share the context discovery and clamp contract")
    cases = (
        ("clamped", {"context_length": len(PROMPT) + 2}, False, 2, 1),
        ("room to spare", {"context_length": len(PROMPT) + 100}, False, PARAMS["max_tokens"], 1),
        ("list-wrapped info", [{"context_length": len(PROMPT) + 100}], False, PARAMS["max_tokens"], 1),
        ("default launch", {"context_length": None, "max_req_input_len": len(PROMPT) + 2}, False, 2, 1),
        ("fetch failure", None, True, PARAMS["max_tokens"], 2),
        ("unknown window", {"model_path": "/x"}, False, PARAMS["max_tokens"], 1),
    )
    for mode in ("slime", "direct"):
        for label, info, fail_info, cap, fetches in cases:
            http = provider_call(FakeRouterHttp(
                sglang_response(include_output_ids=True, tokens=TOKENS[:2], logprobs=LOGPROBS[:2]),
                server_info=info, fail_info=fail_info,
            ), turns=2, mode=mode)
            check(f"{mode}/{label}: effective token budget",
                  [c["json"]["sampling_params"]["max_new_tokens"] for c in http.calls] == [cap, cap])
            check(f"{mode}/{label}: discovery caches responses and retries failures", http.info_calls == fetches)

        http = FakeRouterHttp(sglang_response(include_output_ids=True),
                              server_info={"context_length": len(PROMPT)})
        try:
            provider_call(http, mode=mode)
        except EngineError as e:
            check(f"{mode}: a full window is refused before generation",
                  "no room left to generate" in str(e) and not http.calls)
        else:
            raise AssertionError(f"{mode}: a prompt filling the context window was accepted")


def test_context_discovery_recovery() -> None:
    print("\nContext discovery retries failures and cancellation, and serializes concurrent fetches")

    class GatedHttp:
        def __init__(self):
            self.calls = 0
            self.fail = True
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def get(self, url):
            assert url == "/get_server_info"
            self.calls += 1
            if self.fail:
                raise RuntimeError("temporarily unavailable")
            self.entered.set()
            await self.release.wait()
            return FakeResponse({"context_length": 1024})

    async def scenario(mode):
        http = GatedHttp()
        provider = await provider_with_http(http, mode)
        if mode == "slime":
            discover = provider._engine_context_length
        else:
            async def discover():
                return await provider._sglang_context_length(0)

        with patch("proxyserver.rollout_provider.logger.warning") as warning:
            assert await discover() is None
            assert await discover() is None
            check(f"{mode}: failed fetches retry and warn once", http.calls == 2 and warning.call_count == 1)

            http.fail = False
            cancelled = asyncio.create_task(discover())
            await asyncio.wait_for(http.entered.wait(), timeout=1)
            cancelled.cancel()
            result = await asyncio.gather(cancelled, return_exceptions=True)
            check(f"{mode}: cancellation propagates without a fetch warning",
                  isinstance(result[0], asyncio.CancelledError) and warning.call_count == 1)

            http.entered.clear()
            first = asyncio.create_task(discover())
            await asyncio.wait_for(http.entered.wait(), timeout=1)
            second = asyncio.create_task(discover())
            await asyncio.sleep(0)
            check(f"{mode}: concurrent discovery shares one in-flight fetch", http.calls == 4 and not second.done())
            http.release.set()
            values = await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
            check(f"{mode}: discovery recovers after cancellation", values == [1024, 1024])
            check(f"{mode}: the recovered window is cached", await discover() == 1024 and http.calls == 4)

    for mode in ("slime", "direct"):
        asyncio.run(scenario(mode))


#: What slime's sgl-router actually answers /get_server_info with: its own
#: RouterManager stub, not the worker payload discovery is written against.
ROUTER_MANAGER_STUB = {"router_manager": True, "routers_count": 1, "workers_count": 4}


def test_configured_context_window() -> None:
    print("\nA configured context_length pins the window the router cannot report")

    http = provider_call(FakeRouterHttp(sglang_response(include_output_ids=True), server_info=ROUTER_MANAGER_STUB),
                      turns=2, context_length=len(PROMPT) + 2)
    check("max_new_tokens is clamped to the configured window",
          [c["json"]["sampling_params"]["max_new_tokens"] for c in http.calls] == [2, 2])
    check("the router is not asked at all", http.info_calls == 0)

    # Left unconfigured, that same stub is what silently disables the clamp for
    # a whole run -- the failure this field exists to close.
    http = provider_call(FakeRouterHttp(sglang_response(include_output_ids=True), server_info=ROUTER_MANAGER_STUB))
    check("the router's stub alone leaves requests unclamped",
          http.calls[0]["json"]["sampling_params"]["max_new_tokens"] == PARAMS["max_tokens"])

    for bad in (0, -1):
        try:
            SlimeRolloutProvider("http://router.test", context_length=bad)
        except ValueError:
            check(f"context_length={bad} refused at construction", True)
        else:
            check(f"context_length={bad} refused at construction", False)


def test_configured_context_window_from_yaml() -> None:
    print("\nThe window comes off the config file")

    cfg = load_yaml_config(SLIME_YAML + "context_length: 262144\n")
    check("config: context_length parses", cfg.context_length == 262144)
    check("config: absent context_length stays None", load_yaml_config(SLIME_YAML).context_length is None)

    # Refused rather than ignored where it would do nothing: under direct
    # the window comes from the engines themselves.
    for label, body in [
        ("non-slime transport", "inference_engine: sglang\ntransport_mode: direct\ncontext_length: 262144\n"),
        ("non-numeric", SLIME_YAML + "context_length: wide\n"),
        ("zero", SLIME_YAML + "context_length: 0\n"),
        ("negative", SLIME_YAML + "context_length: -1\n"),
    ]:
        try:
            load_yaml_config(body)
        except ValueError as e:
            check(f"config: {label} refused, error names the field", "context_length" in str(e))
        else:
            check(f"config: {label} refused, error names the field", False)


def main() -> None:
    test_tokens_only_engine()
    test_tokenizer_initialized_engine()
    test_empty_output_ids_key()
    test_genuine_mismatch_still_refused()
    test_truncation_reported()
    test_routing_key_header()
    test_weight_version_carried()
    test_context_window_clamp()
    test_context_discovery_recovery()
    test_configured_context_window()
    test_configured_context_window_from_yaml()
    print("\n" + "=" * 70)
    print("PASS: the SGLang transport recovers sampled token ids from an engine\n"
          "      started either way — tokens-only or with its tokenizer — while\n"
          "      still refusing a genuinely misaligned payload; it sends the\n"
          "      stream id as the router's session-affinity key; and it carries\n"
          "      the engine's weight version out for the trainer.")
    print("=" * 70)


if __name__ == "__main__":
    main()
