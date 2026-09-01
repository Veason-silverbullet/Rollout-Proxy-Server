"""Offline test of the tokenizer-identity fingerprint (training–rollout consistency).

Pins, in order:

1. **The cross-repo corpus contract** — ``PROBE_CORPUS_SHA256`` matches the
   corpus.  ``harbor.slime_bridge.fingerprint`` carries an independent copy of
   the algorithm; both repos' tests pin the same constant, so the two
   implementations cannot drift apart silently.
2. **The algorithm** — deterministic; sensitive to probe behavior, vocab and
   added-token changes; *insensitive* to the chat template (the template is a
   deliberate adaptation, pinned separately via ``template_pin``).
3. **The registry payload** — computed once per profile on the runtime
   tokenizer, cached, and carrying the profile's strict-TITO markers.
4. **The HTTP surface** — ``GET /tokenizer_fingerprint`` is open, 404s an
   unmapped model naming the mapping, and answers 501 in relay mode.
5. **The bundled Qwen3.5 tokenizer** — its fingerprint and template pin equal
   recorded constants.  These ARE the deployment pin: changing the bundle or
   the profile markers is a deliberate, reward-adjacent operation, and this
   test is where it gets acknowledged.

Run:
    python test/test-tokenizer-fingerprint.py
"""

from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver.model_registry import ModelRegistry, TOKENIZATION_DIR, UnknownModelError  # noqa: E402
from proxyserver.server import LLMProxyServer, make_local_completion_handler  # noqa: E402
from proxyserver.tokenization import fingerprint as fp  # noqa: E402
from proxyserver.tokenization.qwen3_5 import PROFILE as QWEN3_5  # noqa: E402
from offline_common import ScriptedProvider, check, load_real_tokenizer  # noqa: E402

# Any model in tokenization/mapping.json mapped to the qwen3_5 profile.
MODEL = "Qwen3.5-9B"
PROXY_API_KEY = "tokenizer-fingerprint-test-key"

# The deployment pin of the bundled Qwen3.5 tokenizer + the profile's markers
# (section 5).  Update BOTH deliberately when the bundle or markers change —
# and treat that change as reward-adjacent (RL-training.md).
BUNDLE_FINGERPRINT = "431cc5b086450cb390000e7e26f966307115da3d43242b252984daa0cdd9aacb"
BUNDLE_TEMPLATE_PIN = "93b9422db5d6c59805907bfe2d69ae72874f457d4bc91967ffedd5b998b4b1c4"


class _Added:
    def __init__(self, content: str, special: bool = True) -> None:
        self.content = content
        self.special = special
        self.lstrip = False
        self.rstrip = False
        self.single_word = False
        self.normalized = False


class _Backend:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def to_str(self) -> str:
        return json.dumps(self._data)


class RichFakeTokenizer:
    """Deterministic tokenizer exposing every surface the fingerprint reads."""

    def __init__(self, salt: int = 0, template: str = "{{ messages }}") -> None:
        self._salt = salt
        self.chat_template = template
        self.added_tokens_decoder = {7: _Added("<|marker|>")}
        self.backend_tokenizer = _Backend(
            {
                "model": {"vocab": {"a": 1, "b": 2}, "merges": ["a b", ["b", "c"]]},
                "normalizer": {"type": "NFC"},
                "pre_tokenizer": None,
                "post_processor": None,
                "decoder": None,
            }
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [(ord(ch) + self._salt) % 65536 for ch in text]

    def get_vocab(self) -> dict[str, int]:
        return {"a": 1, "b": 2, "<|marker|>": 7 + self._salt}

    # TokenStreamManager tokenizes its generation prompt through encode();
    # apply_chat_template is unused by the fingerprint but the registry
    # builds a manager around the tokenizer, so keep the surface honest.
    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=True):
        if not messages:
            raise ValueError("empty conversation")
        return [100 + i for i in range(2 * len(messages))]

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        return "".join(f"<{t}>" for t in token_ids)


def test_algorithm() -> None:
    check("corpus digest matches the cross-repo pin", fp.probe_corpus_digest() == fp.PROBE_CORPUS_SHA256)

    base = fp.tokenizer_fingerprint(RichFakeTokenizer())
    again = fp.tokenizer_fingerprint(RichFakeTokenizer())
    check("fingerprint is deterministic", base == again)
    check("payload names the algorithm and corpus", base["algorithm"] == fp.FINGERPRINT_ALGORITHM and base["corpus_sha256"] == fp.PROBE_CORPUS_SHA256)

    salted = fp.tokenizer_fingerprint(RichFakeTokenizer(salt=1))
    check("probe/vocab change changes the fingerprint", salted["fingerprint"] != base["fingerprint"])
    check("components name what changed", salted["components"]["probe"] != base["components"]["probe"] and salted["components"]["vocab"] != base["components"]["vocab"])

    retemplated = fp.tokenizer_fingerprint(RichFakeTokenizer(template="{{ other }}"))
    check("template change leaves the strict fingerprint alone", retemplated["fingerprint"] == base["fingerprint"])
    check("template change changes template_sha256", retemplated["template_sha256"] != base["template_sha256"])

    pin_a = fp.template_pin(base["template_sha256"], "<|im_start|>assistant\n", "<|im_end|>\n")
    pin_b = fp.template_pin(base["template_sha256"], "<|im_start|>assistant\n<think>\n", "<|im_end|>\n")
    check("template_pin covers the markers", pin_a != pin_b)

    check("structural hashes ride along from the backend", base["components"]["structural"] is not None and "merges" in base["components"]["structural"])
    minimal = fp.tokenizer_fingerprint(type("T", (), {"encode": lambda self, text, add_special_tokens=False: [1]})())
    check("a backend-less tokenizer still fingerprints (structural null)", minimal["components"]["structural"] is None and minimal["fingerprint"])

    # "a b" and ["a","b"] are the same merge across tokenizers versions.
    check("merge serialization shapes normalize identically", fp._normalized_merges(["a b", ["c", "d"]]) == [["a", "b"], ["c", "d"]])


def test_registry() -> None:
    loads = 0

    def loader(path: str) -> RichFakeTokenizer:
        nonlocal loads
        loads += 1
        return RichFakeTokenizer()

    registry = ModelRegistry(tokenizer_loader=loader)
    payload = registry.fingerprint_payload(MODEL)
    check("payload echoes the model and profile", payload["model"] == MODEL and payload["profile"] == "qwen3_5")
    check("payload carries the profile's markers", payload["generation_prompt"] == QWEN3_5.generation_prompt and payload["assistant_turn_end"] == QWEN3_5.assistant_turn_end)
    check(
        "template_pin = template + markers",
        payload["template_pin"] == fp.template_pin(payload["template_sha256"], QWEN3_5.generation_prompt, QWEN3_5.assistant_turn_end),
    )
    again = registry.fingerprint_payload(MODEL)
    check("second call is cached (one tokenizer load)", loads == 1 and again == payload)

    try:
        registry.fingerprint_payload("no-such-model")
    except UnknownModelError as e:
        check("unmapped model raises UnknownModelError naming the mapping", "No profile mapped" in str(e))
    else:
        raise AssertionError("FAIL: unmapped model accepted")


async def test_endpoint() -> None:
    provider = ScriptedProvider()
    proxy = LLMProxyServer(
        host="127.0.0.1", port=0, api_key=PROXY_API_KEY,
        completion_handler=make_local_completion_handler(provider),
        on_session_deleted=provider.release_session,
        get_tokenizer_fingerprint=provider.tokenizer_fingerprint,
        save_rollout_sessions=False,
    )
    url = await proxy.start()
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{url}/tokenizer_fingerprint", params={"model": MODEL})
            check("GET is open and answers 200", resp.status_code == 200)
            body = resp.json()
            check("GET echoes model, fingerprint, template_pin", body["model"] == MODEL and body["fingerprint"] and body["template_pin"])

            resp = await http.get(f"{url}/tokenizer_fingerprint", params={"model": "no-such-model"})
            check("unmapped model -> 404 naming the mapping", resp.status_code == 404 and "No profile mapped" in resp.json()["detail"])

            resp = await http.get(f"{url}/tokenizer_fingerprint")
            check("missing model param -> 422", resp.status_code == 422)
    finally:
        await proxy.stop()


async def test_relay_mode_501() -> None:
    proxy = LLMProxyServer(host="127.0.0.1", port=0, api_key=PROXY_API_KEY, save_rollout_sessions=False)
    url = await proxy.start()
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{url}/tokenizer_fingerprint", params={"model": MODEL})
            check("relay mode -> 501", resp.status_code == 501)
    finally:
        await proxy.stop()


def test_real_bundle() -> None:
    tokenizer = load_real_tokenizer(str(TOKENIZATION_DIR / QWEN3_5.tokenizer_dir))
    if tokenizer is None:
        return
    payload = fp.tokenizer_fingerprint(tokenizer)
    check("bundle: a real vocab was hashed", payload["components"]["vocab_size"] > 100_000)
    check("bundle: structural hashes present (fast tokenizer)", payload["components"]["structural"] is not None)
    check("bundle: deterministic", fp.tokenizer_fingerprint(tokenizer)["fingerprint"] == payload["fingerprint"])
    pin = fp.template_pin(payload["template_sha256"], QWEN3_5.generation_prompt, QWEN3_5.assistant_turn_end)
    if BUNDLE_FINGERPRINT.startswith("__"):
        print(f"  note  record these pins in this file:\n        BUNDLE_FINGERPRINT = \"{payload['fingerprint']}\"\n        BUNDLE_TEMPLATE_PIN = \"{pin}\"")
        raise AssertionError("FAIL: bundle pins not recorded yet")
    check("bundle: fingerprint matches the recorded pin", payload["fingerprint"] == BUNDLE_FINGERPRINT)
    check("bundle: template pin matches the recorded pin", pin == BUNDLE_TEMPLATE_PIN)


async def main() -> None:
    print("algorithm:")
    test_algorithm()
    print("registry payload:")
    test_registry()
    print("endpoint:")
    await test_endpoint()
    print("relay mode:")
    await test_relay_mode_501()
    print("bundled tokenizer pin:")
    test_real_bundle()
    print("ALL TOKENIZER-FINGERPRINT TESTS PASS")


if __name__ == "__main__":
    asyncio.run(main())
