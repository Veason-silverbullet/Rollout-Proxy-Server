"""Offline test of the tokenizer-identity fingerprint (training–rollout consistency).

Pins, in order:

1. **The corpus contract** — the probe corpus hashes to the pinned
   ``PROBE_CORPUS_SHA256`` value.
2. **The algorithm** — deterministic; sensitive to probe behavior, vocab and
   added-token and backend-rule changes; *insensitive* to the chat template (the
   template is a deliberate adaptation, pinned separately via ``template_pin``).
3. **The registry payload** — computed once per profile on the runtime
   tokenizer, cached, and carrying the profile's strict-TITO markers.
4. **The HTTP surface** — ``GET /tokenizer_fingerprint`` is open, 404s an
   unmapped model naming the mapping, and answers 501 without a callback.
5. **The bundled Qwen3.5, Qwen3.6, and Qwen3.8 tokenizers** — their fingerprints and
   template pins equal recorded constants.  These ARE the deployment pin: changing
   the bundle or the profile markers is a deliberate, reward-adjacent operation,
   and this test is where it gets acknowledged.

Run:
    python test/test-tokenizer-fingerprint.py
"""

from __future__ import annotations
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver.model_registry import ModelRegistry, TOKENIZATION_DIR, UnknownModelError
from proxyserver.service import LLMProxyServer, make_completion_handler
from proxyserver.tokenization import fingerprint as fp
from proxyserver.tokenization.base import get_profile
from proxyserver.tokenization.qwen3_5 import PROFILE as QWEN3_5
from proxyserver.tokenization.qwen3_6 import PROFILE as QWEN3_6
from proxyserver.tokenization.qwen3_8 import PROFILE as QWEN3_8
from offline_common import ScriptedProvider, check, load_real_tokenizer

# Any model in tokenization/mapping.json mapped to the qwen3_5 profile.
MODEL = "Qwen3.5-9B"
PROXY_API_KEY = "tokenizer-fingerprint-test-key"

# Independent pins for Qwen3.5 token identity and its serving template plus
# profile markers. Review changes deliberately: template-only edits update
# the template pin while leaving the token-identity fingerprint unchanged.
BUNDLE_FINGERPRINT = "3e0ce0521c3ddde1e820945fbb411e2dd35f2eb0af0d4013e8398a7309289c9e"
BUNDLE_TEMPLATE_PIN = "85a9141fc1f14b1157ce907f5fcb9732921b996c04ef5b41b61942376d37e381"


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
        self.split_special_tokens = False
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


def test_algorithm() -> None:
    check("corpus digest matches the protocol pin", fp.probe_corpus_digest() == fp.PROBE_CORPUS_SHA256)

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
    for backend in (None, _Backend({}), _Backend({"model": []}), _Backend({"model": {}})):
        invalid = RichFakeTokenizer()
        invalid.backend_tokenizer = backend
        try:
            fp.tokenizer_fingerprint(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete tokenizer rules were accepted as a strict identity")
    check("missing backend rules fail closed", True)

    # "a b" and ["a","b"] are the same merge across tokenizers versions.
    check("merge serialization shapes normalize identically", fp._normalized_merges(["a b", ["c", "d"]]) == [["a", "b"], ["c", "d"]])
    equivalent = RichFakeTokenizer()
    equivalent.backend_tokenizer._data["model"]["merges"] = [["a", "b"], "b c"]
    check("equivalent backend serialization preserves strict identity",
          fp.tokenizer_fingerprint(equivalent)["fingerprint"] == base["fingerprint"])

    for component in ("normalizer", "pre_tokenizer", "post_processor", "decoder"):
        changed = RichFakeTokenizer()
        changed.backend_tokenizer._data[component] = {"type": "changed"}
        payload = fp.tokenizer_fingerprint(changed)
        check(f"{component} changes strict identity even when probes are unchanged",
              payload["components"]["probe"] == base["components"]["probe"]
              and payload["fingerprint"] != base["fingerprint"])
    changed = RichFakeTokenizer()
    changed.backend_tokenizer._data["model"]["byte_fallback"] = True
    check("model options are part of strict identity", fp.tokenizer_fingerprint(changed)["fingerprint"] != base["fingerprint"])

    changed = RichFakeTokenizer()
    changed.split_special_tokens = True
    payload = fp.tokenizer_fingerprint(changed)
    check("special-token splitting changes identity even when probes and backend match",
          payload["components"]["probe"] == base["components"]["probe"]
          and payload["components"]["backend"] == base["components"]["backend"]
          and payload["fingerprint"] != base["fingerprint"])


def test_unprobed_merge_priority() -> None:
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "Ω": 1, "Ξ": 2, "Ψ": 3, "ΩΞ": 4, "ΞΨ": 5}
    merges = [("Ω", "Ξ"), ("Ξ", "Ψ")]
    tokenizers = [PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.BPE(vocab=vocab, merges=order, unk_token="[UNK]")),
        unk_token="[UNK]",
    ) for order in (merges, list(reversed(merges)))]
    left, right = map(fp.tokenizer_fingerprint, tokenizers)
    check("regression tokenizers agree on vocab and every fixed probe",
          left["components"]["vocab"] == right["components"]["vocab"]
          and left["components"]["probe"] == right["components"]["probe"])
    check("regression tokenizers disagree on an unprobed input",
          tokenizers[0].encode("ΩΞΨ") == [4, 3] and tokenizers[1].encode("ΩΞΨ") == [1, 5])
    check("unprobed merge priority changes strict identity", left["fingerprint"] != right["fingerprint"])


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
    check("template_pin = template + markers", payload["template_pin"] == fp.template_pin(payload["template_sha256"], QWEN3_5.generation_prompt, QWEN3_5.assistant_turn_end))
    again = registry.fingerprint_payload(MODEL)
    check("second call is cached (one tokenizer load)", loads == 1 and again == payload)

    try:
        registry.fingerprint_payload("no-such-model")
    except UnknownModelError as e:
        check("unmapped model raises UnknownModelError naming the mapping", "No profile mapped" in str(e))
    else:
        raise AssertionError("FAIL: unmapped model accepted")


def resolve_concurrently(registry: ModelRegistry, count: int = 8) -> list[Any]:
    """Resolve MODEL from daemon threads; a deadlock fails instead of hanging the suite."""
    results: list[Any] = [None] * count

    def worker(index: int) -> None:
        try:
            results[index] = registry.resolve(MODEL)
        except Exception as e:
            results[index] = e

    threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(count)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 10
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    assert not any(thread.is_alive() for thread in threads), "FAIL: concurrent resolves did not finish"
    return results


def test_concurrent_loading() -> None:
    loads = 0

    def slow_loader(path: str) -> RichFakeTokenizer:
        nonlocal loads
        loads += 1
        time.sleep(0.2)  # every first request arrives while the load runs
        return RichFakeTokenizer()

    registry = ModelRegistry(tokenizer_loader=slow_loader)
    bundles = resolve_concurrently(registry)
    check("concurrent first requests share one tokenizer load",
          loads == 1 and all(bundle is bundles[0] for bundle in bundles))

    attempts = 0

    def failing_loader(path: str) -> RichFakeTokenizer:
        nonlocal attempts
        attempts += 1
        time.sleep(0.2)
        if attempts == 1:
            raise OSError("transient tokenizer read failure")
        return RichFakeTokenizer()

    registry = ModelRegistry(tokenizer_loader=failing_loader)
    results = resolve_concurrently(registry)
    check("waiters share a failed load instead of retrying it in turn",
          attempts == 1 and all(isinstance(result, OSError) for result in results))
    check("the next request retries the failed load", registry.resolve(MODEL) is not None and attempts == 2)


def test_concurrent_first_import() -> None:
    # transformers' lazy module fails concurrent first imports. A fresh
    # interpreter keeps this process's earlier imports from masking the race;
    # the empty directory makes each load fail fast after the import.
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    script = """
import sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from proxyserver.model_registry import load_local_tokenizer

def load(path):
    try:
        load_local_tokenizer(path)
    except (OSError, ValueError):
        return None  # an empty directory is not a tokenizer
    except Exception as e:
        return repr(e)

with tempfile.TemporaryDirectory() as empty, ThreadPoolExecutor(8) as pool:
    errors = [e for e in pool.map(load, [empty] * 8) if e]
sys.exit("\\n".join(errors) or None)
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, f"FAIL: concurrent first imports failed:\n{result.stderr[-2000:]}"
    check("concurrent first tokenizer loads import transformers safely", True)

    script = """
import sys
from proxyserver.model_registry import ModelRegistry
assert "transformers" not in sys.modules
ModelRegistry()
assert "transformers" in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True, timeout=300, env=env)
    assert result.returncode == 0, f"FAIL: default registry did not import transformers:\n{result.stderr[-2000:]}"
    check("a default registry imports transformers before worker threads load tokenizers", True)


async def test_endpoint() -> None:
    provider = ScriptedProvider()
    provider.models = ModelRegistry(tokenizer_loader=lambda path: RichFakeTokenizer())
    proxy = LLMProxyServer(
        host="127.0.0.1", port=0, api_key=PROXY_API_KEY,
        completion_handler=make_completion_handler(provider),
        on_session_deleted=provider.release_session,
        get_tokenizer_fingerprint=provider.tokenizer_fingerprint,
        save_rollout_sessions=False,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as http:
        resp = await http.get("/tokenizer_fingerprint", params={"model": MODEL})
        check("GET is open and answers 200", resp.status_code == 200)
        body = resp.json()
        check("GET echoes model, fingerprint, template_pin", body["model"] == MODEL and body["fingerprint"] and body["template_pin"])

        resp = await http.get("/tokenizer_fingerprint", params={"model": "no-such-model"})
        check("unmapped model -> 404 naming the mapping", resp.status_code == 404 and "No profile mapped" in resp.json()["detail"])

        resp = await http.get("/tokenizer_fingerprint")
        check("missing model param -> 422", resp.status_code == 422)


async def test_missing_callbacks_501() -> None:
    proxy = LLMProxyServer(host="127.0.0.1", port=0, api_key=PROXY_API_KEY, save_rollout_sessions=False,
                           completion_handler=make_completion_handler(ScriptedProvider()))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as http:
        resp = await http.get("/tokenizer_fingerprint", params={"model": MODEL})
        check("missing callback -> 501", resp.status_code == 501)


def test_real_bundle() -> None:
    # The Qwen profiles share the token vocabulary/encoding, but each has its own template.
    bundles = (
        (QWEN3_5, BUNDLE_FINGERPRINT, BUNDLE_TEMPLATE_PIN),
        (QWEN3_6, "3e0ce0521c3ddde1e820945fbb411e2dd35f2eb0af0d4013e8398a7309289c9e",
         "7b812d0bf2c637c0f3ff3e7edead331ea19e3342b31d677db0652b3b300d00a7"),
        (QWEN3_8, "3e0ce0521c3ddde1e820945fbb411e2dd35f2eb0af0d4013e8398a7309289c9e",
         "0fa30614f0bb600bad3e71e9c334000e10d3a5bc87bb00a9049ec238c0023a64"),
    )
    for profile, expected_fingerprint, expected_pin in bundles:
        tokenizer = load_real_tokenizer(str(TOKENIZATION_DIR / profile.tokenizer_dir))
        if tokenizer is None:
            continue
        payload = fp.tokenizer_fingerprint(tokenizer)
        label = profile.tokenizer_dir
        check(f"{label}: a real vocab was hashed", payload["components"]["vocab_size"] > 100_000)
        check(f"{label}: structural hashes present (fast tokenizer)", payload["components"]["structural"] is not None)
        check(f"{label}: deterministic", fp.tokenizer_fingerprint(tokenizer)["fingerprint"] == payload["fingerprint"])
        pin = fp.template_pin(payload["template_sha256"], profile.generation_prompt, profile.assistant_turn_end)
        check(f"{label}: fingerprint matches the recorded pin", payload["fingerprint"] == expected_fingerprint)
        check(f"{label}: template pin matches the recorded pin", pin == expected_pin)


def test_glm_bundle() -> None:
    payload = ModelRegistry().fingerprint_payload("GLM-5.3-Flash")
    check("bundled GLM token identity and serving format match their deployment pins",
          payload["fingerprint"] == "421d4d01b4a77fdcf061338db86b0721620e7cae4b9d180fdad6da9053ec1e07"
          and payload["template_pin"] == "be186223d0f57215cb8a1fa0bc3bc67949e9413bcd3ee41ed023f5e773cd7570")
    check("fingerprint advertises and pins sampled continuation headers",
          payload["continuation_headers"] == ["<|user|>", "<|observation|>"]
          and payload["template_pin"] != fp.template_pin(
              payload["template_sha256"], "<|assistant|>", ""))


def test_glm_special_token_splitting() -> None:
    from transformers import AutoTokenizer

    tokenizers = [AutoTokenizer.from_pretrained(
        str(TOKENIZATION_DIR / "GLM-5.3-Flash"), local_files_only=True, split_special_tokens=split,
    ) for split in (False, True)]
    left, right = map(fp.tokenizer_fingerprint, tokenizers)
    check("GLM splitting settings agree on every fixed probe and serialized backend",
          all(left["components"][key] == right["components"][key]
              for key in ("probe", "vocab", "added_tokens", "backend")))
    for text in ("<|assistant|>", "[gMASK]<sop><|user|>\nHi"):
        check("GLM splitting settings disagree on unprobed special markers",
              tokenizers[0].encode(text, add_special_tokens=False)
              != tokenizers[1].encode(text, add_special_tokens=False))
    check("GLM special-token splitting changes strict identity",
          left["algorithm"] == right["algorithm"] == fp.FINGERPRINT_ALGORITHM
          and left["fingerprint"] != right["fingerprint"])


def test_gpt_oss_bundle() -> None:
    registry = ModelRegistry()
    profile = get_profile("gpt_oss")
    model = "gpt-oss-20b"
    payload = registry.fingerprint_payload(model)
    check("identity and format fingerprints are available", bool(payload["fingerprint"]) and bool(payload["template_pin"]))
    check("format pin covers completed action endings", payload["assistant_turn_endings"] == list(profile.assistant_turn_endings)
          and payload["template_pin"] != fp.template_pin(payload["template_sha256"], profile.generation_prompt, profile.assistant_turn_end))


async def main() -> None:
    print("algorithm:")
    test_algorithm()
    test_glm_special_token_splitting()
    test_unprobed_merge_priority()
    print("registry payload:")
    test_registry()
    test_concurrent_loading()
    test_concurrent_first_import()
    print("endpoint:")
    await test_endpoint()
    await test_missing_callbacks_501()
    print("bundled tokenizer pin:")
    test_real_bundle()
    test_glm_bundle()
    test_gpt_oss_bundle()
    print("ALL TOKENIZER-FINGERPRINT TESTS PASS")


if __name__ == "__main__":
    asyncio.run(main())
