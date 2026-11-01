"""Tokenizer identity shared by trainers and the recording proxy.

Strict identity covers probe encodings, vocabulary, added tokens, the serialized
backend, and runtime encoding settings. Serving templates are pinned separately
because they may deliberately differ from checkpoint templates. The algorithm
and corpus hashes are wire contracts: changes require an algorithm version bump.
No service implementation is imported; Transformers is loaded only when reading
a checkpoint tokenizer.
"""

from __future__ import annotations
import hashlib
import json
from typing import Any

#: Version of the fingerprint algorithm.  Bump on any change to the corpus,
#: the canonicalization, or the component set.
FINGERPRINT_ALGORITHM = "v1"

#: Fixed probe corpus shared by trainers and proxies. Encoded with
#: ``add_special_tokens=False`` on both sides; the id sequences are the
#: behavioral half of the fingerprint.  Chosen to exercise the places BPE
#: tokenizers actually diverge: merges over whitespace runs and punctuation,
#: byte-level fallback, unicode normalization, added-token splitting.
#: Append-only in spirit — any edit is an algorithm bump.
PROBE_CORPUS: tuple[str, ...] = (
    # plain text and casing
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "HTTP/2 200 OK\r\nContent-Type: application/json\r\n\r\n",
    "llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch",
    "ALLCAPS lowercase MiXeD CaSe iPhone eBay McDonald's",
    # numbers, dates, identifiers
    "3.14159265358979 1e-6 0xDEADBEEF 1_000_000 -42",
    "1970-01-01T12:34:56.789Z +08:00 v2.7.1-rc.3",
    "a1b2c3d4e5f6a7b8c9d0a1b2c3d4e5f6",
    # whitespace and indentation (merge-rule sensitive)
    "    indented\n\twith tabs\n        deeper\n",
    "spaces:  two   three    four     five\n\n\nnewlines",
    " \t \t mixed \u00a0 nbsp \u2009 thin-space \u3000 ideographic",
    # code
    'def f(x: int) -> list[str]:\n    return [f"{x:04d}" for _ in range(x)]\n',
    "SELECT * FROM t WHERE a <= 3 AND b != 'x' ORDER BY c DESC;",
    "if (a&&b||!c) { p->q += *r++; } // C++ ops\n",
    "const s = `template ${a + b} literal`;\n",
    "<div class=\"a\"><span id='b'>&amp;&lt;&gt;</span></div>",
    '{"key": "value", "n": [1, 2, 3], "b": true, "z": null}',
    're = r"^\\d{3}-\\d{4}$" # backslashes \\\\ \\n \\t',
    # CJK / mixed-script
    "深度学习模型的分词器必须逐字节一致。",
    "日本語のトークナイザーと한국어 토크나이저",
    "中英 mixed 文本 with English words 混排",
    "こんにちは世界、你好世界、안녕하세요 세계",
    # RTL and bidi
    "مرحبا بالعالم — שלום עולם",
    "العربية text מעורב mixed",
    # accents and normalization-sensitive sequences
    "café résumé naïve Zürich São Paulo",
    "e\u0301 (combining) vs \u00e9 (precomposed)",
    "ﬁnal ﬂight ǅungla ǈubav",  # ligatures and titlecase digraphs
    # emoji, ZWJ, variation selectors
    "🙂 🙃 👍🏽 👨‍👩‍👧‍👦 🇨🇳 🇺🇸 ✊🏿",
    "emoji in text 🚀mid🚀word and trailing 🎉",
    # math and symbols
    "∀x∈ℝ: x² ≥ 0 ⇒ ‖x‖ ≠ −∞ ± ε",
    "£100 €50 ¥1000 ₹75 $3.50 ¢99",
    # URLs and paths
    "https://example.com/a/b?c=d&e=f#g",
    "/usr/local/lib/python3.12/site-packages/",
    "C:\\Users\\name\\AppData\\Local\\Temp\\file.txt",
    # chat-format marker strings as *literal text* — exercises added-token
    # splitting behavior (with add_special_tokens=False these should still
    # resolve through the added-token machinery identically on both sides)
    "<|im_start|>assistant\n",
    "<|im_end|>\n<|im_start|>user\n",
    "<think>\n\n</think>\n\n",
    "<tool_call>\n<function=run>\n</function>\n</tool_call>",
    # byte-level edge cases
    "\x00\x01\x02 control bytes \x1b[31mansi\x1b[0m",
    "surrogate-adjacent: \ud7ff \ue000 \ufffd",
    "long run: " + "ab" * 64 + " " + "字" * 32,
)


def _canonical(obj: Any) -> bytes:
    """Canonical JSON bytes: sorted keys, no whitespace, raw unicode."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


def probe_corpus_digest() -> str:
    """Identify the probe corpus before comparing tokenizer fingerprints."""
    return _sha256(list(PROBE_CORPUS))


#: Pinned by tests. Corpus edits require a FINGERPRINT_ALGORITHM bump.
PROBE_CORPUS_SHA256 = "082bb237f99405ca2f558f7fea913234d2adc0bd88addcf3ffcf8fc9d0fb3cef"


def _normalized_merges(merges: Any) -> list[list[str]] | None:
    """Merges as ordered pairs, whatever serialization shape they arrived in.

    ``tokenizers`` has serialized merges both as ``"a b"`` strings and as
    ``["a", "b"]`` pairs across versions; normalizing keeps the structural
    hash comparable across that difference (order is preserved — merge
    priority is positional).
    """
    if merges is None:
        return None
    if not isinstance(merges, list):
        raise ValueError("Tokenizer merges must be an ordered list")
    normalized: list[list[str]] = []
    for entry in merges:
        if isinstance(entry, str):
            parts = entry.split(" ")
            if len(parts) != 2:
                raise ValueError("Cannot canonicalize tokenizer merge: expected two tokens")
        elif isinstance(entry, (list, tuple)):
            parts = list(entry)
        else:
            raise ValueError("Cannot canonicalize tokenizer merge: expected a token pair")
        if len(parts) != 2 or not all(isinstance(part, str) for part in parts):
            raise ValueError("Cannot canonicalize tokenizer merge: expected two strings")
        normalized.append(parts)
    return normalized


def _added_token_table(tokenizer: Any) -> list[list[Any]]:
    """The added/special token table with ids and flags, version-stably.

    ``added_tokens_decoder`` (id -> AddedToken) is the HF-level surface; the
    flags matter because e.g. a differing ``lstrip`` changes how text around
    a marker tokenizes.  Sorted by id so dict ordering cannot leak in.
    """
    table: list[list[Any]] = []
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for token_id, token in decoder.items():
        table.append(
            [
                int(token_id),
                str(getattr(token, "content", token)),
                bool(getattr(token, "special", False)),
                bool(getattr(token, "lstrip", False)),
                bool(getattr(token, "rstrip", False)),
                bool(getattr(token, "single_word", False)),
                bool(getattr(token, "normalized", False)),
            ]
        )
    return sorted(table)


def tokenizer_fingerprint(tokenizer: Any) -> dict[str, Any]:
    """Fingerprint a loaded tokenizer.

    Returns ``{"algorithm", "corpus_sha256", "fingerprint", "components",
    "template_sha256"}``.  ``fingerprint`` is the strict identity (see module
    docstring); ``components`` carries each part plus the diagnostic
    ``structural`` hashes; ``template_sha256`` covers only the raw
    ``chat_template`` — callers with chat markers of their own (the proxy's
    profiles) hash those in via :func:`template_pin`.
    """
    serialize = getattr(getattr(tokenizer, "backend_tokenizer", None), "to_str", None)
    if not callable(serialize):
        raise ValueError("Strict tokenizer identity requires a serializable fast-tokenizer backend")
    try:
        data = json.loads(serialize())
    except Exception as e:
        raise ValueError("Strict tokenizer identity requires a serializable fast-tokenizer backend") from e
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("model"), dict)
        or not data["model"]
    ):
        raise ValueError("Strict tokenizer identity requires a serialized tokenizer model")
    model = dict(data["model"])
    if "merges" in model:
        model["merges"] = _normalized_merges(model["merges"])
    chain = [data.get(key) for key in ("normalizer", "pre_tokenizer", "post_processor", "decoder")]

    probe = [
        [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
        for text in PROBE_CORPUS
    ]
    vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(vocab):
        raise ValueError("Strict tokenizer identity requires get_vocab()")
    vocab_items = sorted((str(token), int(token_id)) for token, token_id in vocab().items())
    added_tokens = _added_token_table(tokenizer)
    # HF applies this runtime default during encode(); backend JSON omits it.
    encoding = {"split_special_tokens": bool(getattr(tokenizer, "split_special_tokens", False))}
    components: dict[str, Any] = {
        "probe": _sha256(probe),
        "vocab": _sha256(vocab_items),
        "vocab_size": len(vocab_items),
        "added_tokens": _sha256(added_tokens),
        "backend": _sha256({"model": model, "chain": chain}),
        "encoding": _sha256(encoding),
        "structural": {
            "merges": _sha256(model.get("merges")),
            "normalizer_chain": _sha256(chain),
        },
    }

    fingerprint = _sha256(
        {
            "algorithm": FINGERPRINT_ALGORITHM,
            "corpus": probe_corpus_digest(),
            "probe": components["probe"],
            "vocab": components["vocab"],
            "vocab_size": components["vocab_size"],
            "added_tokens": components["added_tokens"],
            "backend": components["backend"],
            "encoding": components["encoding"],
        }
    )
    template = getattr(tokenizer, "chat_template", None)
    return {
        "algorithm": FINGERPRINT_ALGORITHM,
        "corpus_sha256": probe_corpus_digest(),
        "fingerprint": fingerprint,
        "components": components,
        "template_sha256": _sha256(template),
    }


def template_pin(
    template_sha256: str,
    generation_prompt: str,
    assistant_turn_end: str,
    *,
    continuation_headers: tuple[str, ...] = (),
    assistant_turn_endings: tuple[str, ...] = (),
) -> str:
    """The pinnable hash of the serving chat format: the loaded template plus
    the profile's strict-TITO markers.  The proxy reports it as
    ``template_pin``; the trainer's recorded trust-on-first-use pin compares
    against it.  This — not equality with the checkpoint's release template —
    is the template invariant: the serving format is a deliberate adaptation,
    and the pin catches it being edited out from under a run."""
    markers: dict[str, Any] = {
        "template_sha256": template_sha256,
        "generation_prompt": generation_prompt,
        "assistant_turn_end": assistant_turn_end,
    }
    # Keep existing profiles' deployment pins unchanged.
    if continuation_headers:
        markers["continuation_headers"] = list(continuation_headers)
    if assistant_turn_endings:
        markers["assistant_turn_endings"] = list(assistant_turn_endings)
    return _sha256(markers)


#: hf_checkpoint path -> fingerprint payload; loading and hashing a 150k-entry
#: vocab is cheap but not per-step cheap, and the checkpoint never changes
#: under a running trainer.
_CHECKPOINT_CACHE: dict[str, dict[str, Any]] = {}


def checkpoint_fingerprint(hf_checkpoint: str) -> dict[str, Any]:
    """Fingerprint a checkpoint tokenizer, cached per path or model identifier.

    ``transformers`` is imported lazily so fingerprinting already-loaded
    tokenizers does not require it.  Callers loading checkpoint tokenizers
    must have it installed.  The cache assumes a checkpoint path's tokenizer
    remains unchanged for the lifetime of the process.
    """
    cached = _CHECKPOINT_CACHE.get(hf_checkpoint)
    if cached is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
        cached = _CHECKPOINT_CACHE.setdefault(hf_checkpoint, tokenizer_fingerprint(tokenizer))
    return cached
