"""Tokenizer identity fingerprint (training–rollout consistency).

The rollout proxy tokenizes every prompt delta with its bundled tokenizer; the
trainer tokenizes and trains with ``--hf-checkpoint``'s.  If the two disagree
by one merge rule the recorded token ids are subtly off-distribution and
nothing ever errors.  This module computes a stable fingerprint of a *loaded*
tokenizer so the trainer can compare the proxy's serving tokenizer against its checkpoint's before any trial runs (``GET /tokenizer_fingerprint`` on the server,
checked by ``harbor.slime_bridge.rollout._sync_tokenizer_identity``).

Cross-repo contract: ``harbor/src/harbor/slime_bridge/fingerprint.py`` carries
an independent copy of this algorithm (the two repos share no dependency).
The probe corpus and the canonicalization MUST stay byte-identical on both
sides — both repos' tests pin :data:`PROBE_CORPUS_SHA256`, and payloads carry
``corpus_sha256`` so a version skew degrades to "incomparable", never to a false
mismatch.  Any change here is an algorithm bump, made in both repos together.

Three tiers, because they fail differently:

* **fingerprint** (strict — a mismatch is a real training hazard): sha256 over
  the probe encodings, the full vocab (``get_vocab()``), and the added/special
  token table with ids and flags.  Built only from version-stable HF-level
  surfaces, so two installs of ``tokenizers`` cannot disagree cosmetically.
* **structural** (diagnostic — compare, warn, never fail): hashes of the
  backend serialization's merges list (order-preserved: BPE merge priority is
  positional), and the normalizer/pre-tokenizer/post-processor/decoder chain.
  These name *which* component drifted when the fingerprint differs, and may
  legitimately differ across ``tokenizers`` versions when it does not.
* **template** (pinned, not equated): sha256 over the loaded
  ``tokenizer.chat_template`` plus the profile's strict-TITO markers.  The
  serving template is a *deliberate adaptation* of the release template (see
  the proxy's ``qwen3_5.py``), so it is never compared against the
  checkpoint's — only against the trust-on-first-use pin the trainer records
  (``--harbor-template-pin-file``).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: Version of the fingerprint algorithm.  Bump on any change to the corpus,
#: the canonicalization, or the component set — both repos together.
FINGERPRINT_ALGORITHM = "v1"

#: Fixed probe corpus, byte-identical in both repos.  Encoded with
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
    "2026-09-01T12:34:56.789Z +08:00 v2.7.1-rc.3",
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
    "re = r\"^\\d{3}-\\d{4}$\" # backslashes \\\\ \\n \\t",
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
    """sha256 of the probe corpus itself — proves both repos hold the same
    corpus version before their probe hashes are compared."""
    return _sha256(list(PROBE_CORPUS))


#: Pinned by tests in BOTH repos.  If you edited the corpus you must bump
#: FINGERPRINT_ALGORITHM and update this constant in both copies.
PROBE_CORPUS_SHA256 = "e9bf1e8ebc946be32acb47e6558a8e36c6b105213ae12eb4ddeda6ea6c39d919"


def _normalized_merges(merges: Any) -> list[list[str]] | None:
    """Merges as ordered pairs, whatever serialization shape they arrived in.

    ``tokenizers`` has serialized merges both as ``"a b"`` strings and as
    ``["a", "b"]`` pairs across versions; normalizing keeps the structural
    hash comparable across that difference (order is preserved — merge
    priority is positional).
    """
    if not isinstance(merges, list):
        return None
    normalized: list[list[str]] = []
    for entry in merges:
        if isinstance(entry, str):
            parts = entry.split(" ")
            normalized.append(parts if len(parts) == 2 else [entry])
        elif isinstance(entry, (list, tuple)):
            normalized.append([str(part) for part in entry])
        else:
            normalized.append([str(entry)])
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
    docstring); ``components`` carries each part plus the warn-tier
    ``structural`` hashes; ``template_sha256`` covers only the raw
    ``chat_template`` — callers with chat markers of their own (the proxy's
    profiles) hash those in via :func:`template_pin`.
    """
    probe = [
        [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]
        for text in PROBE_CORPUS
    ]
    vocab = getattr(tokenizer, "get_vocab", None)
    vocab_items = sorted((str(token), int(token_id)) for token, token_id in (vocab() if vocab else {}).items())
    added_tokens = _added_token_table(tokenizer)
    components: dict[str, Any] = {
        "probe": _sha256(probe),
        "vocab": _sha256(vocab_items),
        "vocab_size": len(vocab_items),
        "added_tokens": _sha256(added_tokens),
    }

    backend = getattr(tokenizer, "backend_tokenizer", None)
    structural: dict[str, Any] | None = None
    if backend is not None and hasattr(backend, "to_str"):
        try:
            data = json.loads(backend.to_str())
        except Exception:
            data = None
        if isinstance(data, dict):
            model = data.get("model") or {}
            structural = {
                "merges": _sha256(_normalized_merges(model.get("merges"))),
                "normalizer_chain": _sha256(
                    [
                        data.get("normalizer"),
                        data.get("pre_tokenizer"),
                        data.get("post_processor"),
                        data.get("decoder"),
                    ]
                ),
            }
    components["structural"] = structural

    fingerprint = _sha256(
        {
            "algorithm": FINGERPRINT_ALGORITHM,
            "corpus": probe_corpus_digest(),
            "probe": components["probe"],
            "vocab": components["vocab"],
            "vocab_size": components["vocab_size"],
            "added_tokens": components["added_tokens"],
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


def template_pin(template_sha256: str, generation_prompt: str, assistant_turn_end: str) -> str:
    """The pinnable hash of the serving chat format: the loaded template plus
    the profile's strict-TITO markers.  The proxy reports it as
    ``template_pin``; the trainer's recorded trust-on-first-use pin compares
    against it.  This — not equality with the checkpoint's release template —
    is the template invariant: the serving format is a deliberate adaptation,
    and the pin catches it being edited out from under a run."""
    return _sha256(
        {
            "template_sha256": template_sha256,
            "generation_prompt": generation_prompt,
            "assistant_turn_end": assistant_turn_end,
        }
    )
