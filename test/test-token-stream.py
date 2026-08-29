"""Offline test of :class:`TokenStreamManager` against every tokenizer return shape.

The incident this guards: the e2e tests used to drive the manager through a
remote HTTP tokenizer facade that always answered with a plain ``list[int]``,
while the production paths load a real ``AutoTokenizer``
(``model_registry.ModelRegistry``, resolving the model names agents send) —
and ``transformers`` >= 5 returns a ``BatchEncoding`` from
``apply_chat_template``.  Treating that mapping as a sequence yields its
*keys*, so the session's authoritative token stream silently becomes two
bogus tokens plus the completion — TITO corrupted, no exception raised.

No test covered that, because no test built the tokenizer type that breaks.
This one does: fakes for each return shape, plus the real tokenizer (default:
the bundled Qwen3.5 directory, so the section runs offline whenever
transformers is installed; override with ``$TOKENIZER_MODEL``).  The e2e
harness now tokenizes locally on the bundled tokenizers too, so it exercises
the real return shape as well.

Run:
    python test/test-token-stream.py
    TOKENIZER_MODEL=Qwen/Qwen3-0.6B python test/test-token-stream.py
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver.token_stream import (  # noqa: E402
    ASSISTANT_TURN_END,
    GENERATION_PROMPT,
    TokenStreamError,
    TokenStreamManager,
    as_token_ids,
)
from offline_common import check, load_real_tokenizer  # noqa: E402

BUNDLED_QWEN = str(Path(__file__).resolve().parents[1] / "proxyserver" / "tokenization" / "Qwen3.5")
MODEL = os.getenv("TOKENIZER_MODEL", BUNDLED_QWEN)
USER = [{"role": "user", "content": "hi"}]
TURN2 = USER + [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}]


class _FakeBatchEncoding(dict):
    """Stand-in for transformers>=5 BatchEncoding: a Mapping whose list() gives keys."""

    @property
    def input_ids(self):
        return self["input_ids"]


class _Tokenizer:
    """Returns prompts in a configurable shape, mimicking the tokenizer zoo."""

    #: Marker ids the fake returns for the manager's generation prompt
    #: (tokenized via ``encode``, like a real tokenizer).
    GEN_IDS = [70, 71]
    #: ...and for the assistant turn ending it closes the previous turn with
    #: (``<|im_end|>`` then a newline — see token_stream.ASSISTANT_TURN_END).
    END_IDS = [80, 81]

    def __init__(self, shape: str):
        self.shape = shape

    def encode(self, text, add_special_tokens=True):
        return list(self.END_IDS if text == ASSISTANT_TURN_END else self.GEN_IDS)

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=True):
        if not messages:
            raise ValueError("empty conversation")  # Qwen-style template
        ids = [100 + i for i in range(2 * len(messages))]
        if self.shape == "list":
            return ids
        if self.shape == "batch_encoding":
            return _FakeBatchEncoding(input_ids=ids, attention_mask=[1] * len(ids))
        if self.shape == "nested":
            return _FakeBatchEncoding(input_ids=[ids], attention_mask=[[1] * len(ids)])
        if self.shape == "no_input_ids":
            return _FakeBatchEncoding(tokens=ids)
        if self.shape == "strings":
            return ["<|im_start|>", "hi"]
        raise AssertionError(self.shape)


def test_as_token_ids() -> None:
    print("as_token_ids normalizes every return shape")
    check("plain list passes through", as_token_ids([1, 2, 3], what="t") == [1, 2, 3])
    check("BatchEncoding -> input_ids",
          as_token_ids(_FakeBatchEncoding(input_ids=[1, 2], attention_mask=[1, 1]), what="t") == [1, 2])
    check("batch of one is unwrapped",
          as_token_ids(_FakeBatchEncoding(input_ids=[[1, 2]]), what="t") == [1, 2])
    check("empty stays empty", as_token_ids([], what="t") == [])

    for bad, why in [
        (_FakeBatchEncoding(tokens=[1, 2]), "mapping without input_ids"),
        (_FakeBatchEncoding(input_ids=[[1], [2]]), "batch of two conversations"),
        (["<|im_start|>", "hi"], "token strings instead of ids"),
    ]:
        try:
            as_token_ids(bad, what="t")
            raise AssertionError(f"FAIL: {why} was accepted")
        except TokenStreamError:
            pass
    check("mapping without input_ids / multi-conversation batch / token strings all rejected", True)


def test_manager_shapes() -> None:
    print("\nTokenStreamManager builds identical streams for every shape")
    streams = {}
    for shape in ("list", "batch_encoding", "nested"):
        m = TokenStreamManager(_Tokenizer(shape))
        p1 = m.build_prompt("s", USER)
        check(f"[{shape}] build_prompt -> list[int]", isinstance(p1, list) and all(isinstance(i, int) for i in p1))
        m.commit("s", USER, p1, [900, 901])
        stream = m._sessions["s"].stream
        # The corruption this guards: list(BatchEncoding) == ["input_ids", "attention_mask"],
        # so the stream would be 2 keys + 2 completion tokens = 4 entries.
        check(f"[{shape}] stream = prompt + completion, not mapping keys",
              stream == p1 + [900, 901] and len(stream) == len(p1) + 2)
        p2 = m.build_prompt("s", TURN2)
        check(f"[{shape}] turn-2 prompt strictly extends the stream", p2[:len(stream)] == stream)
        streams[shape] = (p1, stream, p2)
    check("all shapes produce byte-identical streams", len({repr(v) for v in streams.values()}) == 1)


def test_new_conversation_replaces_stream() -> None:
    print("\nA request with no assistant message opens a new conversation")
    NEW = [{"role": "system", "content": "phase 2"}, {"role": "user", "content": "go"}]

    m = TokenStreamManager(_Tokenizer("list"))
    p1 = m.build_prompt("s", USER)
    m.commit("s", USER, p1, [900, 901])
    old_stream = m._sessions["s"].stream

    # The next phase of the task opens a fresh conversation under the same session id.  Must render fresh, not 400.
    p2 = m.build_prompt("s", NEW)
    check("fresh full render, not an extension", p2[:len(old_stream)] != old_stream)
    check("render matches a brand-new session's",
          p2 == TokenStreamManager(_Tokenizer("list")).build_prompt("x", NEW))

    m.commit("s", NEW, p2, [910])
    check("commit replaces the old stream", m._sessions["s"].stream == p2 + [910])

    follow_up = NEW + [{"role": "assistant", "content": "done"}, {"role": "user", "content": "next"}]
    p3 = m.build_prompt("s", follow_up)
    check("follow-up extends the new conversation", p3[:len(p2) + 1] == p2 + [910])

    # Strictness is intact for conversations that DO carry sampled content.
    try:
        m.build_prompt("s", NEW + [{"role": "assistant", "content": "done"}])
        raise AssertionError("FAIL: short assistant-carrying request accepted")
    except TokenStreamError:
        check("assistant-carrying request that does not extend still raises", True)
    try:
        TokenStreamManager(_Tokenizer("list")).build_prompt("fresh", follow_up)
        raise AssertionError("FAIL: assistant content without a stream accepted")
    except TokenStreamError:
        check("assistant content for a stream-less session still raises", True)


def test_cached_turn_reserves_lost_responses() -> None:
    print("\nAn exact repeat of the last request re-serves the cached turn")
    m = TokenStreamManager(_Tokenizer("list"))
    p1 = m.build_prompt("s", USER)
    m.commit("s", USER, p1, [900, 901], log_probs=[-0.1, -0.2], finish_reason="stop")

    check("a different conversation is no repeat", m.cached_turn("s", TURN2) is None)

    p2 = m.build_prompt("s", TURN2)
    m.commit("s", TURN2, p2, [910], log_probs=[-0.3], finish_reason="stop")

    cached = m.cached_turn("s", TURN2)
    check("verbatim repeat -> cached turn", cached is not None)
    prompt_ids, completion_ids, log_probs, finish_reason, engine_meta = cached
    check("cached prompt is the turn's full engine prompt", prompt_ids == p2)
    check("cached completion is the committed tail", completion_ids == [910])
    check("cached logprobs/finish_reason ride along",
          log_probs == [-0.3] and finish_reason == "stop")
    check("engine_meta defaults to empty when the committer supplied none",
          engine_meta == {})

    # engine_meta is cached with the turn, so a retry arriving after a weight
    # update reports the version its tokens were sampled under — not the
    # current one, which would make an off-policy turn look on-policy.
    m4 = TokenStreamManager(_Tokenizer("list"))
    v1 = m4.build_prompt("v", USER)
    m4.commit("v", USER, v1, [900], log_probs=[-0.1], finish_reason="stop",
              engine_meta={"weight_version": "step-7"})
    replayed = m4.cached_turn("v", USER)
    check("cached turn re-serves the engine_meta it was committed with",
          replayed is not None and replayed[4] == {"weight_version": "step-7"})

    edited = TURN2[:-1] + [{"role": "user", "content": "different"}]
    check("edited repeat -> no cached turn", m.cached_turn("s", edited) is None)
    check("unknown session -> no cached turn", m.cached_turn("nope", TURN2) is None)
    check("missing session_id -> no cached turn", m.cached_turn(None, TURN2) is None)
    check("a different assistant-free request -> no cached turn (fresh conversation)",
          m.cached_turn("s", USER) is None)

    # The opening request carries no assistant message, and losing its
    # response is the most expensive turn to lose (usually the longest of the
    # rollout).  A byte-identical repeat of it is a retry, not a request to
    # re-sample: re-serve it, or the SDK's automatic retry silently pays for a
    # second generation and records a branch the agent never saw.
    m3 = TokenStreamManager(_Tokenizer("list"))
    o1 = m3.build_prompt("o", USER)
    m3.commit("o", USER, o1, [900, 901], log_probs=[-0.1, -0.2], finish_reason="stop")
    opening = m3.cached_turn("o", USER)
    check("verbatim repeat of the opening request -> cached turn", opening is not None)
    check("...re-serving its exact prompt, completion and logprobs",
          opening == (o1, [900, 901], [-0.1, -0.2], "stop", {}))
    check("a different opening request still starts a fresh conversation",
          m3.cached_turn("o", [{"role": "user", "content": "other"}]) is None)

    # Committing without logprobs/finish_reason (older callers, mock workers)
    # caches nothing — the duplicate then fails loudly instead of replaying
    # a half-remembered turn.
    m2 = TokenStreamManager(_Tokenizer("list"))
    q1 = m2.build_prompt("t", USER)
    m2.commit("t", USER, q1, [900])
    q2 = m2.build_prompt("t", TURN2)
    m2.commit("t", TURN2, q2, [910])
    check("commit without logprobs/finish_reason caches nothing",
          m2.cached_turn("t", TURN2) is None)


def test_continuation_missing_the_assistant_echo_is_refused() -> None:
    print("\nA continuation that lost its assistant echo is refused, not reset")
    # build_openai_response sets content: null on every tool-call turn, so an
    # agent that filters falsy-content messages out of its history drops
    # exactly those echoes.  Read as "no assistant message anywhere" that
    # looked like a brand-new conversation, and the full re-render silently
    # discarded the sampled tokens of the turn being answered — every turn,
    # with no error, corrupting the whole rollout.
    m = TokenStreamManager(_Tokenizer("list"))
    p1 = m.build_prompt("s", USER)
    m.commit("s", USER, p1, [900], log_probs=[-0.1], finish_reason="tool_calls")

    lost_echo = USER + [{"role": "tool", "content": "result", "tool_call_id": "c1"}]
    try:
        m.build_prompt("s", lost_echo)
        raise AssertionError("FAIL: a continuation missing its assistant echo was accepted")
    except TokenStreamError as e:
        check("extending the previous request without the echo raises", True)
        check("...and the error names the missing echo", "assistant" in str(e))

    # The echoed form of the same turn is the one that must work.
    echoed = USER + [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "content": "result", "tool_call_id": "c1"},
    ]
    p2 = m.build_prompt("s", echoed)
    check("the properly echoed continuation extends the stream",
          p2[:len(p1) + 1] == p1 + [900])

    # The guard must use the same role+content equality the inter-turn delta
    # uses.  Demanding full dict equality let any extra message key ("name",
    # "tool_call_id", a provider extra) fail the prefix test, fall through to
    # "new conversation", and silently reset the stream after all.
    with_extra_key = [
        dict(USER[0], name="bob"),  # same role+content, one extra key
        {"role": "tool", "content": "result", "tool_call_id": "c1"},
    ]
    try:
        m.build_prompt("s", with_extra_key)
        raise AssertionError("FAIL: an extra message key bypassed the lost-echo guard")
    except TokenStreamError:
        check("an extra message key does not bypass the guard", True)

    # A genuinely new conversation shares no history, so it must still be
    # allowed to replace the stream.
    fresh = [{"role": "user", "content": "unrelated new task"}]
    check("an unrelated conversation still starts fresh",
          m.build_prompt("s", fresh) == m.build_prompt("other", fresh))


def test_replay_cap_breaks_the_re_issue_loop() -> None:
    print("\nRe-serving the cached turn is capped, so a re-issue loop ends")
    # A lost-response retry needs one or two replays.  A caller re-issuing the
    # identical request is instead asking to re-sample — and would be handed
    # the byte-identical completion (including the truncated one it is trying
    # to get past) forever, HTTP 200 every time, with nothing to break on.
    m = TokenStreamManager(_Tokenizer("list"), max_turn_replays=2)
    p1 = m.build_prompt("s", USER)
    m.commit("s", USER, p1, [900], log_probs=[-0.1], finish_reason="stop")
    p2 = m.build_prompt("s", TURN2)
    m.commit("s", TURN2, p2, [910], log_probs=[-0.3], finish_reason="length")

    check("the first replay is served", m.cached_turn("s", TURN2) is not None)
    check("the second replay is served", m.cached_turn("s", TURN2) is not None)
    try:
        m.cached_turn("s", TURN2)
        raise AssertionError("FAIL: unlimited replays of one committed turn")
    except TokenStreamError:
        check("past the cap the repeat is refused instead of replayed again", True)

    # Committing the next turn clears the count: the new turn is replayable.
    turn3 = TURN2 + [{"role": "assistant", "content": "ok"},
                     {"role": "user", "content": "next"}]
    p3 = m.build_prompt("s", turn3)
    m.commit("s", turn3, p3, [920], log_probs=[-0.4], finish_reason="stop")
    check("a newly committed turn is replayable again",
          m.cached_turn("s", turn3) is not None)


def test_evicted_session_says_so() -> None:
    print("\nAn LRU-evicted session explains itself instead of blaming the agent")
    m = TokenStreamManager(_Tokenizer("list"), max_sessions=1)
    p1 = m.build_prompt("victim", USER)
    m.commit("victim", USER, p1, [900], log_probs=[-0.1], finish_reason="stop")
    p2 = m.build_prompt("evictor", USER)
    m.commit("evictor", USER, p2, [900], log_probs=[-0.1], finish_reason="stop")

    try:
        m.build_prompt("victim", TURN2)
        raise AssertionError("FAIL: evicted session served")
    except TokenStreamError as e:
        # Every remaining turn of that rollout fails identically; without
        # naming eviction the message reads like the agent sent bad history.
        check("the error names eviction and max_sessions",
              "evicted" in str(e) and "max_sessions" in str(e))


def test_manager_rejects_garbage() -> None:
    print("\nA tokenizer that does not return token IDs is rejected, not streamed")
    for shape in ("no_input_ids", "strings"):
        m = TokenStreamManager(_Tokenizer(shape))
        try:
            m.build_prompt("s", USER)
            raise AssertionError(f"FAIL: [{shape}] garbage accepted")
        except TokenStreamError:
            pass
        check(f"[{shape}] raises TokenStreamError", "s" not in m._sessions)


def test_real_tokenizer() -> None:
    print(f"\nReal AutoTokenizer ({MODEL}) — the type no other test builds")
    tok = load_real_tokenizer(MODEL)
    if tok is None:
        return

    import transformers
    raw = tok.apply_chat_template(USER, add_generation_prompt=False, tokenize=True)
    print(f"  transformers {transformers.__version__}: apply_chat_template -> {type(raw).__name__}")

    m = TokenStreamManager(tok)
    p1 = m.build_prompt("s", USER)
    check("build_prompt -> list[int]", isinstance(p1, list) and all(isinstance(i, int) for i in p1))
    # The manager renders with add_generation_prompt=False and appends its
    # own header instead of the template's generation prompt (which would
    # pre-fill '<think>\n' — see token_stream.GENERATION_PROMPT).
    expected = as_token_ids(raw, what="t") + tok.encode(GENERATION_PROMPT, add_special_tokens=False)
    check("prompt = headerless render + manager's generation prompt", p1 == expected)
    check("prompt ends with the assistant header, no think pre-fill",
          tok.decode(p1, skip_special_tokens=False).endswith("<|im_end|>\n" + GENERATION_PROMPT))
    check("prompt is non-trivial (not mapping keys)", len(p1) > 2)

    completion = [900, 901, 902]
    m.commit("s", USER, p1, completion)
    stream = m._sessions["s"].stream
    check("stream = prompt + completion", stream == p1 + completion)

    p2 = m.build_prompt("s", TURN2)
    check("turn-2 prompt strictly extends the stream", p2[:len(stream)] == stream)
    check("turn-2 adds a non-empty delta", len(p2) > len(stream))
    decoded = tok.decode(p2[len(stream):])
    check(f"delta decodes to the new user message ({decoded.strip()[:40]!r})", "more" in decoded)


def test_bundled_template_tool_delta() -> None:
    print("\nBundled Qwen3.5 template: tool-result deltas keep the user header")
    tok = load_real_tokenizer(BUNDLED_QWEN)
    if tok is None:
        return

    # The strict-TITO inter-turn delta renders the new tool results *alone*
    # (token_stream._delta_ids).  The unpatched upstream template dropped the
    # '<|im_start|>user' header on such renders (loop.previtem is undefined on
    # the first message), so every tool turn's engine prompt — and the
    # recorded training stream — was malformed.  See tokenization/README.md.
    delta = tok.apply_chat_template(
        [{"role": "tool", "content": "sunny, 21C"}],
        add_generation_prompt=True, tokenize=False,
    )
    check("leading tool message opens a user block",
          delta.startswith("<|im_start|>user\n<tool_response>"))

    full = tok.apply_chat_template(
        [
            {"role": "user", "content": "weather in SF?"},
            {"role": "assistant", "content": "checking",
             "tool_calls": [{"type": "function", "function": {
                 "name": "get_weather", "arguments": {"city": "SF"}}}]},
            {"role": "tool", "content": "sunny, 21C"},
        ],
        add_generation_prompt=True, tokenize=False,
    )
    check("delta is byte-identical to the full render's tool segment",
          full.endswith(delta))

    two = tok.apply_chat_template(
        [{"role": "tool", "content": "r1"}, {"role": "tool", "content": "r2"}],
        add_generation_prompt=True, tokenize=False,
    )
    check("consecutive tool results share one user block",
          two.count("<|im_start|>user") == 1)

    # End to end through the manager: turn 2 of a tool-calling session.
    m = TokenStreamManager(tok)
    msgs1 = [{"role": "user", "content": "weather in SF?"}]
    p1 = m.build_prompt("s", msgs1)
    completion = tok("checking", add_special_tokens=False).input_ids
    m.commit("s", msgs1, p1, completion)
    p2 = m.build_prompt("s", msgs1 + [
        {"role": "assistant", "content": "checking"},
        {"role": "tool", "content": "sunny, 21C"},
    ])
    tail = tok.decode(p2[len(p1) + len(completion):], skip_special_tokens=False)
    # The completion was truncated (no sampled <|im_end|>), so the manager
    # closes the assistant message the way the template would before the tool
    # block — see ASSISTANT_TURN_END.
    check("manager's tool delta closes the assistant turn, then opens the user block",
          tail == ASSISTANT_TURN_END + "<|im_start|>user\n<tool_response>\nsunny, 21C\n"
                  "</tool_response><|im_end|>\n" + GENERATION_PROMPT)


def test_stream_matches_the_chat_template() -> None:
    """The oracle: a built stream must equal the template's own render.

    Every other TITO check here compares the manager against the *scheme* it
    implements — render the new messages, strip the implicit system prefix,
    append the generation prompt — so a flaw in the scheme itself passes them
    all.  One did: the template ends every message with ``<|im_end|>\\n``, the
    engine stops *at* the sampled ``<|im_end|>``, and the delta render starts
    fresh at ``<|im_start|>`` — so nobody supplied that newline and every
    continuation prompt read ``<|im_end|><|im_start|>user``, one token off the
    format the checkpoint was trained on, in the engine prompt *and* in the
    recorded training stream.

    This compares against an independent oracle instead: for a conversation
    whose assistant turns carry the reasoning block the model actually sampled
    (so the template has nothing to rewrite), the incrementally built stream
    must be byte-identical to ``apply_chat_template(full_conversation)`` plus
    the manager's generation prompt.
    """
    print("\nBuilt streams are byte-identical to the chat template's own render")
    tok = load_real_tokenizer(BUNDLED_QWEN)
    if tok is None:
        return

    def render(messages: list[dict]) -> list[int]:
        return as_token_ids(
            tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=True),
            what="oracle render",
        ) + tok.encode(GENERATION_PROMPT, add_special_tokens=False)

    def sampled(text: str, *, truncated: bool = False) -> list[int]:
        """Tokens a rollout engine returns: EOS included unless truncated."""
        return tok.encode(text if truncated else text + "<|im_end|>", add_special_tokens=False)

    convo = [{"role": "system", "content": "You are helpful."},
             {"role": "user", "content": "weather in SF? 你好 🎉"}]

    # The reply text differs per case because the template *rewrites*
    # assistant history depending on what follows it: before a user message it
    # strips a <think> block, before a tool result it keeps one (and injects an
    # empty one when absent).  That rewriting is exactly why strict TITO
    # exists — and it means the template is a valid oracle only where it
    # renders the sampled text verbatim, which each case asserts below.
    for label, reply, follow_up, truncated in (
        ("user follow-up", "Checking.",
         {"role": "user", "content": "and tomorrow?"}, False),
        ("tool result", "<think>\n\n</think>\n\nCalling the tool.",
         {"role": "tool", "content": "sunny, 21C", "tool_call_id": "c1"}, False),
        ("truncated turn (no sampled <|im_end|>)", "Chec",
         {"role": "user", "content": "go on"}, True),
    ):
        m = TokenStreamManager(tok)
        messages = list(convo)
        prompt = m.build_prompt(label, messages)
        check(f"[{label}] opening prompt matches the template", prompt == render(messages))

        completion = sampled(reply, truncated=truncated)
        m.commit(label, messages, prompt, completion,
                 log_probs=[0.0] * len(completion),
                 finish_reason="length" if truncated else "stop")

        # Two more turns, so the gap is exercised on a stream that already
        # carries one — the delta scheme must stay lossless as it compounds.
        for turn in (2, 3):
            messages = messages + [{"role": "assistant", "content": reply}, dict(follow_up)]
            expected = render(messages)
            check(
                f"[{label}] the template renders turn-{turn - 1}'s sampled reply "
                f"verbatim (so it is a valid oracle)",
                tok.decode(expected, skip_special_tokens=False).count(reply) == turn - 1,
            )
            prompt = m.build_prompt(label, messages)
            check(f"[{label}] turn-{turn} prompt matches the template", prompt == expected)
            completion = sampled(reply)
            m.commit(label, messages, prompt, completion,
                     log_probs=[0.0] * len(completion), finish_reason="stop")


def main() -> None:
    test_as_token_ids()
    test_manager_shapes()
    test_new_conversation_replaces_stream()
    test_cached_turn_reserves_lost_responses()
    test_continuation_missing_the_assistant_echo_is_refused()
    test_replay_cap_breaks_the_re_issue_loop()
    test_evicted_session_says_so()
    test_manager_rejects_garbage()
    test_real_tokenizer()
    test_bundled_template_tool_delta()
    test_stream_matches_the_chat_template()
    print("\n" + "=" * 70)
    print("PASS: TokenStreamManager accepts list[int] and transformers>=5\n"
          "      BatchEncoding alike, builds identical token streams from\n"
          "      both, rejects non-token-ID renders instead of letting mapping\n"
          "      keys corrupt the session's authoritative stream, and builds\n"
          "      streams byte-identical to the chat template's own render.")
    print("=" * 70)


if __name__ == "__main__":
    main()
