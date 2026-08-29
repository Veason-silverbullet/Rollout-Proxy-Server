"""Per-model profiles: strict TITO beyond ChatML, stated not inferred.

Each model family is described by a ``ModelProfile``
(``proxyserver/tokenization/{qwen3_5,llama3_1}.py``) that *states* its
strict-TITO chat markers (generation prompt, assistant turn end), its
tool-call parser, and its reasoning format — so a new family is served by a
``mapping.json`` entry pointing at a small profile module, with no change to
the shared core.  ``mapping.json`` maps each served model name to a profile.

The load-bearing check is :func:`test_profile_stream_matches_template`: driving
``TokenStreamManager`` with the markers a profile *states*, the incrementally
built stream must be byte-identical to the model's own chat-template render —
for ChatML (Qwen) and Llama alike.  The implicit per-render prefix a template
prepends (Llama's bos + dated default system block) is measured and stripped
(:meth:`TokenStreamManager._measure_implicit_prefix`).
"""

from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from offline_common import check, load_real_tokenizer  # noqa: E402
from proxyserver.model_registry import ModelRegistry  # noqa: E402
from proxyserver.token_stream import TokenStreamManager, as_token_ids  # noqa: E402
from proxyserver.tokenization.base import (  # noqa: E402
    NoReasoning,
    ThinkReasoning,
    get_profile,
)
from proxyserver.tokenization.tool_parser import (  # noqa: E402
    Llama3JsonToolParser,
    Qwen3CoderToolParser,
    split_reasoning,
)

_TOKENIZATION = Path(__file__).resolve().parents[1] / "proxyserver" / "tokenization"


def test_profiles_load() -> None:
    print("\nProfiles load with stated markers, parser, and reasoning")
    qwen = get_profile("qwen3_5")
    check("qwen3_5 states ChatML markers",
          (qwen.generation_prompt, qwen.assistant_turn_end) == ("<|im_start|>assistant\n", "<|im_end|>\n"))
    check("qwen3_5 tokenizer dir / parser / reasoning",
          qwen.tokenizer_dir == "Qwen3.5" and qwen.tool_call_parser is Qwen3CoderToolParser
          and isinstance(qwen.reasoning, ThinkReasoning))
    llama = get_profile("llama3_1")
    check("llama3_1 states Llama markers",
          (llama.generation_prompt, llama.assistant_turn_end)
          == ("<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>"))
    check("llama3_1 tokenizer dir / parser / reasoning",
          llama.tokenizer_dir == "Llama-3.1" and llama.tool_call_parser is Llama3JsonToolParser
          and isinstance(llama.reasoning, NoReasoning))
    try:
        get_profile("no_such_profile")
        check("bad profile name raises", False)
    except ValueError:
        check("bad profile name raises loudly (not a silent default)", True)


def test_reasoning_parsers() -> None:
    print("\nReasoning parsers own the <think> logic (one place, not scattered)")
    samples = ["<think>r</think>c", "r</think>c", "<think>\n\n</think>\n\nok",
               "plain content", "<think>a</think>x<think>b</think>y", "<think>truncated"]
    think = ThinkReasoning()
    # strip == the compat split_reasoning shim (which now delegates here)
    check("ThinkReasoning.strip matches split_reasoning", all(think.strip(s) == split_reasoning(s) for s in samples))
    check("ThinkReasoning.extract pulls the leading think block",
          think.extract("<think>reasoning</think>answer") == "reasoning")
    check("ThinkReasoning.extract is None without a closed block", think.extract("<think>truncated") is None)
    check("ThinkReasoning.extract is None for an empty block", think.extract("<think>\n\n</think>\n\nx") is None)
    none = NoReasoning()
    check("NoReasoning strips nothing / extracts nothing",
          all(none.strip(s) == s and none.extract(s) is None for s in samples))


def test_response_carries_reasoning_content() -> None:
    print("\nThe OpenAI response surfaces reasoning_content per the pinned model's profile")
    from proxyserver.server import build_openai_response, build_openai_stream_response

    def message(model: str, completion_text: str, **kw):
        body = build_openai_response(
            token_ids=[1], content_text="answer", tool_calls=kw.get("tool_calls"),
            finish_reason=kw.get("finish_reason", "stop"), model=model,
            completion_text=completion_text,
        )
        return body, body["choices"][0]["message"]

    _, msg = message("Qwen3.5-9B", "<think>weighing it</think>answer")
    check("a <think> model's assistant turn carries reasoning_content",
          msg["reasoning_content"] == "weighing it" and msg["content"] == "answer")
    _, msg = message("Qwen3.5-9B", "answer")
    check("no thinking block -> no reasoning_content field", "reasoning_content" not in msg)
    _, msg = message("Llama-3.1-8B-Instruct", "<think>x</think>answer")
    check("a no-reasoning model never carries it", "reasoning_content" not in msg)
    _, msg = message("mystery-model", "<think>x</think>answer")
    check("an unmapped model omits it rather than guessing a format",
          "reasoning_content" not in msg)

    tc = [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    body, msg = message("Qwen3.5-9B", "<think>pick f</think><tool_call>...</tool_call>",
                        tool_calls=tc, finish_reason="tool_calls")
    check("a tool-call turn carries it alongside tool_calls",
          msg["reasoning_content"] == "pick f" and msg["tool_calls"] == tc)

    async def sse_events(resp):
        return [e async for e in resp.body_iterator]

    events = asyncio.run(sse_events(build_openai_stream_response(body)))
    first_delta = json.loads(events[0].removeprefix("data: "))["choices"][0]["delta"]
    check("the streaming delta mirrors reasoning_content",
          first_delta["reasoning_content"] == "pick f")


def test_registry_wires_the_profile() -> None:
    print("\nModelRegistry builds a bundle straight from the profile")
    reg = ModelRegistry()
    qwen = reg.resolve("Qwen3.5-9B")
    check("qwen bundle carries the profile's ChatML markers",
          (qwen.token_streams.generation_prompt, qwen.token_streams.assistant_turn_end)
          == ("<|im_start|>assistant\n", "<|im_end|>\n"))
    check("qwen bundle: qwen3_coder parser + <think> reasoning",
          isinstance(qwen.tool_parser, Qwen3CoderToolParser) and isinstance(qwen.reasoning, ThinkReasoning))
    llama = reg.resolve("Llama-3.1-8B-Instruct")
    check("llama bundle carries the profile's Llama markers",
          (llama.token_streams.generation_prompt, llama.token_streams.assistant_turn_end)
          == ("<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>"))
    check("all Qwen3.5-* names share one bundle", reg.resolve("Qwen3.5-27B") is qwen)


def test_implicit_prefix_measured() -> None:
    print("\nThe implicit per-render prefix is measured (Qwen: none, Llama: bos+system)")
    qwen = load_real_tokenizer(str(_TOKENIZATION / "Qwen3.5"))
    if qwen is not None:
        m = TokenStreamManager(qwen)  # ChatML defaults
        check("Qwen has no implicit render prefix", m._get_sys_ids() == [] and m._sys_ids_measured)
    llama = load_real_tokenizer(str(_TOKENIZATION / "Llama-3.1"))
    if llama is not None:
        p = get_profile("llama3_1")
        m = TokenStreamManager(llama, generation_prompt=p.generation_prompt, assistant_turn_end=p.assistant_turn_end)
        prefix = m._get_sys_ids()
        decoded = llama.decode(prefix, skip_special_tokens=False) if prefix else ""
        check("Llama's implicit prefix is measured (bos + a system block)",
              len(prefix) > 0 and m._sys_ids_measured
              and "<|begin_of_text|>" in decoded and "<|start_header_id|>system" in decoded)


def _oracle(profile_name: str, tokenizer_dir_name: str) -> None:
    """Byte-identity oracle: a stream built with the profile's *stated* markers
    must equal the template's own render, turn after turn."""
    tok = load_real_tokenizer(str(_TOKENIZATION / tokenizer_dir_name))
    if tok is None:
        return
    profile = get_profile(profile_name)
    gen_prompt = profile.generation_prompt
    eos = profile.assistant_turn_end.rstrip("\n")  # the engine samples the EOS, not a trailing newline

    def render(messages: list[dict]) -> list[int]:
        return as_token_ids(
            tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=True),
            what="oracle render",
        ) + tok.encode(gen_prompt, add_special_tokens=False)

    def sampled(text: str, *, truncated: bool = False) -> list[int]:
        return tok.encode(text if truncated else text + eos, add_special_tokens=False)

    # The Qwen template *rewrites* assistant history by what follows it: before
    # a tool result it keeps (and injects an empty) <think> block, before a
    # user message it strips one.  So the tool-result reply must already carry
    # the empty block a think-model's template renders there — otherwise the
    # sampled tokens and the template's render differ and the oracle is invalid.
    # A no-reasoning model (Llama) has no such rewriting.
    think = "<think>\n\n</think>\n\n" if isinstance(profile.reasoning, ThinkReasoning) else ""
    convo = [{"role": "system", "content": "You are helpful."},
             {"role": "user", "content": "weather in SF? 你好 🎉"}]
    for label, reply, follow_up, truncated in (
        ("user follow-up", "Let me check.", {"role": "user", "content": "and tomorrow?"}, False),
        ("tool result", think + "Calling the tool.", {"role": "tool", "content": "sunny, 21C", "tool_call_id": "c1"}, False),
        ("truncated turn", "Let me che", {"role": "user", "content": "go on"}, True),
    ):
        mgr = TokenStreamManager(tok, generation_prompt=gen_prompt, assistant_turn_end=profile.assistant_turn_end)
        messages = list(convo)
        prompt = mgr.build_prompt(label, messages)
        check(f"[{profile_name}/{label}] opening prompt matches the template", prompt == render(messages))
        completion = sampled(reply, truncated=truncated)
        mgr.commit(label, messages, prompt, completion, log_probs=[0.0] * len(completion),
                   finish_reason="length" if truncated else "stop")
        for turn in (2, 3):
            messages = messages + [{"role": "assistant", "content": reply}, dict(follow_up)]
            expected = render(messages)
            check(f"[{profile_name}/{label}] turn-{turn-1} sampled reply renders verbatim (valid oracle)",
                  tok.decode(expected, skip_special_tokens=False).count(reply) == turn - 1)
            prompt = mgr.build_prompt(label, messages)
            check(f"[{profile_name}/{label}] turn-{turn} prompt matches the template", prompt == expected)
            completion = sampled(reply)
            mgr.commit(label, messages, prompt, completion, log_probs=[0.0] * len(completion), finish_reason="stop")


def test_profile_stream_matches_template() -> None:
    print("\nBuilt streams are byte-identical to the template's own render (profile markers)")
    _oracle("qwen3_5", "Qwen3.5")
    _oracle("llama3_1", "Llama-3.1")


def main() -> None:
    test_profiles_load()
    test_reasoning_parsers()
    test_response_carries_reasoning_content()
    test_registry_wires_the_profile()
    test_implicit_prefix_measured()
    test_profile_stream_matches_template()
    print("\n" + "=" * 70)
    print("PASS: model profiles state each family's chat markers, tool-call\n"
          "      parser, and reasoning format in one place; the registry builds\n"
          "      a bundle straight from the profile; and streams built with a\n"
          "      profile's stated markers are byte-identical to the template's\n"
          "      own render — ChatML/Qwen and Llama alike.")
    print("=" * 70)


if __name__ == "__main__":
    main()
