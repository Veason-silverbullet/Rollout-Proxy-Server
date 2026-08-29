"""Offline test of the built-in tool-call parsers (``proxyserver.tokenization.tool_parser``).

The direct-mode incident this guards: the agent looped for multiple turns
because the proxy ran with tool parsing disabled — the model's
``<tool_call><function=task_done>...`` XML went back to the agent as plain
text with ``tool_calls: null``, so the agent never saw a tool call.  The
built-in parsers serve each model's ``tool_call_parser`` from
``tokenization/mapping.json`` whenever no framework factory is supplied,
and these tests pin their behavior:

* qwen3_coder (the Qwen3.5 / Qwen3-Coder template dialect) and hermes JSON
  blocks both parse into OpenAI-shaped calls (``.name`` + JSON-string
  ``.arguments``);
* the pre-filled-``<think>`` reasoning is stripped from content *before*
  parsing, so hypothetical calls inside the reasoning are never executed;
* unparseable/truncated blocks stay in the content instead of vanishing;
* ``build_tool_parser`` falls back to the built-ins without a factory;
* every ``tokenization/mapping.json`` entry resolves to a bundled tokenizer
  directory and a built-in parser.

Run:
    python test/test-tool-parser.py
    TOKENIZER_MODEL=<hf-name-or-path> python test/test-tool-parser.py
"""

from __future__ import annotations
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver.model_registry import ModelRegistry, build_tool_parser,  load_model_mapping, resolve_model_spec
from proxyserver.tokenization.tool_parser import (SUPPORTED_TOOL_CALL_PARSERS,  # noqa: E402
                                                  HermesToolParser, Llama3JsonToolParser,
                                                  NullToolParser, Qwen3CoderToolParser,
                                                  get_tool_parser, resolve_parser_class)
from offline_common import check, load_real_tokenizer  # noqa: E402

# Default to the bundled Qwen3.5 tokenizer so the real-tokenizer section runs
# offline whenever transformers is installed.
MODEL = os.getenv(
    "TOKENIZER_MODEL",
    str(Path(__file__).resolve().parents[1] / "proxyserver" / "tokenization" / "Qwen3.5"),
)


class _TextTokenizer:
    """decode() replays a canned string; the parser only ever decodes."""

    def __init__(self, text: str):
        self.text = text

    def decode(self, token_ids, skip_special_tokens=False):
        assert skip_special_tokens, "parsers must decode with skip_special_tokens=True"
        return self.text


def extract(parser_cls, text, tools=None):
    parser = parser_cls(_TextTokenizer(text))
    return asyncio.run(parser.extract_tool_calls([1, 2, 3], tools=tools))


def test_qwen3_coder() -> None:
    print("qwen3_coder parses the Qwen3.5 template dialect")

    content, calls = extract(
        Qwen3CoderToolParser,
        "planning the call\n</think>\n\nChecking the weather.\n\n"
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nSF\n</parameter>\n"
        "</function>\n</tool_call>",
    )
    check("reasoning stripped, content kept", content == "Checking the weather.")
    check("one call parsed", len(calls) == 1 and calls[0].name == "get_weather")
    check("arguments JSON-encoded", json.loads(calls[0].arguments) == {"city": "SF"})

    _, calls = extract(
        Qwen3CoderToolParser,
        "<tool_call>\n<function=configure>\n"
        "<parameter=ids>\n[1, 2]\n</parameter>\n"
        "<parameter=flag>\ntrue\n</parameter>\n"
        "<parameter=status>\ncompleted\n</parameter>\n"
        "<parameter=note>\nline one\nline two\n</parameter>\n"
        "</function>\n</tool_call>",
    )
    args = json.loads(calls[0].arguments)
    check("JSON values recovered, strings kept",
          args == {"ids": [1, 2], "flag": True, "status": "completed", "note": "line one\nline two"})

    _, calls = extract(
        Qwen3CoderToolParser,
        "<tool_call>\n<function=a>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=b>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
    )
    check("multiple blocks -> multiple calls", [c.name for c in calls] == ["a", "b"])
    check("parameterless call -> empty arguments", json.loads(calls[0].arguments) == {})

    content, calls = extract(
        Qwen3CoderToolParser,
        "cut off mid-call\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls",
    )
    check("truncated block yields no call", calls == [])
    check("truncated block stays in content", "<function=bash>" in content)

    content, calls = extract(
        Qwen3CoderToolParser,
        "maybe like <tool_call>\n<function=rm>\n</function>\n</tool_call>?\n</think>\n\nDone.",
    )
    check("calls inside reasoning are not executed", calls == [] and content == "Done.")

    content, calls = extract(Qwen3CoderToolParser, "just a plain answer\n")
    check("plain text passes through", content == "just a plain answer" and calls == [])


def test_qwen3_coder_ddb_regression() -> None:
    print("\nqwen3_coder parses the exact task_done call DDB never received")
    completion = (
        "The iteration limit has been reached. Let me make one final task_done call.\n"
        "</think>\n\n"
        "<tool_call>\n"
        "<function=task_done>\n"
        "<parameter=task_summary>\nCreated 5-day weather app specification for web platform\n</parameter>\n"
        "<parameter=task_name>\nrewrite_query\n</parameter>\n"
        "<parameter=key_files>\n[]\n</parameter>\n"
        "<parameter=completion_status>\ncompleted\n</parameter>\n"
        "<parameter=final_answer>\n"
        '{"type": "rewrite", "content": "# 5-Day Weather Forecast App"}\n'
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    content, calls = extract(Qwen3CoderToolParser, completion)
    check("task_done extracted", len(calls) == 1 and calls[0].name == "task_done")
    args = json.loads(calls[0].arguments)
    check("all five parameters recovered",
          args["task_name"] == "rewrite_query" and args["key_files"] == []
          and args["completion_status"] == "completed"
          and args["final_answer"]["type"] == "rewrite")
    check("no free text left besides the call", content == "")


def test_reasoning_is_fully_stripped() -> None:
    print("\nreasoning is stripped wherever it appears, not just at the front")

    # A checkpoint that reasons again mid-completion.  Stripping only the
    # leading block would leak the raw reasoning to the agent *and* run the
    # tool call the model was only weighing up inside it.
    content, calls = extract(
        Qwen3CoderToolParser,
        "<think>\nplanning\n</think>\n\nLet me think more.\n"
        "<think>\nmaybe I should <tool_call>\n<function=rm_rf>\n"
        "<parameter=path>\n/\n</parameter>\n</function>\n</tool_call>\n</think>\n\n"
        "Actually no.",
    )
    check("a call considered inside a later think block is not executed", calls == [])
    check("no reasoning markup leaks into the content",
          "<think>" not in content and content == "Let me think more.\nActually no.")

    # An unterminated opener is truncation, not reasoning to hide: the text
    # after it is the only content there is.
    content, _ = extract(Qwen3CoderToolParser, "done\n</think>\n\nAnswer.\n<think>\ncut off")
    check("an unterminated later <think> keeps its text (truncation)",
          content == "Answer.\n<think>\ncut off")


def test_qwen3_coder_argument_typing() -> None:
    print("\nqwen3_coder types arguments by the declared tool schema")

    tools = [{"type": "function", "function": {"name": "notify", "parameters": {
        "type": "object", "properties": {
            "phone": {"type": "string"},
            "version": {"type": "string"},
            "retries": {"type": "integer"},
            "tags": {"type": "array"},
            "note": {"type": ["string", "null"]},
        }}}}]
    completion = (
        "t\n</think>\n\n<tool_call>\n<function=notify>\n"
        "<parameter=phone>\n1234567890\n</parameter>\n"
        "<parameter=version>\n1.0\n</parameter>\n"
        "<parameter=retries>\n3\n</parameter>\n"
        '<parameter=tags>\n["a", "b"]\n</parameter>\n'
        "<parameter=note>\nnull\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    _, calls = extract(Qwen3CoderToolParser, completion, tools=tools)
    args = json.loads(calls[0].arguments)
    # The template renders every scalar through |string, so the text alone
    # cannot say whether "1234567890" was an int or a phone number.  Guessing
    # wrong fails the agent's own schema validation, and the model — seeing an
    # unchanged world — re-issues the identical call forever.
    check("a declared string parameter is never retyped",
          args["phone"] == "1234567890" and args["version"] == "1.0"
          and args["note"] == "null")
    check("declared non-string parameters are decoded",
          args["retries"] == 3 and args["tags"] == ["a", "b"])

    _, calls = extract(Qwen3CoderToolParser, completion)
    args = json.loads(calls[0].arguments)
    check("without schemas the JSON-first fallback still applies",
          args["phone"] == 1234567890 and args["note"] is None)

    # json.loads accepts NaN/Infinity and json.dumps writes them back bare,
    # which every strict JSON reader on the agent side rejects.
    _, calls = extract(
        Qwen3CoderToolParser,
        "t\n</think>\n\n<tool_call>\n<function=f>\n"
        "<parameter=a>\nNaN\n</parameter>\n"
        "<parameter=b>\n-Infinity\n</parameter>\n"
        "</function>\n</tool_call>",
    )
    check("non-finite constants stay strings, keeping arguments strict JSON",
          json.loads(calls[0].arguments) == {"a": "NaN", "b": "-Infinity"})
    json.loads(calls[0].arguments,
               parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    check("arguments parse under a strict (NaN-rejecting) JSON reader", True)


def test_qwen3_coder_truncated_blocks() -> None:
    print("\nqwen3_coder refuses partially recoverable calls")

    # The dialect has no escaping, so a value containing "</tool_call>" ends
    # the block early.  Emitting what parsed would hand the agent a
    # write_file call with no content — an error the model cannot diagnose
    # from the tool's reply, so it re-issues the same call.
    content, calls = extract(
        Qwen3CoderToolParser,
        "t\n</think>\n\n<tool_call>\n<function=write_file>\n"
        "<parameter=path>\nt.py\n</parameter>\n"
        "<parameter=content>\nS = '<tool_call>\\n<function=x>\\n</function>\\n</tool_call>'\n"
        "</parameter>\n</function>\n</tool_call>",
    )
    check("a block cut short mid-parameter yields no call", calls == [])
    check("...and stays in the content instead of losing a parameter",
          "<function=write_file>" in content)

    # A value that merely *mentions* the markup closes nothing, so it must
    # still parse — in full.  Rejecting it (an earlier marker-counting check
    # did) leaves a documentation-writing agent with no tool call at all,
    # which is the retry loop this whole guard exists to prevent.
    _, calls = extract(
        Qwen3CoderToolParser,
        "t\n</think>\n\n<tool_call>\n<function=write_file>\n<parameter=content>\n"
        "Docs: a call uses <parameter=name> ... to pass args.\n</parameter>\n"
        "</function>\n</tool_call>",
    )
    check("a value mentioning the markup still parses, with its value intact",
          len(calls) == 1
          and json.loads(calls[0].arguments)["content"].endswith("to pass args."))

    # ...but a value containing a matched <parameter=…></parameter> pair does
    # close a section early, stranding its tail outside the markup.
    _, calls = extract(
        Qwen3CoderToolParser,
        "t\n</think>\n\n<tool_call>\n<function=edit>\n<parameter=content>\n"
        "Use <parameter=x>value</parameter> syntax\n</parameter>\n"
        "</function>\n</tool_call>",
    )
    check("a value that closes a section early yields no call (was: silent truncation)",
          calls == [])

    # A value containing "</parameter>" ends that parameter early.  Emitting
    # the call would write a silently truncated value — the tool reports
    # success on the wrong content, which is worse than not calling it.
    content, calls = extract(
        Qwen3CoderToolParser,
        "t\n</think>\n\n<tool_call>\n<function=edit>\n"
        "<parameter=body>\nsee </parameter> tag\n</parameter>\n"
        "</function>\n</tool_call>",
    )
    check("a value that ends its own parameter early yields no call", calls == [])
    check("...and the whole block reaches the agent intact",
          "see </parameter> tag" in content)

    # The retry that recovers a real call behind an abandoned opener must not
    # also invent one out of markup sitting inside a parameter value.
    _, calls = extract(
        Qwen3CoderToolParser,
        "t\n</think>\n\n<tool_call>\n<tool_call>\n<function=f>\n"
        "<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>",
    )
    check("a call behind an abandoned opener is still recovered",
          [c.name for c in calls] == ["f"])


def test_repetition_loop_is_linear() -> None:
    print("\na completion looping on <tool_call> costs O(1) parses per block")
    # "<tool_call>" is a single token, so a repetition-looping completion
    # emits one opener per sampled token.  Re-scanning from inside each block
    # made this quadratic: ~6s of GIL-held CPU for one 32k-token completion.
    #
    # Asserted by counting _parse_block calls rather than wall clock: the
    # invariant is "a bounded number of block parses per close marker", and
    # a timing threshold on a sub-millisecond baseline is pure noise on a
    # loaded machine.
    class _Counting(Qwen3CoderToolParser):
        calls = 0

        def _parse_block(self, block, param_types):
            type(self).calls += 1
            return super()._parse_block(block, param_types)

    for n in (4000, 32768):
        _Counting.calls = 0
        text = "<tool_call>" * n + "</tool_call>"
        content, calls = extract(_Counting, text)
        check(f"{n} openers, 1 close marker -> {_Counting.calls} block parse(s), no call invented",
              _Counting.calls <= 2 and calls == [])
    check("the whole completion is still handed to the agent",
          content.startswith("<tool_call>") and content.endswith("</tool_call>"))


def test_hermes() -> None:
    print("\nhermes parses JSON blocks")
    content, calls = extract(
        HermesToolParser,
        'thinking\n</think>\n\nOn it.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "SF"}}\n</tool_call>',
    )
    check("call parsed", len(calls) == 1 and calls[0].name == "get_weather")
    check("dict arguments JSON-encoded", json.loads(calls[0].arguments) == {"city": "SF"})
    check("content kept", content == "On it.")

    content, calls = extract(HermesToolParser, "<tool_call>\nnot json at all\n</tool_call>")
    check("malformed JSON yields no call, stays in content",
          calls == [] and "not json at all" in content)

    _, calls = extract(
        HermesToolParser,
        '<tool_call>\n{"name": "x", "arguments": "{\\"already\\": \\"encoded\\"}"}\n</tool_call>',
    )
    check("string arguments pass through", json.loads(calls[0].arguments) == {"already": "encoded"})

    # OpenAI's contract is that ``arguments`` decodes to an object; "null" or
    # "[1, 2]" would make the json.loads(...)["key"] every tool dispatcher
    # does raise, instead of the tool simply reporting a bad call.
    for block in ('{"name": "f", "arguments": null}', '{"name": "f", "arguments": [1, 2]}'):
        _, calls = extract(HermesToolParser, f"<tool_call>\n{block}\n</tool_call>")
        check(f"non-object arguments become an empty object ({block})",
              json.loads(calls[0].arguments) == {})


def test_llama3_json() -> None:
    print("\nllama3_json parses a bare JSON object (no wrapper)")
    # Llama emits the call as the whole turn: {"name", "parameters"}.
    content, calls = extract(
        Llama3JsonToolParser,
        '{"name": "get_weather", "parameters": {"city": "SF", "days": 3}}',
    )
    check("call parsed", len(calls) == 1 and calls[0].name == "get_weather")
    check("parameters JSON-encoded as arguments", json.loads(calls[0].arguments) == {"city": "SF", "days": 3})
    check("tool-call turn leaves no content", content == "")

    # Plain prose is not JSON -> stays content, no fabricated call.
    content, calls = extract(Llama3JsonToolParser, "The weather in SF is sunny, 21C.")
    check("prose stays content, no calls", calls == [] and content == "The weather in SF is sunny, 21C.")

    # A JSON object without a name is not a tool call -> kept as content.
    content, calls = extract(Llama3JsonToolParser, '{"result": 42}')
    check("nameless JSON is not a tool call", calls == [] and content == '{"result": 42}')

    # Parallel calls as a JSON array.
    _, calls = extract(
        Llama3JsonToolParser,
        '[{"name": "a", "parameters": {}}, {"name": "b", "parameters": {"x": 1}}]',
    )
    check("array of calls parsed", [c.name for c in calls] == ["a", "b"])

    # Reasoning stripping still runs before parse (shared base pipeline); a
    # non-finite constant is refused so arguments stay strict-JSON valid.
    _, calls = extract(Llama3JsonToolParser, '{"name": "f", "parameters": {"x": NaN}}')
    check("NaN refused -> not a valid tool call, kept as content", calls == [])


def test_factory() -> None:
    print("\nparser resolution and the build_tool_parser fallback")
    tok = _TextTokenizer("")
    check("qwen3_coder resolves", isinstance(get_tool_parser("qwen3_coder", tok), Qwen3CoderToolParser))
    check("hermes resolves", isinstance(get_tool_parser("hermes", tok), HermesToolParser))
    check("parser names are case/space-insensitive",
          resolve_parser_class(" Hermes ") is HermesToolParser)
    try:
        resolve_parser_class("no_such_parser")
        raise AssertionError("FAIL: unknown parser name accepted")
    except ValueError:
        check("unknown parser name raises ValueError", True)

    check("no factory -> built-in parser",
          isinstance(build_tool_parser("qwen3_coder", tok), Qwen3CoderToolParser))
    check("no tool_call_parser -> null parser (content cleaned, no calls)",
          isinstance(build_tool_parser(None, tok), NullToolParser))
    content, calls = extract(
        NullToolParser,
        "planning\n</think>\n\nanswer <tool_call>left as-is</tool_call>\n",
    )
    check("null parser strips reasoning, extracts nothing",
          calls == [] and content == "answer <tool_call>left as-is</tool_call>")
    # A framework's parser still serves the name — adapted to the proxy's
    # contract, so callers never have to ask which kind of parser they hold.
    # verl's ToolParser takes token_ids alone; the adapter drops `tools`.
    class _FrameworkParser:
        seen: list = []

        async def extract_tool_calls(self, token_ids):
            self.seen.append(token_ids)
            return "framework content", []

    built = build_tool_parser("anything", tok, lambda name, t: _FrameworkParser())
    content, calls = asyncio.run(built.extract_tool_calls([9, 9], tools=[{"name": "x"}]))
    check("a framework parser serves the name, adapted to the tools signature",
          content == "framework content" and calls == [] and _FrameworkParser.seen == [[9, 9]])
    check("supported parsers advertised",
          set(SUPPORTED_TOOL_CALL_PARSERS) == {"hermes", "llama3_json", "qwen3_coder"})
    check("llama3_json resolves", isinstance(get_tool_parser("llama3_json", tok), Llama3JsonToolParser))


def test_mapping() -> None:
    print("\ntokenization/mapping.json entries resolve without a factory")
    # Exercise the production validators, not copies of them: registry
    # construction runs the eager parser-name check (no tokenizer loads),
    # and resolve_model_spec raises on a missing tokenizer directory.
    ModelRegistry()
    check("registry construction validates every mapping entry eagerly", True)
    specs = load_model_mapping()
    check("mapping lists served models", len(specs) > 0)
    for model in specs:
        spec = resolve_model_spec(model)
        check(f"{model}: resolves (tokenizer={Path(spec.tokenizer_path).name}, "
              f"tool_call_parser={spec.tool_call_parser or 'disabled'})", True)


def test_real_tokenizer() -> None:
    print(f"\nReal tokenizer ({MODEL}) — special tokens dropped, markup survives")
    tok = load_real_tokenizer(MODEL)
    if tok is None:
        return

    completion = (
        "reasoning here\n</think>\n\nCalling now.\n\n"
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nSF\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    ids = tok(completion, add_special_tokens=False).input_ids
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(eos, int) and eos >= 0:
        ids = ids + [eos]  # the engine's stop token rides along, as in production

    parser = get_tool_parser("qwen3_coder", tok)
    content, calls = asyncio.run(parser.extract_tool_calls(ids))
    check("call parsed from real token IDs", len(calls) == 1 and calls[0].name == "get_weather")
    check("arguments recovered", json.loads(calls[0].arguments) == {"city": "SF"})
    check("reasoning and <|im_end|> absent from content", content == "Calling now.")


def main() -> None:
    test_qwen3_coder()
    test_qwen3_coder_ddb_regression()
    test_reasoning_is_fully_stripped()
    test_qwen3_coder_argument_typing()
    test_qwen3_coder_truncated_blocks()
    test_repetition_loop_is_linear()
    test_hermes()
    test_llama3_json()
    test_factory()
    test_mapping()
    test_real_tokenizer()
    print("\n" + "=" * 70)
    print("PASS: built-in parsers turn qwen3_coder / hermes / llama3_json\n"
          "      completions into OpenAI tool_calls, strip reasoning before\n"
          "      parsing, keep unparseable blocks visible, and back\n"
          "      build_tool_parser whenever no framework factory is supplied.")
    print("=" * 70)


if __name__ == "__main__":
    main()
