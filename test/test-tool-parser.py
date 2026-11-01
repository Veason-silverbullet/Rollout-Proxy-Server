"""Offline test of the built-in tool-call parsers (``proxyserver.tokenization.tool_parser``).

The direct-mode incident this guards: the agent looped for multiple turns
because the proxy ran with tool parsing disabled — the model's
``<tool_call><function=task_done>...`` XML went back to the agent as plain
text with ``tool_calls: null``, so the agent never saw a tool call.  The
built-in parsers serve the profile selected by each model's entry in
``tokenization/mapping.json``, and these tests pin their behavior:

* qwen3_coder (the Qwen3.x / Qwen3-Coder template dialect) and hermes JSON
  blocks both parse into OpenAI-shaped calls (``.name`` + JSON-string
  ``.arguments``);
* the pre-filled-``<think>`` reasoning is stripped from content *before*
  parsing, so hypothetical calls inside the reasoning are never executed;
* unparseable/truncated blocks stay in the content instead of vanishing;
* ``build_tool_parser`` selects a built-in parser or the null parser;
* every ``tokenization/mapping.json`` entry resolves to a bundled tokenizer
  directory and a built-in parser, with the mapping fixed at startup.

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
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver import model_registry
from proxyserver.model_registry import ModelRegistry, build_tool_parser, load_model_mapping, resolve_profile
from proxyserver.tokenization.base import ThinkReasoning, get_profile
from proxyserver.tokenization.tool_parser import (SUPPORTED_TOOL_CALL_PARSERS,
                                                  HermesToolParser, Llama3JsonToolParser, NullToolParser, Qwen3CoderToolParser,
                                                  _split_sections, get_tool_parser, resolve_parser_class)
from offline_common import check, load_real_tokenizer

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
    check("JSON values recovered, strings kept", args == {"ids": [1, 2], "flag": True, "status": "completed", "note": "line one\nline two"})

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

    # An unfinished block is still reasoning; raw text is retained in the
    # completion record, but must not enter the tool parser.
    content, _ = extract(Qwen3CoderToolParser, "done\n</think>\n\nAnswer.\n<think>\ncut off")
    check("an unterminated later <think> is excluded from content",
          content == "Answer.")

    hypothetical_calls = [
        (Qwen3CoderToolParser,
         '<tool_call><function=bash><parameter=command>echo hypothetical</parameter></function></tool_call>'),
        (HermesToolParser,
         '<tool_call>{"name":"bash","arguments":{"command":"echo hypothetical"}}</tool_call>'),
        (Llama3JsonToolParser,
         '{"name":"bash","parameters":{"command":"echo hypothetical"}}'),
    ]
    for parser_cls, call in hypothetical_calls:
        content, calls = extract(parser_cls, "Before.<think>" + call + "</think>After.")
        check(f"{parser_cls.NAME}: first explicit reasoning block preserves visible prefix",
              content == "Before.After." and calls == [])
        for prefix, expected in [("", ""), ("done</think>Answer.\n", "Answer."), ("Answer.\n", "Answer.")]:
            content, calls = extract(parser_cls, prefix + "<think>" + call)
            check(f"{parser_cls.NAME}: unfinished reasoning cannot dispatch calls ({prefix!r})",
                  calls == [] and content == expected)
        # Actual calls outside reasoning remain available, even if a later
        # unfinished block includes a second hypothetical call.
        content, calls = extract(parser_cls, "done</think>" + call + "<think>" + call)
        check(f"{parser_cls.NAME}: actual call survives unfinished later reasoning",
              content == "" and len(calls) == 1 and calls[0].name == "bash")


def test_reasoning_with_tool_tag_mentions() -> None:
    print("\nmentions of tool tags do not hide the end of reasoning")
    registry = ModelRegistry()
    call = '<tool_call><function=run><parameter=command>pwd</parameter></function></tool_call>'
    for model in ("Qwen3.5-9B", "Qwen3.6-27B", "Qwen3.8-27B"):
        bundle = registry.resolve(model)
        for thought in ("Use the <tool_call> format.",
                        "Use `<tool_call>` after planning.",
                        "Start with <tool_call>\nthen name the function.",
                        "Emit <tool_call>, not another <tool_call> mention.",
                        "Use `<tool_call><function=run>` to issue the command.",
                        'A JSON tool call starts with `<tool_call>{"name":`.',
                        'A JSON name starts with `<tool_call>{"name":"`.',
                        'The incomplete JSON key is `<tool_call>{"na`.',
                        'Use `<tool_call>{"name":"run","arguments":` for the call.',
                        "Use `<tool_call><function=run><parameter=command>` to issue a command.",
                        'Use "<tool_call><function=run><parameter=command>" to issue a command.',
                        "Use '<tool_call><function=run><parameter=command>' to issue a command.",
                        "Use “<tool_call><function=run><parameter=command>” to issue a command.",
                        "I’ll use ‘<tool_call><function=run><parameter=command>’ to issue a command.",
                        "使用「<tool_call><function=run><parameter=command>」来运行命令。",
                        "The call must start with <tool_call><function=run><parameter=command> before I supply the command.",
                        "Example header:\n\n    <tool_call><function=run><parameter=command>\n\nNow run the command.",
                        "Use <tool_call><function=run><parameter=command> and then <tool_call><function=run><parameter=command>.",
                        'Use `<tool_call>{"name":"run","arguments":"` to issue a command.',
                        'Use `<tool_call>{"name":"run","arguments":{"command":"` to issue a command.',
                        "The incomplete header is <tool_call><function=run"):
            for opener in ("<think>", ""):
                for answer in (call, "Done."):
                    completion = opener + thought + "</think>\n\n" + answer
                    ids = bundle.tokenizer.encode(completion, add_special_tokens=False)
                    content, calls = asyncio.run(bundle.tool_parser.extract_tool_calls(ids))
                    check(f"{model}: {opener!r} reasoning mentioning tags preserves the answer",
                          (content == "" and len(calls) == 1 and calls[0].name == "run"
                           and json.loads(calls[0].arguments) == {"command": "pwd"})
                          if answer == call else content == "Done." and not calls)
                    check("reasoning extraction ends at the actual closing marker",
                          bundle.reasoning.extract(completion) == thought)
            completion = "Answer.<think>" + thought + "</think>\n\n" + call
            ids = bundle.tokenizer.encode(completion, add_special_tokens=False)
            content, calls = asyncio.run(bundle.tool_parser.extract_tool_calls(ids))
            check("a later reasoning block mentioning tags preserves surrounding content and calls",
                  content == "Answer." and len(calls) == 1 and calls[0].name == "run")

    # The shared reasoning parser also serves Hermes: recognize its JSON
    # prefix while keeping literal reasoning tags in argument strings intact.
    call = '<tool_call>{"name":"run","arguments":{"command":"echo <think>literal</think>"}}</tool_call>'
    for thought in ("Use the <tool_call> format.",
                    "Use `<tool_call><function=run>` to issue the command.",
                    'A JSON tool call starts with `<tool_call>{"name":`.',
                    'A JSON name starts with `<tool_call>{"name":"`.',
                    'The incomplete JSON key is `<tool_call>{"na`.',
                    'Use `<tool_call>{"name":"run","arguments":` for the call.',
                    "Use `<tool_call><function=run><parameter=command>` to issue a command.",
                    "The XML header is <tool_call><function=run><parameter=command> before the value.",
                    'Use `<tool_call>{"name":"run","arguments":"` to issue a command.',
                    'Use `<tool_call>{"name":"run","arguments":{"command":"` to issue a command.'):
        for opener in ("<think>", ""):
            completion = opener + thought + "</think>\n\n" + call
            content, calls = extract(HermesToolParser, completion)
            check("Hermes keeps its call and literal argument after a tag mention",
                  content == "" and len(calls) == 1 and calls[0].name == "run"
                  and json.loads(calls[0].arguments) == {"command": "echo <think>literal</think>"})


def test_qwen_literal_reasoning_in_arguments() -> None:
    print("\nQwen profiles preserve literal reasoning tags inside tool arguments")
    registry = ModelRegistry()
    tools = [{"type": "function", "function": {"name": "write_file", "parameters": {
        "type": "object", "properties": {"content": {"type": "string"}},
    }}}]
    for model in ("Qwen3.5-9B", "Qwen3.6-27B", "Qwen3.8-27B"):
        bundle = registry.resolve(model)
        def parse(text):
            ids = bundle.tokenizer.encode(text, add_special_tokens=False)
            return asyncio.run(bundle.tool_parser.extract_tool_calls(ids, tools=tools))

        for value in ("A<think>literal</think>B", "<think>unclosed", "orphan</think>",
                      "“<think>literal</think>” and ‘another’", "I’ll keep <think>literal</think>",
                      '\'echo "<think>literal</think>"\'', "Don't change <think>literal</think>",
                      "<think>上海</think>\n<think>more</think>",
                      "`echo <think>literal</think>`", "```xml\n<think>literal</think>\n```",
                      "```xml\n<think>literal</think>",
                      "<tool_call><function=example><parameter=text><think>literal</think>"):
            call = ("<tool_call><function=write_file><parameter=content>\n"
                    + value + "\n</parameter></function></tool_call>")
            for prefix, reasoning in (("", None), ("<think>plan</think>\n\n", "plan"), ("plan</think>\n\n", "plan")):
                completion = prefix + call + call
                content, calls = parse(completion)
                check(f"{model}: literal {value!r} with prefix {prefix!r}",
                      content == "" and len(calls) == 2
                      and all(json.loads(c.arguments) == {"content": value} for c in calls))
                check("argument tags never become reasoning_content", bundle.reasoning.extract(completion) == reasoning)

            # Tool bodies stay opaque inside reasoning too: a closing tag in
            # a hypothetical argument must not end the actual reasoning block.
            for prefix in ("<think>consider ", "consider "):
                completion = prefix + call + " maybe</think>\n\nDone."
                content, calls = parse(completion)
                check("hypothetical calls remain hidden with explicit or prefilled reasoning",
                      content == "Done." and calls == []
                      and bundle.reasoning.extract(completion) == "consider " + call + " maybe")
            content, calls = parse(call + "<think>consider " + call)
            check("a real call survives later unfinished reasoning",
                  content == "" and len(calls) == 1
                  and json.loads(calls[0].arguments) == {"content": value})
            partial = "<tool_call><function=write_file><parameter=content>" + value
            check("truncated tool bodies retain their literal tags",
                  bundle.reasoning.strip(partial) == partial
                  and bundle.reasoning.extract(partial) is None)

        # Once actual calls begin, a literal closing think followed by a
        # nested opening header must not trigger unfinished-header recovery.
        value = "orphan</think>\n<tool_call><function=example><parameter=text>literal"
        header = "<tool_call><function=write_file><parameter=content>"
        call = header + value + "</parameter></function></tool_call>"
        for prefix, thought in (("", None), ("\n  ", None), ("<think>plan</think>", "plan"),
                                ("plan</think>", "plan")):
            completion = prefix + call + call
            content, calls = parse(completion)
            check("actual calls preserve closing think tags followed by nested opening markup",
                  content == "" and len(calls) == 2
                  and all(json.loads(c.arguments) == {"content": value} for c in calls)
                  and bundle.reasoning.extract(completion) == thought)
            completion = prefix + header + value
            check("truncated actual arguments cannot become unfinished reasoning headers",
                  bundle.reasoning.strip(completion).strip() == header + value
                  and bundle.reasoning.extract(completion) == thought)


def test_hermes_literal_reasoning_in_arguments() -> None:
    print("\nHermes preserves literal reasoning tags in JSON argument values")
    reasoning = ThinkReasoning()
    arguments = {
        "content": 'A<think>literal</think>B with "quotes" and \\backslashes\\',
        "nested": [{"</think>": "<think>unclosed"}, ["orphan</think>",
                   "<tool_call><function=example><parameter=text><think>literal</think>"]],
        "code": "`echo <think>literal</think>`\n```xml\n<think>literal</think>\n```",
        "unfinished_code": "```xml\n<think>literal</think>",
    }
    for key in ("arguments", "parameters"):
        for value in (arguments, json.dumps(arguments)):
            call = "<tool_call>" + json.dumps({"name": "write_file", key: value}) + "</tool_call>"
            for prefix, expected in (("", None), ("<think>plan</think>\n\n", "plan"),
                                     ("plan</think>\n\n", "plan")):
                completion = prefix + call + call
                content, calls = extract(HermesToolParser, completion)
                check("JSON escapes and nested argument strings survive in multiple calls",
                      content == "" and len(calls) == 2
                      and all(json.loads(c.arguments) == arguments for c in calls)
                      and reasoning.extract(completion) == expected)
            for prefix in ("<think>consider ", "consider "):
                completion = prefix + call + " maybe</think>\n\nDone."
                content, calls = extract(HermesToolParser, completion)
                check("hypothetical JSON calls stay inside explicit and prefilled reasoning",
                      content == "Done." and calls == []
                      and reasoning.extract(completion) == "consider " + call + " maybe")
            completion = call + "<think>consider " + call + "</think>\n\nDone."
            content, calls = extract(HermesToolParser, completion)
            check("reasoning resumes after a JSON call's final argument",
                  content == "Done." and len(calls) == 1
                  and json.loads(calls[0].arguments) == arguments)
            content, calls = extract(HermesToolParser, call + "<think>consider " + call)
            check("unfinished later reasoning preserves only the actual JSON call",
                  content == "" and len(calls) == 1
                  and json.loads(calls[0].arguments) == arguments)

    for start in ('<tool_call>{"name":"write_file","arguments":{"content":"',
                  '<tool_call>{"name":"write_file","arguments":"'):
        for value in ("<think>unclosed", "orphan</think>", '<think>literal</think>\\"quoted\\"',
                      "<think>literal</think>\\"):
            partial = start + value
            content, calls = extract(HermesToolParser, partial)
            check("truncated JSON strings retain their literal tags and escapes",
                  content == partial and calls == [] and reasoning.extract(partial) is None)
            check("an unfinished hypothetical JSON call stays in reasoning",
                  reasoning.strip("<think>consider " + partial) == "")


def test_reasoning_with_code_examples() -> None:
    print("\nquoted tool examples cannot consume reasoning boundaries")
    reasoning = ThinkReasoning()
    dialects = (
        (Qwen3CoderToolParser, '<tool_call><function=run><parameter=command>',
         '<tool_call><function=run><parameter=command>pwd</parameter></function></tool_call>'),
        (HermesToolParser, '<tool_call>{"name":"run","arguments":"',
         '<tool_call>{"name":"run","arguments":{"command":"pwd"}}</tool_call>'),
        (HermesToolParser, '<tool_call>{"name":"run","arguments":{"command":"',
         '<tool_call>{"name":"run","arguments":{"command":"pwd"}}</tool_call>'),
    )
    for parser_cls, header, call in dialects:
        example = call.replace("pwd", "echo <think>literal</think>")
        thoughts = [
            "Use `" + header + "` to issue a command.",
            "Use ``" + header + " `example` `` to issue a command.",
            "Use `\n" + header + "\n` to issue a command.",
            "Consider `" + example + "` before acting.",
            "Consider `" + header + "` and then `" + header + "`.",
            "An unmatched ` in prose does not hide the boundary.",
            "An unmatched fence:\n```xml\nstill reasoning",
        ]
        for quote, close_quote in (("'", "'"), ('"', '"'), ("“", "”"), ("‘", "’"),
                                   ("«", "»"), ("‹", "›"), ("„", "“"), ("‚", "‘"),
                                   ("「", "」"), ("『", "』")):
            thoughts.extend([
                "Use " + quote + header + close_quote + " to issue a command.",
                "Use " + quote + "example:\n" + header + "\n" + close_quote + " to issue a command.",
                "Consider " + quote + example + close_quote + " before acting.",
                "Consider " + quote + header + close_quote + " and then " + quote + header + close_quote + ".",
                "Consider " + quote + header + close_quote + ", then " + quote + header + close_quote + ".",
                "I'll use " + quote + header + close_quote + " when I'm ready.",
                "I’ll use " + quote + "the tool’s prefix: " + header + close_quote + " when I’m ready.",
                "The tools' syntax is " + quote + header + close_quote + ".",
                'The 12" display shows ' + quote + header + close_quote + ".",
                "Use " + quote + "example: \\" + close_quote + header + close_quote + " to issue a command.",
            ])
            if quote != close_quote:
                thoughts.append("使用" + quote + header + close_quote + "来运行命令。")
        for opener, closer in (("```xml", "```"), ("````xml", "`````"), ("~~~json", "~~~")):
            thoughts.append("Example:\n" + opener + "\n" + header + "\n" + closer + "\nNow act.")
            thoughts.append("Example:\n" + opener + "\n" + example + "\n" + closer + "\nNow act.")
            thoughts.append(opener + "\n" + header + "\n" + closer)
        if parser_cls is Qwen3CoderToolParser:
            thoughts.append("Use `<tool_call>`, then `<function=run>`, then `<parameter=command>`.")
        for thought in thoughts:
            for prefix in ("<think>", "", "Before.<think>"):
                expected = "Before." if prefix == "Before.<think>" else ""
                completion = prefix + thought + "</think>\n\n" + call
                content, calls = extract(parser_cls, completion)
                check(f"{parser_cls.NAME}: quoted examples preserve only the actual command",
                      content == expected and len(calls) == 1 and calls[0].name == "run"
                      and json.loads(calls[0].arguments) == {"command": "pwd"}
                      and reasoning.extract(completion) == thought)
                content, calls = extract(parser_cls, prefix + thought + "</think>\n\nDone.")
                check("quoted examples preserve plain answers too",
                      content == expected + "Done." and calls == [])
            content, calls = extract(parser_cls, "<think>" + thought + call)
            check("quoted examples cannot dispatch calls from unfinished reasoning", content == "" and calls == [])

        for delimiter in ("`", "```", "'", '"', "“", "‘", "「"):
            thought = "Unfinished quoted example:\n" + delimiter + "\nstill planning"
            value = "echo " + delimiter + "<think>literal</think>" + delimiter
            real = call.replace("pwd", value if parser_cls is Qwen3CoderToolParser else json.dumps(value)[1:-1])
            for prefix in ("<think>", ""):
                completion = prefix + thought + "</think>\n\n" + real
                content, calls = extract(parser_cls, completion)
                check("reasoning boundaries survive unmatched quotes followed by quoted arguments",
                      content == "" and len(calls) == 1
                      and json.loads(calls[0].arguments) == {"command": value}
                      and reasoning.extract(completion) == thought)

        # A fence in a real argument is data, including an unmatched opener.
        # It must not steal a later quoted example's closing fence.
        real = call.replace("pwd", "\n```xml\n\n" if parser_cls is Qwen3CoderToolParser else "```xml\\n")
        thought = "Example:\n```xml\n" + header + "\n```\nNow act."
        completion = real + "\n<think>" + thought + "</think>\n\n" + call
        content, calls = extract(parser_cls, completion)
        check("backticks in actual arguments cannot interfere with later reasoning examples",
              content == "" and len(calls) == 2
              and [json.loads(c.arguments) for c in calls] == [{"command": "```xml\n"}, {"command": "pwd"}]
              and reasoning.extract(completion) == thought)

        for quote, close_quote in (("'", "'"), ('"', '"'), ("“", "”"), ("‘", "’"), ("「", "」")):
            value = "Unmatched " + quote + " <think>literal</think>"
            real = call.replace("pwd", value if parser_cls is Qwen3CoderToolParser else json.dumps(value)[1:-1])
            thought = "Use " + quote + header + close_quote + "."
            completion = real + "<think>" + thought + "</think>" + call
            content, calls = extract(parser_cls, completion)
            check("quotes in actual arguments cannot steal a later example's delimiter",
                  content == "" and len(calls) == 2
                  and [json.loads(c.arguments) for c in calls] == [{"command": value}, {"command": "pwd"}]
                  and reasoning.extract(completion) == thought)


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
    json.loads(calls[0].arguments, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
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


def test_qwen3_coder_incomplete_first_parameter() -> None:
    print("\nqwen3_coder distinguishes empty arguments from an incomplete first parameter")

    for body in (
        "<parameter=command>incomplete",
        "<parameter=command",
        "<parameter=>value</parameter>",
        "unexpected text",
    ):
        block = f"<tool_call><function=run>{body}</function></tool_call>"
        text = f"Before.\n{block}\nAfter."
        content, calls = extract(Qwen3CoderToolParser, text)
        check(f"malformed body emits no call ({body!r})", calls == [])
        check("malformed block and surrounding content are preserved", content == text)

    for body in ("", "\n \t\n"):
        content, calls = extract(Qwen3CoderToolParser, f"<tool_call><function=run>{body}</function></tool_call>")
        check("empty or whitespace-only body remains a parameterless call",
              content == "" and len(calls) == 1
              and calls[0].name == "run" and json.loads(calls[0].arguments) == {})

    malformed = "<function=run><parameter=command>incomplete</function>"
    block = f"<tool_call><function=ready></function>{malformed}</tool_call>"
    content, calls = extract(Qwen3CoderToolParser, block)
    check("a malformed function rejects the whole containing block",
          content == block and calls == [])

    block = f"<tool_call>{malformed}</tool_call>"
    content, calls = extract(Qwen3CoderToolParser, block + "\n<tool_call><function=ready></function></tool_call>")
    check("a separate valid block still parses after a malformed block",
          content == block and [c.name for c in calls] == ["ready"])


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

    # Bound characters searched and copied, not just outer parser calls:
    # malformed names used to rescan and slice a shared suffix per opener.
    class MeasuredText(str):
        work = 0

        def find(self, sub, start=0, end=None):
            end = len(self) if end is None else end
            result = super().find(sub, start, end)
            self.work += max(0, (end if result < 0 else result + len(sub)) - start)
            return result

        def __getitem__(self, key):
            value = super().__getitem__(key)
            if isinstance(key, slice):
                self.work += len(value)
            return value

    for marker, closer in (("<parameter=", "</parameter>"), ("<function=", "</function>")):
        for n in (4000, 32768):
            for suffix, expected in ((">", []), (f"{marker}valid>body{closer}", [("valid", "body")])):
                text = MeasuredText((marker + "\n") * n + suffix)
                sections, well_formed = _split_sections(text, marker, closer)
                check(f"{n} malformed {marker} names preserve recovery", sections == expected and well_formed)
                check("scanning and copying remain bounded by input length", text.work <= 8 * len(text))


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


def test_glm_tool_parser() -> None:
    bundle = ModelRegistry().resolve("GLM-5.3-Flash")
    tools = [{"type": "function", "function": {
        "name": "lookup", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "count": {"type": "integer"},
            "options": {"type": "object"}, "enabled": {"type": "boolean"},
        }},
    }}]
    call_text = "<tool_call>lookup<arg_key>city</arg_key><arg_value>上海</arg_value></tool_call>"
    parser = bundle.tool_parser
    pairs = {"city": "001", "count": "3", "options": '{"x": [1, 2]}', "enabled": "true"}
    block = "<tool_call>lookup" + "".join(
        f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in pairs.items()
    ) + "</tool_call>"
    content, calls = parser.parse("before" + block + "<tool_call>ping</tool_call>after", tools)
    check("GLM parallel and zero-argument calls preserve surrounding content",
          content == "beforeafter" and [c.name for c in calls] == ["lookup", "ping"])
    check("GLM arguments use declared string types and JSON for containers/scalars",
          json.loads(calls[0].arguments) == {
              "city": "001", "count": 3, "options": {"x": [1, 2]}, "enabled": True,
          } and json.loads(calls[1].arguments) == {})
    value = "\n  first\nsecond  \n"
    _, calls = parser.parse(f"<tool_call>lookup<arg_key>city</arg_key><arg_value>{value}</arg_value></tool_call>", tools)
    check("GLM string arguments retain whitespace", json.loads(calls[0].arguments)["city"] == value)
    for raw in (
        "<tool_call>lookup<arg_key>city</arg_key><arg_value>cut off",
        "<tool_call>lookup<arg_key>city</arg_key></tool_call>",
        "<tool_call>lookup<arg_key>city</arg_key><arg_value>x</arg_value>stranded</tool_call>",
        "<tool_call>lookup<arg_key>x</arg_key><arg_value>1</arg_value><arg_key>x</arg_key><arg_value>2</arg_value></tool_call>",
    ):
        content, calls = parser.parse(raw, tools)
        check("damaged GLM call stays content without partial dispatch", content == raw and not calls)
    for ending in ("</think>", ""):
        completion = "<think>look up the city" + ending + call_text + "<|observation|>"
        ids = bundle.tokenizer.encode(completion, add_special_tokens=False)
        content, calls = asyncio.run(parser.extract_tool_calls(ids, tools=tools))
        check(f"real GLM tokens parse tools after reasoning ending {ending!r}",
              content == "" and len(calls) == 1 and calls[0].name == "lookup"
              and json.loads(calls[0].arguments) == {"city": "上海"})
        check("GLM extracts reasoning without including the tool call",
              bundle.reasoning.extract(completion) == "look up the city")
    completion = "<think>plan" + call_text + "<tool_call>ping</tool_call>"
    ids = bundle.tokenizer.encode(completion, add_special_tokens=False)
    content, calls = asyncio.run(parser.extract_tool_calls(ids, tools=tools))
    check("GLM retains parallel calls after an implicit reasoning ending",
          content == "" and [c.name for c in calls] == ["lookup", "ping"])
    partial = "<tool_call>lookup<arg_key>city</arg_key><arg_value>cut off"
    ids = bundle.tokenizer.encode("<think>plan" + partial, add_special_tokens=False)
    content, calls = asyncio.run(parser.extract_tool_calls(ids, tools=tools))
    check("GLM retains a truncated tool call after an implicit reasoning ending",
          content == partial and not calls)
    for value in ("<think>literal</think>", "<think>literal", "literal</think>",
                  "\n <think>上海</think>\n<think>unfinished "):
        literal_call = f"<tool_call>lookup<arg_key>city</arg_key><arg_value>{value}</arg_value></tool_call>"
        for prefix, reasoning in (("", None), ("<think>plan", "plan"), ("<think>plan</think>", "plan")):
            completion = prefix + literal_call + "<tool_call>ping</tool_call><|observation|>"
            ids = bundle.tokenizer.encode(completion, add_special_tokens=False)
            content, calls = asyncio.run(parser.extract_tool_calls(ids, tools=tools))
            check(f"GLM retains literal argument {value!r} after prefix {prefix!r}",
                  content == "" and [c.name for c in calls] == ["lookup", "ping"]
                  and json.loads(calls[0].arguments) == {"city": value})
            check("GLM never extracts reasoning from a tool argument",
                  bundle.reasoning.extract(completion) == reasoning)


def test_parser_selection() -> None:
    print("\nparser resolution and built-in parser selection")
    tok = _TextTokenizer("")
    check("qwen3_coder resolves", isinstance(get_tool_parser("qwen3_coder", tok), Qwen3CoderToolParser))
    check("hermes resolves", isinstance(get_tool_parser("hermes", tok), HermesToolParser))
    check("parser names are case/space-insensitive", resolve_parser_class(" Hermes ") is HermesToolParser)
    try:
        resolve_parser_class("no_such_parser")
        raise AssertionError("FAIL: unknown parser name accepted")
    except ValueError:
        check("unknown parser name raises ValueError", True)

    check("named parser -> built-in parser", isinstance(build_tool_parser("qwen3_coder", tok), Qwen3CoderToolParser))
    check("no tool_call_parser -> null parser (content cleaned, no calls)", isinstance(build_tool_parser(None, tok), NullToolParser))
    content, calls = extract(
        NullToolParser,
        "planning\n</think>\n\nanswer <tool_call>left as-is</tool_call>\n",
    )
    check("null parser strips reasoning, extracts nothing", calls == [] and content == "answer <tool_call>left as-is</tool_call>")
    check("supported parsers advertised",
          set(SUPPORTED_TOOL_CALL_PARSERS) == {"hermes", "llama3_json", "qwen3_coder", "glm47", "gpt_oss"})
    check("llama3_json resolves", isinstance(get_tool_parser("llama3_json", tok), Llama3JsonToolParser))


def test_mapping() -> None:
    print("\ntokenization/mapping.json entries resolve to built-in parsers")
    # Exercise the production validators, not copies of them: registry
    # construction runs the eager profile check (no tokenizer loads),
    # and resolve_profile raises on a missing tokenizer directory.
    ModelRegistry()
    check("registry construction validates every mapping entry eagerly", True)
    mapping = load_model_mapping()
    check("mapping lists served models", len(mapping) > 0)
    for model in mapping:
        profile = resolve_profile(model)
        parser_name = profile.tool_call_parser.NAME if profile.tool_call_parser is not None else "disabled"
        check(f"{model}: resolves (tokenizer={Path(profile.tokenizer_path).name}, "
              f"tool_call_parser={parser_name})", True)


def test_mapping_is_fixed_after_startup() -> None:
    print("\nModel mapping remains fixed for the proxy process")
    with TemporaryDirectory() as directory:
        source = Path(directory) / "mapping.json"
        source.write_text(json.dumps({"original-model": "qwen3_5"}))
        with patch.object(model_registry, "MODEL_MAPPING_PATH", source):
            load_model_mapping.cache_clear()
            try:
                ModelRegistry()
                source.write_text(json.dumps({"new-model": "llama3_1"}))
                source.unlink()
                check("existing model resolves without rereading the mapping file",
                      resolve_profile("original-model").name == "qwen3_5")
                try:
                    resolve_profile("new-model")
                except model_registry.UnknownModelError:
                    check("mapping edits do not add models to the running proxy", True)
                else:
                    raise AssertionError("A mapping edit changed the running proxy")
                ModelRegistry()
                check("another registry in the same process shares the startup mapping", load_model_mapping() == {"original-model": "qwen3_5"})
            finally:
                load_model_mapping.cache_clear()


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


async def test_gpt_oss_parsing() -> None:
    bundle = ModelRegistry().resolve("gpt-oss-20b")
    profile = get_profile("gpt_oss")
    tools = [{"type": "function", "function": {
        "name": "lookup", "description": "Look up a value",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
    }}]
    call = ('<|channel|>analysis<|message|>Need a lookup.<|end|>'
            '<|start|>assistant<|channel|>commentary<|message|>Checking.<|end|>'
            '<|start|>assistant to=functions.lookup<|channel|>commentary'
            '<|constrain|>json<|message|>{"key": "x"}<|call|>')
    final = ('<|channel|>analysis<|message|>Use the result.<|end|>'
             '<|start|>assistant<|channel|>final<|message|>The value is 42.<|return|>')
    tok, parser = bundle.tokenizer, bundle.tool_parser
    async def parse(text):
        return await parser.extract_tool_calls(tok.encode(text, add_special_tokens=False), tools=tools)
    content, calls = await parse(call)
    check("analysis, visible preamble, recipient and JSON arguments are separated",
          content == "Checking." and len(calls) == 1 and calls[0].name == "lookup"
          and calls[0].arguments == '{"key": "x"}' and profile.reasoning.extract(call) == "Need a lookup.")
    for header in (" to=functions.lookup<|channel|>analysis", "<|channel|>commentary to=functions.lookup",
                   " to=functions.lookup<|channel|>commentary json"):
        content, calls = await parse(header + '<|message|>{"key":"<think>literal</think>"}<|call|>')
        check("both recipient positions and analysis-channel tools parse without rewriting arguments",
              content == "" and len(calls) == 1 and json.loads(calls[0].arguments)["key"] == "<think>literal</think>")
    content, calls = await parse(final)
    check("final answer excludes Harmony and analysis", content == "The value is 42." and not calls
          and profile.reasoning.extract(final) == "Use the result.")
    content, calls = await parse('<|channel|>final<|message|>{"name":"Alice"}<|return|>')
    check("ordinary JSON stays content", content == '{"name":"Alice"}' and not calls)
    analysis = '<|channel|>analysis<|message|>one<|end|><|start|>assistant<|channel|>analysis<|message|>two'
    check("multiple and truncated analysis blocks stay reasoning", profile.reasoning.extract(analysis) == "one\n\ntwo"
          and (await parse(analysis)) == ("", []))
    for text in ('<|channel|>anal', ' to=functions.look', '<|channel|>analysis<|message|>unfinished'):
        check("truncated header/reasoning does not leak", (await parse(text)) == ("", []))
    for args, ending in (('{"key":', ''), ('{"key":"x"}', ''), ('{"key":"x"}', '<|return|>'),
                         ('{"key":NaN}', '<|call|>'), ('{"key":[1e999]}', '<|call|>'), ('[]', '<|call|>')):
        content, calls = await parse(' to=functions.lookup<|channel|>commentary<|message|>' + args + ending)
        check("incomplete/malformed calls stay content without execution", content == args and not calls)
    content, calls = await parse('<|channel|>final<|message|>part')
    check("truncated final content survives", content == "part" and not calls)


async def test_gpt_oss_literal_markers() -> None:
    bundle = ModelRegistry().resolve("gpt-oss-20b")
    tok, parser = bundle.tokenizer, bundle.tool_parser
    encode = lambda text: tok.encode(text, add_special_tokens=False)

    def literal(text):
        # Ordinary vocabulary tokens can spell any Harmony marker without
        # containing its structural token ID. Preserve that distinction.
        ids = [token for char in text for token in encode(char)]
        assert tok.decode(ids) == text
        assert not set(ids) & {199999, 200002, 200003, 200005, 200006, 200007, 200008, 200012}
        return ids

    example = ('<|start|>assistant to=functions.lookup<|channel|>commentary'
               '<|message|>{"key":"x"}<|call|>')
    body = 'Literal markers: <|end|> <|return|> <|endoftext|> <|constrain|>\nExample: ' + example
    for prefix in ("", "<|start|>assistant"):
        for ending in ("", "<|return|>"):
            ids = encode(prefix + '<|channel|>final<|message|>') + literal(body) + encode(ending)
            check("literal boundaries and quoted calls stay intact in final content",
                  await parser.extract_tool_calls(ids) == (body, []))
            check("quoted headers cannot fabricate reasoning",
                  bundle.reasoning.extract_tokens(ids, tok) is None)

    args = json.dumps({"key": body}, ensure_ascii=False)
    ids = encode(' to=functions.lookup<|channel|>commentary<|message|>') + literal(args) + encode('<|call|>')
    content, calls = await parser.extract_tool_calls(ids)
    check("genuine handoffs preserve literal markers inside JSON arguments",
          content == "" and len(calls) == 1 and calls[0].name == "lookup" and calls[0].arguments == args)

    ids = ids[:-1] + literal('<|call|>')
    check("a literal handoff spelling does not complete a truncated call",
          await parser.extract_tool_calls(ids) == (args + '<|call|>', []))
    ids = literal(' to=functions.lookup<|channel|>commentary<|message|>{"key":"x"}<|call|>')
    check("literal header delimiters cannot create a tool call",
          await parser.extract_tool_calls(ids) == ("", []))

    ids = encode('<|channel|>analysis<|message|>') + literal(body)
    ids += encode('<|end|><|start|>assistant<|channel|>final<|message|>done<|return|>')
    check("literal markers stay within analysis without truncation or dispatch",
          await parser.extract_tool_calls(ids) == ("done", [])
          and bundle.reasoning.extract_tokens(ids, tok) == body)

    from proxyserver.tokenization import harmony

    with patch.object(harmony, "assistant_messages_from_tokens", wraps=harmony.assistant_messages_from_tokens) as scan:
        decoded = await parser.extract_completion(ids, completion_text=tok.decode(ids))
        check("Harmony decodes content and reasoning in one structural pass",
              scan.call_count == 1 and decoded.content == "done"
              and decoded.reasoning_content == body and decoded.tool_calls == [])


def main() -> None:
    test_qwen3_coder()
    test_qwen3_coder_ddb_regression()
    test_reasoning_is_fully_stripped()
    test_reasoning_with_tool_tag_mentions()
    test_qwen_literal_reasoning_in_arguments()
    test_hermes_literal_reasoning_in_arguments()
    test_reasoning_with_code_examples()
    test_qwen3_coder_argument_typing()
    test_qwen3_coder_truncated_blocks()
    test_qwen3_coder_incomplete_first_parameter()
    test_repetition_loop_is_linear()
    test_hermes()
    test_llama3_json()
    test_glm_tool_parser()
    asyncio.run(test_gpt_oss_parsing())
    asyncio.run(test_gpt_oss_literal_markers())
    test_parser_selection()
    test_mapping()
    test_mapping_is_fixed_after_startup()
    test_real_tokenizer()
    print("\n" + "=" * 70)
    print("PASS: built-in parsers turn qwen3_coder / hermes / llama3_json\n"
          "      completions into OpenAI tool_calls, strip reasoning before\n"
          "      parsing, keep unparseable blocks visible, and back\n"
          "      the model registry with a mapping fixed at startup.")
    print("=" * 70)


if __name__ == "__main__":
    main()
