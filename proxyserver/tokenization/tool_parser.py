"""Built-in tool-call parsers for local mode (no training framework in the path).

The rollout providers turn a sampled completion into OpenAI ``tool_calls``
through a parser bound to the session's tokenizer
(:func:`proxyserver.model_registry.build_tool_parser`).  A training framework
may supply its own factory (e.g. verl's ``ToolParser.get_tool_parser``); when
none is given, this module provides the fallback so that standalone
deployments (``transport_mode: direct`` / ``slime``) still return structured
tool calls instead of raw text — an agent that dispatches on
``message.tool_calls`` never sees the call otherwise and stalls the trial.

Supported parsers (the ``tool_call_parser`` field of each model's entry in
``tokenization/mapping.json``):

* ``hermes`` — ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``
  JSON blocks (Hermes/Qwen2.5 templates; also verl's default format).
* ``llama3_json`` — a bare ``{"name": ..., "parameters": {...}}`` JSON object
  (or array of them), the whole assistant turn, no wrapper (Llama 3.x
  templates).
* ``qwen3_coder`` — the XML dialect of Qwen3.5 /
  Qwen3-Coder chat templates (vLLM's ``qwen3_coder`` parser)::

      <tool_call>
      <function=get_weather>
      <parameter=city>
      SF
      </parameter>
      </function>
      </tool_call>

Parsing contract: ``await parser.extract_tool_calls(token_ids, tools=...)``
returns ``(content_text, calls)`` where each call has ``.name`` and
``.arguments`` (a JSON-encoded string).  ``tools`` is the request's OpenAI
tool list, used to type each argument by its declared schema; it is
optional, and a training framework's own parser (verl's ``ToolParser``,
which takes ``token_ids`` alone) is adapted to this signature by
:class:`proxyserver.model_registry._FrameworkToolParser`.

Behavior notes, all deliberate:

* Tokens are decoded with ``skip_special_tokens=True`` — special tokens such
  as ``<|im_end|>`` never reach the agent-visible content.  The markup the
  parsers read (``<tool_call>``, ``<think>``) consists of *non-special* added
  tokens, which survive that decode.
* Reasoning is stripped from the content before tool-call parsing: Qwen3.5
  completions open with a think block the model samples itself
  (``<think>reasoning</think>content``; a template may also pre-fill the
  opener, arriving as ``reasoning</think>content``).  *Every* complete
  ``<think>...</think>`` span is removed, not just the leading one — a
  checkpoint that reasons again mid-completion would otherwise leak raw
  reasoning to the agent and, worse, have the tool calls it merely
  *considered* there executed.  The raw tokens (and therefore the recorded
  training data) keep everything.
* A block that fails to parse is left in the content verbatim (with a
  warning) rather than silently dropped — a truncated or malformed call is
  the agent's to see and retry.  So is a block cut short mid-structure: a
  call whose parameters are only partly recoverable is reported unparseable
  rather than dispatched with parameters silently missing.
* ``qwen3_coder`` parameter values are recovered with ``json.loads`` when
  they parse, else kept as strings.  The template renders scalars via
  ``|string`` and containers via ``|tojson``, so the original types are not
  recoverable from the text alone; when the request's tool schemas are
  available they decide (a ``"string"`` parameter is never retyped), and
  without them JSON-first matches vLLM's parser.  ``NaN``/``Infinity`` are
  never decoded: Python would accept them and re-emit them bare, producing
  ``arguments`` that a strict JSON parser on the agent side rejects.
"""

from __future__ import annotations
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any
from .base import ThinkReasoning

logger = logging.getLogger(__name__)

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"

#: JSON-schema type names that mean "text".  A parameter declared one of
#: these is never re-decoded — the model wrote a string and a string is what
#: the tool expects, however numeric it happens to look.
_STRING_TYPES = frozenset({"string", "str"})


def _reject_nonfinite(constant: str) -> Any:
    """``json.loads`` hook refusing ``NaN`` / ``Infinity`` / ``-Infinity``.

    Python's decoder accepts them and its encoder writes them back out
    bare, so a parameter value of ``NaN`` would travel to the agent as
    ``{"value": NaN}`` — which ``JSON.parse`` and every other strict JSON
    reader rejects, failing the tool call the model correctly made.
    Refusing the decode keeps the model's literal text instead.
    """
    raise ValueError(f"non-finite JSON constant {constant!r}")


def _parameter_types(tools: list[dict] | None) -> dict[str, dict[str, str]]:
    """``{function_name: {parameter_name: json_type}}`` from OpenAI tool defs.

    Best-effort: anything that does not have the expected shape is simply
    absent from the result, which puts that parameter back on the
    schema-less JSON-first path.
    """
    types: dict[str, dict[str, str]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            function = tool
        name = function.get("name")
        parameters = function.get("parameters")
        if not name or not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            continue
        declared: dict[str, str] = {}
        for param, schema in properties.items():
            if not isinstance(schema, dict):
                continue
            param_type = schema.get("type")
            if isinstance(param_type, list):  # e.g. ["string", "null"]
                param_type = next((t for t in param_type if t != "null"), None)
            if isinstance(param_type, str):
                declared[str(param)] = param_type.strip().lower()
        types[str(name)] = declared
    return types


def _split_sections(text: str, open_marker: str, close_marker: str) -> tuple[list[tuple[str, str]], bool]:
    """``(sections, well_formed)`` for ``{open_marker}NAME>body{close_marker}``.

    The scan is plain ``str.find`` — strictly linear in ``len(text)`` no
    matter how degenerate the input.  Regex alternatives
    (``<function=([^>\\n]+)>(.*?)</function>``) go quadratic when a
    repetition-looping completion emits the opener over and over without a
    closer: every opener restarts a lazy scan over the rest of the text,
    and one such completion then pins a CPU core (and the GIL) for minutes.

    Section semantics mirror that regex: the name must be non-empty and
    single-line, and the body ends at the *first* close marker.

    ``well_formed`` reports whether everything *between* the sections — and
    after the last one — is whitespace, which is all the template ever
    writes there.  The dialect has no escaping, so this is the only reliable
    signal that a parameter value contained a close marker and ended its own
    section early, or that the markup was cut off mid-call: either way some
    of what the model wrote is left stranded outside a section.  Counting
    markers cannot do this job — a value merely *mentioning* ``<parameter=``
    keeps the counts unbalanced while parsing perfectly, and a value
    containing a matched pair keeps them balanced while losing its tail.

    Text *before* the first section does not count: that is the model
    abandoning an opener before the real call, which the caller recovers
    rather than rejects.
    """
    sections: list[tuple[str, str]] = []
    well_formed = True
    gap_from = 0  # not derivable from pos: an invalid name advances pos alone
    pos = 0
    while True:
        start = text.find(open_marker, pos)
        if start < 0:
            break
        name_start = start + len(open_marker)
        name_end = text.find(">", name_start)
        if name_end < 0:
            break
        name = text[name_start:name_end]
        if "\n" in name or not name.strip():
            pos = name_start
            continue
        end = text.find(close_marker, name_end + 1)
        if end < 0:
            # An opener with no closer ahead: nothing later can be a complete
            # section either.  The text it opened is stranded, which the
            # trailing-gap check below reports.
            break
        if sections and text[gap_from:start].strip():
            well_formed = False
        # "<function= foo >" names the same tool as "<function=foo>".
        sections.append((name.strip(), text[name_end + 1:end]))
        pos = gap_from = end + len(close_marker)
    if sections and text[gap_from:].strip():
        well_formed = False
    return sections, well_formed


@dataclass
class ToolCall:
    """One parsed tool call: ``arguments`` is always a JSON-encoded string."""

    name: str
    arguments: str


#: Built-in ``<think>`` reasoning parser, the default when a tool parser is
#: constructed without a model-specific one (e.g. the tests and the
#: :func:`split_reasoning` compat shim below).  A model profile injects its own
#: (:class:`proxyserver.tokenization.base.ReasoningParser`) instead.
_DEFAULT_REASONING = ThinkReasoning()


def split_reasoning(text: str) -> str:
    """Deprecated compat shim — prefer a ``ReasoningParser``'s ``strip``.

    Reasoning stripping is now a per-model concern owned by the model profile
    (:mod:`proxyserver.tokenization.base`).  This delegates to the built-in
    ``<think>`` dialect and is kept only because the e2e harness imports it
    directly.
    """
    return _DEFAULT_REASONING.strip(text)


class BaseToolParser:
    """Decode, strip reasoning, and hand the text to the format's ``parse``."""

    #: Parser-name a training framework's ``tool_parser_factory`` is keyed by
    #: (the ``tool_call_parser`` value that used to live in ``mapping.json``);
    #: set by each concrete dialect.
    NAME: str = ""

    def __init__(self, tokenizer: Any, reasoning: Any = None) -> None:
        self.tokenizer = tokenizer
        #: How this model delimits chain-of-thought — supplied by the model
        #: profile, defaulting to the built-in ``<think>`` dialect so a parser
        #: built without one (the tests) behaves as before.
        self.reasoning = reasoning if reasoning is not None else _DEFAULT_REASONING

    async def extract_tool_calls(self, token_ids: list[int], tools: list[dict] | None = None) -> tuple[str, list[ToolCall]]:
        """``(content_text, calls)`` for one sampled completion.

        ``tools`` is the request's OpenAI tool list, when the caller has it:
        the declared parameter types decide how values are typed (see
        :meth:`Qwen3CoderToolParser._coerce`).  Without it parsing still
        works, just without schema guidance.

        The whole pipeline — decode, reasoning strip (via the model's
        :attr:`reasoning` parser), parse — runs in a worker thread: all three
        are CPU-bound on long completions (an injected tokenizer may even do
        I/O), and a degenerate max-length completion must never stall the
        event loop and with it every other in-flight session.
        """
        def _decode_and_parse() -> tuple[str, list[ToolCall]]:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            return self.parse(self.reasoning.strip(text), tools)

        return await asyncio.to_thread(_decode_and_parse)

    def parse(self, text: str, tools: list[dict] | None = None) -> tuple[str, list[ToolCall]]:
        """Split ``text`` into agent-visible content and parsed tool calls.

        Scans with ``str.find`` (see :func:`_split_sections` for why
        regexes are avoided here).  Each close marker is visited once and
        each block is parsed at most twice, so a pathological completion
        that repeats ``<tool_call>`` for its whole length still costs time
        linear in that length.
        """
        param_types: dict[str, dict[str, str]] | None = None
        calls: list[ToolCall] = []
        content_parts: list[str] = []
        last = 0        # end of the last consumed (successfully parsed) block
        pos = 0         # scan cursor
        unparseable = 0
        sample = ""     # first unparseable block, for one summary warning
        while True:
            start = text.find(_TOOL_CALL_OPEN, pos)
            if start < 0:
                break
            body_start = start + len(_TOOL_CALL_OPEN)
            end = text.find(_TOOL_CALL_CLOSE, body_start)
            if end < 0:
                # No close marker anywhere ahead: no later opener can be a
                # complete block either (e.g. a truncated trailing call) —
                # everything from ``last`` stays in the content.
                break
            if param_types is None:
                # Only now that a block exists: most turns carry none, and
                # only some formats read the schemas at all.
                param_types = self._param_types(tools)
            block_calls = self._parse_block(text[body_start:end].strip(), param_types)
            if block_calls == []:
                # Parsed nothing, but the markup itself was intact — so a
                # stray opener the model abandoned is swallowing the real
                # call behind it.  Retry once from the innermost opener this
                # close marker can pair with.  (``None`` instead means the
                # block's own markup was damaged: the inner opener is then
                # part of a parameter *value*, and "parsing" it would invent
                # a tool call the model never made.)  At most one retry, so
                # the scan cannot go quadratic on a completion that loops on
                # the opener.
                inner = text.rfind(_TOOL_CALL_OPEN, body_start, end)
                if inner >= 0:
                    start, body_start = inner, inner + len(_TOOL_CALL_OPEN)
                    block_calls = self._parse_block(text[body_start:end].strip(), param_types)
            if not block_calls:
                # Unparseable block: keep it in the content (do not advance
                # ``last`` past it) so nothing the model said disappears.
                unparseable += 1
                # Slice the sample, never the block: a completion looping on
                # ``<tool_call>`` holds thousands of these, and formatting
                # each one in full would copy megabytes to throw away.
                sample = sample or text[body_start:min(end, body_start + 200)]
                pos = end + len(_TOOL_CALL_CLOSE)
                continue
            content_parts.append(text[last:start])
            last = end + len(_TOOL_CALL_CLOSE)
            pos = last
            calls.extend(block_calls)
        if unparseable:
            logger.warning(
                "%s: left %d unparseable <tool_call> block(s) in the content; first: %s",
                type(self).__name__, unparseable, sample,
            )
        content_parts.append(text[last:])
        return "".join(content_parts).strip(), calls

    def _param_types(self, tools: list[dict] | None) -> dict[str, dict[str, str]]:
        """Declared parameter types for :meth:`_parse_block`, by function name.

        Empty by default: a format that cannot use the schemas should not
        pay to build them (see :func:`_parameter_types` for the cost).
        """
        return {}

    def _parse_block(self, block: str, param_types: dict[str, dict[str, str]]) -> list[ToolCall] | None:
        """Parse one ``<tool_call>`` body.

        Returns the calls it found, ``[]`` if the block held none but its
        markup was intact, or ``None`` if the markup itself was damaged —
        a distinction :meth:`parse` needs, because only in the ``[]`` case
        can an opener *inside* this block be a real call rather than part
        of a parameter value.

        ``param_types`` comes from :meth:`_param_types`.
        """
        raise NotImplementedError


def _tool_call_from_object(obj: Any) -> ToolCall | None:
    """One OpenAI :class:`ToolCall` from a decoded JSON tool-call object.

    Shared by the JSON dialects (``hermes``, ``llama3_json``).  ``obj`` must
    be a mapping carrying a ``name``; its arguments are read from
    ``arguments`` (or ``parameters`` — Llama's spelling), passed through
    untouched when the model already encoded them as a string, and re-encoded
    to a JSON object string otherwise.  Returns ``None`` when ``obj`` is not a
    name-carrying object, so the caller can keep the text as content.
    """
    if not isinstance(obj, dict) or not obj.get("name"):
        return None
    name = str(obj["name"])
    arguments = obj.get("arguments")
    if arguments is None:
        arguments = obj.get("parameters")
    if isinstance(arguments, str):
        # Already encoded by the model — pass through untouched.
        return ToolCall(name=name, arguments=arguments)
    if arguments is not None and not isinstance(arguments, dict):
        # OpenAI's contract is that ``arguments`` decodes to an object.  A
        # list or scalar here would reach the agent as e.g. "[1, 2]", and the
        # ``json.loads(...)["key"]`` every tool dispatcher does would raise
        # instead of the tool simply reporting a bad call.
        logger.warning(
            "tool call %s: arguments decoded to %s, not an object; sending an "
            "empty argument object instead", name, type(arguments).__name__,
        )
        arguments = None
    return ToolCall(name=name, arguments=json.dumps(arguments or {}, ensure_ascii=False))


class HermesToolParser(BaseToolParser):
    """``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks."""

    NAME = "hermes"

    def _parse_block(self, block: str, param_types: dict[str, dict[str, str]]) -> list[ToolCall] | None:
        # JSON has no nested-markup ambiguity, so a failure here is always
        # "malformed" and never "a value ate the markup" — this parser never
        # reports damage (returns [], not None), and never reads the schemas.
        try:
            call = json.loads(block, parse_constant=_reject_nonfinite)
        except ValueError:
            return []
        parsed = _tool_call_from_object(call)
        return [parsed] if parsed is not None else []


class Llama3JsonToolParser(BaseToolParser):
    """Llama 3.x native tool calls: a bare JSON object, no ``<tool_call>`` wrapper.

    Llama's chat template emits a tool call as the *whole* assistant turn —
    ``{"name": "get_weather", "parameters": {"city": "SF"}}`` — with no
    surrounding markup (a builtin ``ipython`` call is prefixed with a
    ``<|python_tag|>`` special token that ``skip_special_tokens`` removes).
    There is no delimiter to scan for, so unlike ``hermes`` / ``qwen3_coder``
    this parser reads the whole reasoning-stripped content as JSON.

    Handled: one JSON object, or a JSON array of them (a model that emits
    parallel calls as ``[{...}, {...}]``).  A turn that is not a JSON
    name-carrying object stays content verbatim — Llama writes prose *or* a
    call, not both, so a parse failure means it was prose.  Not handled (the
    bundled template emits neither, so such a completion is returned as
    content): the ``<|python_tag|>name.call(...)`` builtin-tool spelling,
    which is not JSON, and ``;``-separated parallel calls.
    """

    NAME = "llama3_json"

    def parse(self, text: str, tools: list[dict] | None = None) -> tuple[str, list[ToolCall]]:
        stripped = text.strip()
        calls = self._parse_json_calls(stripped)
        if not calls:
            return stripped, []
        # A Llama tool-call turn carries no prose alongside the JSON.
        return "", calls

    @staticmethod
    def _parse_json_calls(text: str) -> list[ToolCall]:
        if not text or text[0] not in "{[":
            return []
        try:
            decoded = json.loads(text, parse_constant=_reject_nonfinite)
        except ValueError:
            return []
        objects = decoded if isinstance(decoded, list) else [decoded]
        calls: list[ToolCall] = []
        for obj in objects:
            call = _tool_call_from_object(obj)
            if call is None:
                # A JSON payload that is not (all) tool calls is not a
                # tool-call turn — keep the whole thing as content rather than
                # emit a partial set.
                return []
            calls.append(call)
        return calls


class Qwen3CoderToolParser(BaseToolParser):
    """The ``<function=...><parameter=...>`` XML dialect of Qwen3.5 /
    Qwen3-Coder chat templates (vLLM's ``qwen3_coder`` parser).

    ``<parameter=KEY>\\nVALUE\\n</parameter>``: the template writes exactly
    one newline on each side of the value; only those are stripped so
    multi-line values keep their inner whitespace.

    The dialect has no escaping, so a value containing ``</parameter>``,
    ``</function>`` or ``</tool_call>`` ends its own section early.  That is
    unfixable in the format, but it is detectable: whatever the value's tail
    was then leaves text stranded *outside* every section, where the
    template only ever writes whitespace (see :func:`_split_sections`).  A
    block like that is reported unparseable rather than yielding a call with
    a parameter silently missing or a value silently truncated — the tool
    would otherwise do the wrong thing, or report an error the model cannot
    diagnose and answers by re-issuing the same call.

    A value that merely *mentions* the markup without closing a section
    still parses: nothing is stranded, so nothing is lost.
    """

    NAME = "qwen3_coder"

    def _param_types(self, tools: list[dict] | None) -> dict[str, dict[str, str]]:
        return _parameter_types(tools)

    def _parse_block(self, block: str, param_types: dict[str, dict[str, str]]) -> list[ToolCall] | None:
        functions, well_formed = _split_sections(block, "<function=", "</function>")
        if not well_formed:
            return self._damaged(block, "<function=...>")
        calls = []
        for name, body in functions:
            params, well_formed = _split_sections(body, "<parameter=", "</parameter>")
            if not well_formed:
                return self._damaged(block, f"<function={name}>")
            declared = param_types.get(name, {})
            arguments = {
                key: self._coerce(value.removeprefix("\n").removesuffix("\n"), declared.get(key))
                for key, value in params
            }
            calls.append(ToolCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
        return calls

    @staticmethod
    def _damaged(block: str, where: str) -> None:
        """Report a block whose own markup was eaten by a parameter value."""
        logger.warning(
            "Qwen3CoderToolParser: %s left text stranded outside its sections — "
            "a value ended the markup early or the call was cut off; leaving the "
            "whole block in the content rather than emitting a partial call: %.200s",
            where, block,
        )
        return None

    @staticmethod
    def _coerce(value: str, declared_type: str | None = None) -> Any:
        """Recover ``value``'s type, guided by its declared JSON-schema type.

        The chat template renders scalars through ``|string`` and containers
        through ``|tojson``, so the text alone cannot say whether ``1234567890``
        was an integer or a phone number.  Guessing wrong is not cosmetic: an
        agent validating its tool arguments against the same schema rejects
        the call, and the model — seeing an unchanged world — emits it again.

        So when the schema says the parameter is text, it stays text.  Only
        otherwise is JSON decoding attempted, matching vLLM's parser for the
        schema-less case.
        """
        if declared_type in _STRING_TYPES:
            return value
        try:
            return json.loads(value, parse_constant=_reject_nonfinite)
        except ValueError:
            return value


class NullToolParser(BaseToolParser):
    """Tool parsing disabled: clean the content, never extract a call.

    Handed out by :func:`proxyserver.model_registry.build_tool_parser` when
    a model's mapping entry names no ``tool_call_parser``, so the
    agent-visible content is cleaned identically (special tokens dropped,
    reasoning stripped) whether tool parsing is on or off.
    """

    def parse(self, text: str, tools: list[dict] | None = None) -> tuple[str, list[ToolCall]]:
        return text.strip(), []


_PARSERS: dict[str, type[BaseToolParser]] = {
    "hermes": HermesToolParser,
    "llama3_json": Llama3JsonToolParser,
    "qwen3_coder": Qwen3CoderToolParser,
}

#: Parser names the built-in fallback understands (``tool_call_parser``
#: values in ``tokenization/mapping.json``).
SUPPORTED_TOOL_CALL_PARSERS = tuple(sorted(_PARSERS))


def resolve_parser_class(name: str) -> type[BaseToolParser]:
    """The parser class named ``name``; use to validate a mapping eagerly."""
    cls = _PARSERS.get(str(name).strip().lower())
    if cls is None:
        raise ValueError(
            f"Unknown tool_call_parser {name!r}; built-in parsers: "
            f"{', '.join(SUPPORTED_TOOL_CALL_PARSERS)} (a training framework "
            f"may support others via its own tool_parser_factory)"
        )
    return cls


def get_tool_parser(name: str, tokenizer: Any, reasoning: Any = None) -> BaseToolParser:
    """Build the built-in parser named ``name`` on ``tokenizer``.

    ``reasoning`` is the model profile's
    :class:`~proxyserver.tokenization.base.ReasoningParser`; omitted, the
    parser defaults to the built-in ``<think>`` dialect.
    """
    return resolve_parser_class(name)(tokenizer, reasoning=reasoning)
