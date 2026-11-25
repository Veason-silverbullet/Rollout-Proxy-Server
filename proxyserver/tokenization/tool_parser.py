"""Built-in completion parsers selected by model profiles.

extract_completion returns content, reasoning, and ToolCalls with JSON-encoded
arguments. Optional OpenAI tool schemas guide XML argument typing. Supported
dialects are documented on their parser classes and in tokenization/README.md.

Text parsers skip special tokens, retaining non-special tool/reasoning markup,
and strip reasoning before parsing calls. Harmony reads structural token IDs
before decoding, preserving literal marker spellings as content. Recorded tokens
are unaffected by parsing.

Malformed or truncated calls remain visible as content. The JSON dialects also
accept string arguments verbatim and replace decoded scalar/list arguments with
an empty object; see _tool_call_from_object.
"""

from __future__ import annotations
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any
from .base import ThinkReasoning
from .stops import decode_stopped

logger = logging.getLogger(__name__)

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"

#: JSON-schema type names that mean "text".  A parameter declared one of
#: these is never re-decoded — the model wrote a string and a string is what
#: the tool expects, however numeric it happens to look.
_STRING_TYPES = frozenset({"string", "str"})

# Retain property constraints and all type alternatives in declaration order.
_ParameterSchemas = dict[str, dict[str, dict[str, Any]]]

# Nullable inference understands only these assertions and inert annotations.
# Unknown keywords may constrain either candidate, so retain the old fallback.
_NULLABLE_HINT_KEYS = frozenset({
    "type", "enum", "const", "title", "description", "default", "examples",
    "deprecated", "readOnly", "writeOnly", "$comment",
})


def _reject_nonfinite(constant: str) -> Any:
    """``json.loads`` hook refusing ``NaN``/``Infinity``, which Python would
    re-emit as invalid JSON; the model's literal text is kept instead."""
    raise ValueError(f"non-finite JSON constant {constant!r}")


def _nullable_anyof_branches(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Recognize a nullable pair only when all its assertion keywords are supported."""
    if schema.keys() - _NULLABLE_HINT_KEYS - {"anyOf"}:
        return []
    alternatives = schema.get("anyOf")
    if (isinstance(alternatives, list) and len(alternatives) == 2
            and all(
                isinstance(branch, dict) and branch.keys() <= _NULLABLE_HINT_KEYS
                and branch.get("type") in ("string", "number", "integer", "boolean", "object", "array", "null")
                for branch in alternatives
            )
            and sum(branch["type"] == "null" for branch in alternatives) == 1):
        return alternatives
    return []


def _parameter_schemas(tools: list[dict] | None) -> _ParameterSchemas:
    """Property schemas with normalized type tuples, indexed by function/parameter.

    Direct types and simple nullable anyOf pairs provide type hints. Inferred
    pairs allow enum/const and annotations; other constraints, references, and
    general unions retain the schema-less JSON-first path. Explicit type hints
    keep their existing behavior. This is not general schema validation.
    """
    schemas: _ParameterSchemas = {}
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
        declared: dict[str, dict[str, Any]] = {}
        for param, schema in properties.items():
            if not isinstance(schema, dict):
                continue
            param_type = schema.get("type")
            if "type" not in schema:
                alternatives = _nullable_anyof_branches(schema)
                if alternatives:
                    param_type = [branch["type"] for branch in alternatives]
            if isinstance(param_type, str):
                param_type = [param_type]
            if isinstance(param_type, list) and all(isinstance(t, str) for t in param_type):
                declared[str(param)] = {**schema, "type": tuple(t.strip().lower() for t in param_type)}
        schemas[str(name)] = declared
    return schemas


def _split_sections(text: str, open_marker: str, close_marker: str) -> tuple[list[tuple[str, str]], bool]:
    """``(sections, well_formed)`` for ``{open_marker}NAME>body{close_marker}``.

    A linear ``str.find`` scan; the equivalent lazy regex goes quadratic on
    completions that loop on the opener. Names are non-empty and single-line,
    and a body ends at the first close marker. ``well_formed`` is false when
    non-whitespace text sits between or after sections: the dialect has no
    escaping, so that is the signal of a value that ended its own section or
    truncated markup. Text before the first section is an abandoned opener.
    """
    sections: list[tuple[str, str]] = []
    well_formed = True
    gap_from = 0  # not derivable from pos: an invalid name advances pos alone
    pos = 0
    name_end = -1
    while True:
        start = text.find(open_marker, pos)
        if start < 0:
            break
        name_start = start + len(open_marker)
        # Invalid names can share one distant '>'. Remember it so repeated
        # openers do not rescan the same suffix. Skip through the first
        # newline before slicing: all names starting before it are invalid,
        # but a valid opener after it must still be recoverable.
        if name_end < name_start:
            name_end = text.find(">", name_start)
        if name_end < 0:
            break
        newline = text.find("\n", name_start, name_end)
        if newline >= 0:
            pos = newline + 1
            continue
        name = text[name_start:name_end]
        if not name.strip():
            pos = name_end + 1
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


@dataclass(frozen=True)
class DecodedCompletion:
    """Model-format output before the provider assigns stable tool-call IDs."""

    content: str | None
    reasoning_content: str | None
    tool_calls: list[ToolCall]


#: Built-in ``<think>`` reasoning parser, the default when a tool parser is
#: constructed without a model-specific one (e.g. in tests). A model profile
#: injects its own :class:`proxyserver.tokenization.base.ReasoningParser` instead.
_DEFAULT_REASONING = ThinkReasoning()


class BaseToolParser:
    """Decode, strip reasoning, and hand the text to the format's ``parse``."""

    #: Parser name set by each concrete dialect and selected by the model profile.
    NAME: str = ""
    skip_special_tokens = True

    def __init__(self, tokenizer: Any, reasoning: Any = None) -> None:
        self.tokenizer = tokenizer
        self.reasoning = reasoning if reasoning is not None else _DEFAULT_REASONING

    async def extract_completion(
        self, token_ids: list[int], tools: list[dict] | None = None,
        *, completion_text: str | None = None,
        stop_strings: tuple[str, ...] = (),
        stop_skip_special_tokens: bool = False,
    ) -> DecodedCompletion:
        """Decode all response fields once, before the provider caches them.

        The raw decode supplied by generate() preserves the reasoning semantics
        for formats whose tool parser skips special tokens.
        stop_skip_special_tokens selects the engine's stop-matching text view.
        """
        if stop_strings:
            # Truncate before interpreting tool syntax or reasoning, including
            # matches inside a single sampled token. Never re-encode this view.
            def parse_stopped():
                text, raw = decode_stopped(
                    token_ids, self.tokenizer, stop_strings,
                    skip_special_tokens=self.skip_special_tokens, stop_skip_special_tokens=stop_skip_special_tokens,
                )
                content, calls = self.parse_completion(text, tools)
                reasoning = self.extract_reasoning(token_ids, raw)
                return DecodedCompletion(content, reasoning, calls or [])

            return await asyncio.to_thread(parse_stopped)
        content, calls = await self.extract_tool_calls(token_ids, tools=tools)
        reasoning = await asyncio.to_thread(self.extract_reasoning, token_ids, completion_text)
        return DecodedCompletion(content, reasoning, calls or [])

    def extract_reasoning(self, token_ids: list[int], completion_text: str | None = None) -> str | None:
        """Best-effort reasoning extraction, including token-aware formats."""
        try:
            if hasattr(self.reasoning, "extract_tokens"):
                return self.reasoning.extract_tokens(token_ids, self.tokenizer)
            if completion_text is None:
                completion_text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            return self.reasoning.extract(completion_text)
        except Exception:
            # Reasoning has always been optional: failure to extract it must
            # not discard otherwise valid content or tool calls.
            return None

    async def extract_tool_calls(self, token_ids: list[int], tools: list[dict] | None = None) -> tuple[str, list[ToolCall]]:
        """``(content_text, calls)`` for one sampled completion, parsed in a
        worker thread. Optional ``tools`` schemas type argument values."""
        return await asyncio.to_thread(self.parse_token_ids, token_ids, tools)

    def parse_token_ids(self, token_ids: list[int], tools: list[dict] | None = None) -> tuple[str, list[ToolCall]]:
        """Synchronous parsing entry point that retains sampled token identity."""
        text = self.tokenizer.decode(token_ids, skip_special_tokens=self.skip_special_tokens)
        return self.parse_completion(text, tools)

    def parse_completion(self, text: str, tools: list[dict] | None = None) -> tuple[str, list[ToolCall]]:
        """Prepare a decoded completion before the format-specific parser."""
        return self.parse(self.reasoning.strip(text), tools)

    def parse(self, text: str, tools: list[dict] | None = None) -> tuple[str, list[ToolCall]]:
        """Split ``text`` into agent-visible content and parsed tool calls.

        Scans with ``str.find`` (see :func:`_split_sections` for why regexes
        are avoided here).  Each close marker is visited once and each block
        is parsed at most twice, so a pathological completion that repeats
        ``<tool_call>`` for its whole length still costs time linear in that length.
        """
        param_schemas: _ParameterSchemas | None = None
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
            if param_schemas is None:
                # Only now that a block exists: most turns carry none, and
                # only some formats read the schemas at all.
                param_schemas = self._param_schemas(tools)
            block_calls = self._parse_block(text[body_start:end].strip(), param_schemas)
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
                    block_calls = self._parse_block(text[body_start:end].strip(), param_schemas)
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

    def _param_schemas(self, tools: list[dict] | None) -> _ParameterSchemas:
        """Declared parameter schemas for :meth:`_parse_block`, by function name.

        Empty by default: a format that cannot use the schemas should not
        pay to build them (see :func:`_parameter_schemas` for the cost).
        """
        return {}

    def _parse_block(self, block: str, param_schemas: _ParameterSchemas) -> list[ToolCall] | None:
        """Parse one ``<tool_call>`` body.

        Returns the calls it found, ``[]`` if the block held none but its
        markup was intact, or ``None`` if the markup itself was damaged —
        a distinction :meth:`parse` needs, because only in the ``[]`` case
        can an opener *inside* this block be a real call rather than part
        of a parameter value.

        ``param_schemas`` comes from :meth:`_param_schemas`.
        """
        raise NotImplementedError


def _tool_call_from_object(obj: Any) -> ToolCall | None:
    """Decode a Hermes or Llama call, leaving unnamed objects as content."""
    if not isinstance(obj, dict) or not obj.get("name"):
        return None
    name = str(obj["name"])
    arguments = obj.get("arguments")
    if arguments is None:
        arguments = obj.get("parameters")
    if isinstance(arguments, str):
        return ToolCall(name=name, arguments=arguments)
    if arguments is not None and not isinstance(arguments, dict):
        logger.warning(
            "tool call %s: arguments decoded to %s, not an object; sending an "
            "empty argument object instead", name, type(arguments).__name__,
        )
        arguments = None
    return ToolCall(name=name, arguments=json.dumps(arguments or {}, ensure_ascii=False))


class HermesToolParser(BaseToolParser):
    """``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks."""

    NAME = "hermes"

    def _parse_block(self, block: str, param_schemas: _ParameterSchemas) -> list[ToolCall] | None:
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
    call, not both, so a parse failure leaves it as content. Not handled:
    the ``<|python_tag|>name.call(...)`` builtin-tool spelling, which the
    bundled template can emit with ``builtin_tools`` (not supplied by the
    proxy), and ``;``-separated parallel calls. Both stay content.
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
                # A JSON payload that is not (all) tool calls is not a tool-call
                # turn — keep the whole thing as content rather than emit
                # a partial set.
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

    def _param_schemas(self, tools: list[dict] | None) -> _ParameterSchemas:
        return _parameter_schemas(tools)

    def _parse_block(self, block: str, param_schemas: _ParameterSchemas) -> list[ToolCall] | None:
        functions, well_formed = _split_sections(block, "<function=", "</function>")
        if not well_formed:
            return self._damaged(block, "<function=...>")
        calls = []
        for name, body in functions:
            params, well_formed = _split_sections(body, "<parameter=", "</parameter>")
            if not well_formed or (not params and body.strip()):
                return self._damaged(block, f"<function={name}>")
            declared = param_schemas.get(name, {})
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
    def _coerce(value: str, schema: dict[str, Any] | None = None) -> Any:
        """Recover ``value``'s type, guided by its property schema.

        The chat template renders scalars through ``|string`` and containers
        through ``|tojson``, so the text alone cannot say whether ``1234567890``
        was an integer or a phone number.  Guessing wrong is not cosmetic: an
        agent validating its tool arguments against the same schema rejects
        the call, and the model — seeing an unchanged world — emits it again.

        Preserve text when the first non-null type is string, retaining the
        existing union precedence and nullable-string behavior. Otherwise accept
        Qwen3.5's Python-style scalar spellings for declared boolean/null types.
        For string unions, enum/const may disambiguate those spellings: keep
        valid literal text when the converted scalar is excluded. JSON decoding
        remains the fallback. For nullable anyOf strings, parent and branch
        enum/const constraints may select null when the literal text is excluded.
        This is not a general JSON Schema validator.
        """
        schema = schema or {}

        def allowed(candidate: str | bool | None, constraints: dict[str, Any] = schema) -> bool:
            # These candidates are strings, booleans, or null. Python's True == 1
            # and False == 0 must not make numeric enum/const values match them.
            enum = constraints.get("enum")
            if isinstance(enum, list) and not any(type(item) is type(candidate) and item == candidate for item in enum):
                return False
            return "const" not in constraints or (
                type(constraints["const"]) is type(candidate) and constraints["const"] == candidate
            )

        declared_types = schema.get("type", ())
        preferred_type = next((kind for kind in declared_types if kind != "null"), None)
        scalar = value.strip()
        if preferred_type in _STRING_TYPES:
            if "null" in declared_types and scalar in ("null", "None"):
                branches = _nullable_anyof_branches(schema)
                string_branch = next((b for b in branches if b["type"] == "string"), None)
                if string_branch is not None:
                    null_branch = next(b for b in branches if b["type"] == "null")
                    if ((not allowed(value) or not allowed(value, string_branch))
                            and allowed(None) and allowed(None, null_branch)):
                        return None
            return value
        if "boolean" in declared_types and scalar.lower() in ("true", "false"):
            converted = scalar.lower() == "true"
        elif "null" in declared_types and scalar == "None":
            converted = None
        else:
            try:
                return json.loads(value, parse_constant=_reject_nonfinite)
            except ValueError:
                return value

        if _STRING_TYPES.intersection(declared_types) and allowed(value) and not allowed(converted):
            return value
        return converted


class GLMToolParser(BaseToolParser):
    """GLM's ``name<arg_key>key</arg_key><arg_value>value</arg_value>``.

    Like Qwen, strings are emitted literally and other values as JSON.
    Preserve malformed/truncated blocks as content rather than dispatching
    partial arguments. Values retain their whitespace, including newlines.
    """

    NAME = "glm47"

    def _param_schemas(self, tools: list[dict] | None) -> _ParameterSchemas:
        return _parameter_schemas(tools)

    def _parse_block(self, block: str, param_schemas: _ParameterSchemas) -> list[ToolCall] | None:
        name, marker, rest = block.partition("<arg_key>")
        name = name.strip()
        if not name or any(ch.isspace() or ch in "<>" for ch in name):
            return None
        arguments = {}
        declared = param_schemas.get(name, {})
        while marker:
            key, closed, rest = rest.partition("</arg_key>")
            key = key.strip()
            if not closed or not key or "<" in key or ">" in key or key in arguments:
                return None
            rest = rest.lstrip()
            if not rest.startswith("<arg_value>"):
                return None
            value, closed, rest = rest[len("<arg_value>"):].partition("</arg_value>")
            if not closed:
                return None
            arguments[key] = Qwen3CoderToolParser._coerce(value, declared.get(key))
            rest = rest.strip()
            if not rest:
                break
            if not rest.startswith("<arg_key>"):
                return None
            rest = rest[len("<arg_key>"):]
        return [ToolCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False))]


class NullToolParser(BaseToolParser):
    """Tool parsing disabled: clean the content, never extract a call.

    Used when a model's profile sets ``tool_call_parser=None``, so the agent-visible
    content is cleaned identically (special tokens dropped, reasoning stripped)
    whether tool parsing is on or off.
    """

    def parse(self, text: str, tools: list[dict] | None = None) -> tuple[str, list[ToolCall]]:
        return text.strip(), []


class HarmonyToolParser(BaseToolParser):
    """gpt-oss function recipients and JSON bodies, split by structural IDs."""

    NAME = "gpt_oss"

    def parse_completion(self, text: str, tools: list[dict] | None = None) -> tuple[str, list[ToolCall]]:
        return self.parse(text, tools)

    def parse(self, text: str, tools: list[dict] | None = None) -> tuple[str, list[ToolCall]]:
        """Parse explicitly serialized Harmony; sampled output uses parse_token_ids."""
        from .harmony import assistant_messages

        decoded = self._parse_messages(assistant_messages(text))
        return decoded.content, decoded.tool_calls

    def parse_token_ids(self, token_ids: list[int], tools: list[dict] | None = None) -> tuple[str, list[ToolCall]]:
        decoded = self.decode_completion(token_ids)
        return decoded.content, decoded.tool_calls

    async def extract_completion(
        self, token_ids: list[int], tools: list[dict] | None = None,
        *, completion_text: str | None = None,
        stop_strings: tuple[str, ...] = (),
        stop_skip_special_tokens: bool = False,
    ) -> DecodedCompletion:
        return await asyncio.to_thread(self.decode_completion, token_ids, stop_strings, stop_skip_special_tokens)

    def decode_completion(
        self, token_ids: list[int], stop_strings: tuple[str, ...] = (),
        stop_skip_special_tokens: bool = False,
    ) -> DecodedCompletion:
        from .harmony import assistant_messages_from_tokens

        # Read structural IDs before decoding text to preserve channels and recipients.
        return self._parse_messages(assistant_messages_from_tokens(
            token_ids, self.tokenizer, stop_strings, stop_skip_special_tokens,
        ))

    @staticmethod
    def _parse_messages(messages) -> DecodedCompletion:
        content, calls, reasoning = [], [], []
        for message in messages:
            if message.recipient is None:
                if message.channel == "analysis":
                    if message.body.strip():
                        reasoning.append(message.body.strip())
                elif message.body:
                    content.append(message.body)
                continue
            arguments = message.body.strip()
            if message.recipient.startswith("functions.") and message.ending == "<|call|>":
                name = message.recipient.removeprefix("functions.")
                try:
                    decoded = json.loads(arguments, parse_constant=_reject_nonfinite)
                    # Also catch overflow (1e999), including nested values.
                    json.dumps(decoded, allow_nan=False)
                except (ValueError, OverflowError):
                    decoded = None
                if name and isinstance(decoded, dict):
                    calls.append(ToolCall(name=name, arguments=arguments))
                    continue
            # Malformed/truncated or unsupported builtin calls remain visible.
            if message.body:
                content.append(message.body)
        return DecodedCompletion("\n\n".join(content), "\n\n".join(reasoning) or None, calls)
