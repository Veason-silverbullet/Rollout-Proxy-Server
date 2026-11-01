"""Model profiles and reasoning parsers, without concrete tool-parser imports.

Each profile declares its tokenizer, chat markers, parsers, and format hooks;
TokenStreamManager owns the shared strict-TITO algorithm. The model mapping
selects a profile module's ``PROFILE`` instance. See tokenization/README.md
for supported formats and how to add a profile.

Keep this module independent of tool_parser and model_registry to avoid cycles.
"""

from __future__ import annotations
import importlib
import json
import re
from collections.abc import Iterator
from heapq import heappop, heappush
from typing import Any, Protocol
from ..resources import resource_path

#: The directory holding this module and the bundled tokenizer subdirectories.
TOKENIZATION_DIR = resource_path("tokenization")

_THINK_OPEN = "<think>"
# Keep paired quotation styles distinct; do not normalize the completion,
# since quote characters in real argument values must remain byte-exact.
_PROSE_QUOTE_PAIRS = {
    '"': '"', "'": "'", "“": "”", "‘": "’", "«": "»", "‹": "›",
    "„": "“”", "‚": "‘’", "「": "」", "『": "』", "〝": "〞〟",
    "﹁": "﹂", "﹃": "﹄", "＂": "＂", "＇": "＇",
}
_PROSE_QUOTE_CHARS = re.escape("".join(_PROSE_QUOTE_PAIRS) + "".join(_PROSE_QUOTE_PAIRS.values()))
# Scan structure without swallowing a whole call: a quoted or unfinished
# header can precede a real </think>. Quoted examples bound argument spans;
# markup elsewhere still participates in reasoning boundaries.
# Excluding '<' from names bounds scanning even for repeated broken headers.
_REASONING_TOKENS = re.compile(
    r'</?think>|</?tool_call>|<(?:function|parameter)=[^<>\n]+>'
    r'|</(?:function|parameter)>|[{}\[\]' + _PROSE_QUOTE_CHARS + r']|`+|~{3,}'
)
_CODE_DELIMITERS = re.compile(r"`+")
_PROSE_QUOTES = re.compile(r'\\.|[' + _PROSE_QUOTE_CHARS + r']', re.DOTALL)
_CODE_FENCES = re.compile(r"(?m)(?:^|(?<=<think>)|(?<=</think>)) {0,3}(`{3,}|~{3,})([^\n]*)$")
_XML_VALUE_END = re.compile(r"</parameter>|</function>|</tool_call>")
_XML_REASONING_VALUE_MARKERS = re.compile(r"</parameter>|</function>|</tool_call>|<tool_call>|</?think>")
_JSON_OBJECT_START = re.compile(r"\s*\{")
_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*(?:"|\\?\Z)', re.DOTALL)
_JSON_HEADER_MARKERS = re.compile(r'\\.|"|</?think>', re.DOTALL)
_JSON_ARGUMENT_START = re.compile(r'\s*:\s*([\[{"])')
_JSON_STRING_FOLLOW = re.compile(r'\s*(?:[}\]]|[:,]\s*(?:["{\[0-9-]|true\b|false\b|null\b))')


def _quoted_span_ends(text: str) -> dict[int, int]:
    """Index closed prose quotes, inline code spans, and fenced examples.

    Build candidate ends independently: a delimiter inside a real argument
    must not consume the opener of a later example. The scanner only uses
    these entries outside argument values. Code spans may contain newlines;
    unmatched delimiters protect nothing.
    """
    ends: dict[int, int] = {}
    previous: dict[int, int] = {}
    for match in _CODE_DELIMITERS.finditer(text):
        size = match.end() - match.start()
        if size in previous:
            ends[previous[size]] = match.end()
        previous[size] = match.start()

    # A fence can close any earlier opener of the same kind no longer than
    # itself. Each candidate is queued and removed once; no suffix rescans.
    pending: dict[str, list[tuple[int, int]]] = {"`": [], "~": []}
    for match in _CODE_FENCES.finditer(text):
        fence, info = match.groups()
        # Reasoning delimiters are outside the example, even when adjacent
        # to its fence. Keep the original offsets for the structural scan.
        info = info.split("<think>", 1)[0].split("</think>", 1)[0]
        if fence[0] == "`" and "`" in info:
            continue
        candidates = pending[fence[0]]
        if not info.strip():
            while candidates and candidates[0][0] <= len(fence):
                _, start = heappop(candidates)
                ends[start] = match.start(2) + len(info)
        heappush(candidates, (len(fence), match.start(1)))

    pending_quotes: dict[str, list[int]] = {}
    for match in _PROSE_QUOTES.finditer(text):
        quote = match.group()
        if quote.startswith("\\"):
            continue
        start, end = match.span()
        # Apostrophes inside words are not quotation delimiters.
        if quote in ("'", "’", "＇") and start and end < len(text) and text[start - 1].isalnum() and text[end].isalnum():
            continue
        # Every opener gets its first matching closer independently, even
        # when an earlier candidate occurs inside an opaque argument value.
        for opened in pending_quotes.pop(quote, ()):
            ends.setdefault(opened, end)
        for closer in _PROSE_QUOTE_PAIRS.get(quote, ""):
            pending_quotes.setdefault(closer, []).append(start)
    return ends


def _json_header_end(text: str, pos: int, end: int) -> tuple[int, bool]:
    """End a JSON header string at its quote or an actual reasoning boundary.

    Unlike argument strings, an incomplete quoted key/name must not conceal
    the end of reasoning. Escapes are consumed once; no suffix is rescanned.
    """
    while (match := _JSON_HEADER_MARKERS.search(text, pos, end)) is not None:
        marker = match.group()
        if marker == '"':
            return match.end(), True
        if marker.startswith("<"):
            return match.start(), False
        pos = match.end()
    return end, False


def _xml_reasoning_value_end(text: str, pos: int, end: int) -> int:
    """Bound an unquoted XML argument or unfinished header inside reasoning.

    A header mention must not borrow its closing parameter from a new call
    after reasoning ends. Prefer a local XML close; otherwise recover at an
    unmatched </think> before the next call or EOF. Balanced literal think
    tags within a value do not supply a recovery boundary.
    """
    depth = 0
    boundary: int | None = None
    for match in _XML_REASONING_VALUE_MARKERS.finditer(text, pos, end):
        marker = match.group()
        if marker == "<think>":
            depth += 1
        elif marker == "</think>":
            if depth:
                depth -= 1
            else:
                boundary = match.start()
        elif marker == "<tool_call>":
            if boundary is not None:
                return boundary
        else:
            return match.start()
    return boundary if boundary is not None else end


def _reasoning_markers(text: str) -> Iterator[re.Match[str]]:
    """Yield reasoning markers outside XML values and JSON argument strings.

    Quoted examples bound argument scopes: quoted headers cannot claim text
    past their closing delimiter. A bare tool/function header protects nothing.
    Parameter values stay literal through their local XML close. Unquoted
    headers inside reasoning may end at an unmatched reasoning close before
    a new call or EOF. Actual truncated calls keep their argument text. JSON
    argument strings honor escapes and may also be truncated.
    """
    pos = 0
    quote_ends = _quoted_span_ends(text)
    quote_end: int | None = None
    quote_marker: str | None = None
    json_quote = False
    quote_state = (False, False, 0, 0, False)
    in_tool = in_function = False
    json_depth = argument_depth = 0
    in_reasoning: bool | None = None
    tool_in_reasoning = False
    first_content = len(text) - len(text.lstrip())
    while pos < len(text):
        if quote_end is not None and pos >= quote_end:
            in_tool, in_function, json_depth, argument_depth, tool_in_reasoning = quote_state
            quote_end = None
            json_quote = False
        limit = quote_end if quote_end is not None else len(text)
        match = _REASONING_TOKENS.search(text, pos, limit)
        if match is None:
            if quote_end is None:
                break
            pos = quote_end
            continue
        marker = match.group()
        pos = match.end()
        if marker[0] in "`~":
            if quote_end is None:
                quote_end = quote_ends.get(match.start())
                quote_marker = marker
                quote_state = (in_tool, in_function, json_depth, argument_depth, tool_in_reasoning)
        elif json_depth and marker == '"':
            if argument_depth:
                string = _JSON_STRING.match(text, match.start(), limit)
                pos = string.end() if string is not None else limit
            else:
                pos, closed = _json_header_end(text, pos, limit)
                if closed and json_depth == 1:
                    try:
                        key = json.loads(text[match.start():pos])
                    except ValueError:
                        continue
                    argument = _JSON_ARGUMENT_START.match(text, pos, limit)
                    if key in ("arguments", "parameters") and argument is not None:
                        if argument.group(1) == '"':
                            string = _JSON_STRING.match(text, argument.start(1), limit)
                            pos = string.end() if string is not None else limit
                        else:
                            json_depth += 1
                            argument_depth = json_depth
                            pos = argument.end()
            if json_quote and text[pos - 1:pos] == '"':
                # A quote followed by prose also ends an unfinished JSON
                # example; otherwise JSON syntax owns its internal quotes.
                if _JSON_STRING_FOLLOW.match(text, pos, limit) is None:
                    quote_end = pos
        elif not json_depth and marker in _PROSE_QUOTE_PAIRS:
            # A possessive apostrophe or an inch mark cannot open a quote.
            if json_quote and marker == '"':
                quote_end = pos
            elif quote_end is None and (
                marker != _PROSE_QUOTE_PAIRS[marker]
                or match.start() == 0 or not text[match.start() - 1].isalnum()
            ):
                quote_end = quote_ends.get(match.start())
                quote_marker = marker
                quote_state = (in_tool, in_function, json_depth, argument_depth, tool_in_reasoning)
        elif marker in ("<think>", "</think>"):
            # A real boundary ends any unfinished header mentioned in prose.
            in_tool = in_function = False
            json_depth = argument_depth = 0
            quote_end = None
            json_quote = False
            in_reasoning = marker == "<think>"
            yield match
        elif marker == "<tool_call>":
            in_tool, in_function = True, False
            # Before the first boundary, prose followed by a header can be
            # prefilled reasoning. A leading call owns its literal arguments.
            if in_reasoning is None and match.start() == first_content:
                in_reasoning = False
            tool_in_reasoning = in_reasoning is not False
            json_start = _JSON_OBJECT_START.match(text, pos, limit)
            json_depth = 1 if json_start else 0
            argument_depth = 0
            if json_start:
                # In a double-quoted JSON example, JSON's own string parser
                # handles the quotes. Its first key quote must not close the
                # prose scope and expose literal argument reasoning tags.
                if quote_end is not None and quote_marker == '"':
                    quote_end = len(text)
                    json_quote = True
                pos = json_start.end()
        elif marker == "</tool_call>":
            in_tool = in_function = False
            json_depth = argument_depth = 0
        elif json_depth:
            if marker in ("{", "["):
                json_depth += 1
            elif marker in ("}", "]"):
                json_depth -= 1
                if json_depth < argument_depth:
                    argument_depth = 0
        elif in_tool:
            if marker.startswith("<function="):
                in_function = bool(marker[len("<function="):-1].strip())
            elif marker == "</function>":
                in_function = False
            elif in_function and marker.startswith("<parameter=") and marker[len("<parameter="):-1].strip():
                if tool_in_reasoning and quote_end is None:
                    pos = _xml_reasoning_value_end(text, pos, limit)
                else:
                    end = _XML_VALUE_END.search(text, pos, limit)
                    # Actual calls and quoted examples own their arguments
                    # through the closing delimiter or the end of the scope.
                    pos = end.start() if end is not None else limit


def _think_spans(text: str) -> Iterator[tuple[int, int, int | None, int]]:
    """Yield (start, body, closing marker, end) for reasoning outside tools.

    Tool argument values are opaque even inside reasoning; quoted headers are
    not. An initial closing marker implies a prefilled opener. An unfinished
    explicit block owns the remaining completion.
    """
    start: int | None = None
    body = 0
    first = True
    for match in _reasoning_markers(text):
        marker = match.group()
        if marker == _THINK_OPEN:
            if start is None:
                start = 0 if first and not text[:match.start()].strip() else match.start()
            body = match.end()
        elif start is not None or first:
            yield start if start is not None else 0, body, match.start(), match.end()
            start = None
            body = 0
        first = False
    if start is not None:
        yield start, body, None, len(text)


class ReasoningParser(Protocol):
    """How a model delimits chain-of-thought reasoning in a completion.

    Two operations: :meth:`strip` for the tool parser (which must not parse
    tool calls out of a reasoning block), :meth:`extract` for consumers that
    restore a completion's reasoning from its recorded ``completion_text``.
    """

    def extract(self, text: str) -> str | None:
        """The reasoning the model emitted, or ``None`` if none / undelimited."""

    def strip(self, text: str) -> str:
        """``text`` with reasoning removed — the agent-visible content."""


class ThinkReasoning:
    """``<think>...</think>`` chain-of-thought (Qwen3.5 / DeepSeek-R1 dialect).

    :meth:`extract` returns the first completed think block; :meth:`strip`
    removes explicit ``<think>...</think>`` spans wherever they appear,
    preserving surrounding visible content and literal tags in tool bodies.
    A leading block whose opener was prefilled is also removed. Unfinished
    explicit blocks are dropped — a checkpoint that reasons again mid-completion
    must not leak that reasoning to the agent, nor have the tool calls it merely
    weighed there executed.
    """

    def extract(self, text: str) -> str | None:
        for _, body, close, _ in _think_spans(text):
            if close is None:
                return None
            return text[body:close].strip() or None
        return None

    def strip(self, text: str) -> str:
        parts: list[str] = []
        pos = 0
        for start, _, _, end in _think_spans(text):
            parts.append(text[pos:start])
            pos = end
            if start == 0:
                while pos < len(text) and text[pos] == "\n":
                    pos += 1
            elif text.startswith("\n\n", pos):  # the template's separator
                pos += 2
        parts.append(text[pos:])
        return "".join(parts)


class NoReasoning:
    """Models that emit no delimited chain-of-thought (Llama 3.1)."""

    def extract(self, text: str) -> str | None:
        return None

    def strip(self, text: str) -> str:
        return text


class ModelProfile:
    """Declarative model format consumed by the registry and token-stream manager."""

    #: Profile id — the value ``mapping.json`` points a model name at.
    name: str = ""
    #: Bundled tokenizer directory under :data:`TOKENIZATION_DIR`.
    tokenizer_dir: str = ""
    #: Chat-format markers the ``TokenStreamManager`` appends / closes turns
    #: with — stated, never derived from the template.
    generation_prompt: str = ""
    assistant_turn_end: str = ""
    #: Stop markers that also open the next message (GLM). If sampled, the
    #: matching header is already in the stream and is omitted from the delta.
    continuation_headers: tuple[str, ...] = ()
    #: Complete sampled endings that need no synthetic turn gap (Harmony).
    assistant_turn_endings: tuple[str, ...] = ()
    #: Optional preparation of new messages using the preceding assistant.
    #: Does not re-render or alter sampled history.
    prepare_delta_messages: Any = None
    #: Required action-ending tokens, in addition to request-owned stops.
    stop_token_ids: tuple[int, ...] = ()
    #: Supported template controls; other profiles accept effort for compatibility.
    reasoning_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None
    #: Tool-call dialect: a ``BaseToolParser`` subclass, or ``None`` to disable
    #: parsing (annotated ``Any`` so this sink module need not import the class).
    tool_call_parser: Any = None
    #: Chain-of-thought format (a shared, stateless instance).
    reasoning: ReasoningParser = NoReasoning()

    @property
    def tokenizer_path(self) -> str:
        """Absolute path passed to ``AutoTokenizer.from_pretrained``."""
        return str(TOKENIZATION_DIR / self.tokenizer_dir)

    def __repr__(self) -> str:
        return f"<ModelProfile {self.name!r} tokenizer={self.tokenizer_dir!r}>"


def get_profile(name: str) -> ModelProfile:
    """Load the profile from ``tokenization/{name}.py``.

    The module must expose a module-level ``PROFILE`` that is a
    :class:`ModelProfile`.  Any failure — a missing module, a wrong ``PROFILE``
    type — raises here, at load time: unlike an inferred chat format, a missing
    or malformed profile fails loudly instead of ever serving a stream that is
    silently one token off.
    """
    try:
        module = importlib.import_module(f"{__package__}.{name}")
    except ImportError as e:
        raise ValueError(
            f"No model profile module {name!r} (expected tokenization/{name}.py): {e}"
        ) from e
    profile = getattr(module, "PROFILE", None)
    if not isinstance(profile, ModelProfile):
        raise ValueError(
            f"tokenization/{name}.py must define a module-level PROFILE that is "
            f"a ModelProfile; got {type(profile).__name__}"
        )
    return profile
