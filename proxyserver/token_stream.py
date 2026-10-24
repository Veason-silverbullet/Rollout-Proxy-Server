"""Per-session token-stream tracking enforcing strict token-in-token-out (TITO).

Re-tokenizing a conversation from text is not guaranteed to reproduce the
tokens the model actually sampled on earlier turns (token-merge drift at
boundaries, templates re-rendering or stripping past assistant content such
as ``<think>`` blocks, tool calls re-serialized from parsed JSON).

For RL training that mismatch silently breaks the alignment between recorded
completion token IDs / logprobs and the prompts the policy actually consumed.

:class:`TokenStreamManager` therefore keeps, per ``session_id``, the
authoritative token stream — the exact prompt tokens fed to the engine plus
the exact completion tokens it sampled — and builds every follow-up prompt as:

    prompt_{N+1} = prompt_N + completion_ids_N + gap + delta(new msgs)

where ``delta`` tokenizes only the *new* tool/user/system messages (plus the
generation prompt) and never re-tokenizes history, and ``gap`` is the part of
the profile's assistant-message ending that the sampled completion did not
itself contain. For Qwen this is the trailing newline after a natural stop,
or the whole ``<|im_end|>\n`` after a truncated turn; Llama uses ``<|eot_id|>``
without a newline. The gap keeps boundaries aligned with the template.

The echoed assistant message in the request is ignored for tokenization — the
cached sampled tokens are authoritative. Deltas are rendered with
``add_generation_prompt=False``; the template's implicit per-render prefix is
measured and stripped, and the manager appends the profile's generation header.
Collectors consuming recorded token deltas need no further template rendering;
any independent reconstruction must reproduce the exact recorded tokens,
including that header.

TITO is **enforced**, never approximated: a request that cannot be built as a
strict extension of the session's token stream (edited history, unexpected
roles, a missing ``session_id``, or assistant content for a session with no
cached stream) is rejected.

A request with **no assistant message** opens a **new conversation**, replacing
the cached stream on commit, unless it extends the previous request (a
continuation missing its assistant echo, which is rejected). A request whose
normalized messages equal the previous request's is a lost-response retry
(top-level ``tools`` and sampling parameters are not compared):
callers consult :meth:`TokenStreamManager.cached_turn` before
:meth:`~TokenStreamManager.build_prompt` and re-serve the committed turn,
including an opening turn, instead of sampling a branch the agent never saw.

Concurrency: callers must serialize turns of one stream; the rollout provider
holds a per-stream lock across generation and commit. Different streams may
run concurrently, and callers may offload ``build_prompt`` to a worker thread
(the rollout provider does, to keep tokenization off the event loop):
shared session lookup, LRU updates, and eviction are protected by a lock.
Each request retains its session state after lookup, so unrelated eviction
cannot invalidate an in-progress render or replay. Tokenization runs outside
the bookkeeping lock; same-session turns still require caller serialization.
"""

from __future__ import annotations
import logging
import threading
from copy import deepcopy
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Default generation header appended to every engine prompt in place of the
#: template's own, so a template that pre-fills the assistant turn (e.g. a
#: reasoning opener) cannot force tokens the checkpoint must sample itself.
#: Profiles state the per-model value (see ``tokenization/qwen3_5.py``); this
#: ChatML value is the default for direct/test construction.
GENERATION_PROMPT = "<|im_start|>assistant\n"

#: Default assistant-message ending. The engine stops at the sampled EOS, so
#: the newline the template writes after it is never sampled; without it every
#: continuation would read ``<|im_end|><|im_start|>`` instead of the template's
#: ``<|im_end|>\n<|im_start|>``. :meth:`TokenStreamManager._turn_gap_ids`
#: appends whichever suffix is missing, including after truncated turns.
ASSISTANT_TURN_END = "<|im_end|>\n"

#: Content of the throwaway system message used to measure the template's
#: system header (see :meth:`TokenStreamManager._system_header_ids`).  Never
#: reaches an engine.
_HEADER_PROBE = "__rollout_proxy_header_probe__"

#: Content of the throwaway user message used to measure the template's
#: implicit per-render prefix (see
#: :meth:`TokenStreamManager._measure_implicit_prefix`).  Never reaches an
#: engine.
_PREFIX_PROBE = "__rollout_proxy_prefix_probe__"


class TokenStreamError(RuntimeError):
    """A request cannot be served strictly token-in-token-out."""


def _copy_engine_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Keep sampling metadata independent of callers without copying captures."""
    copied = dict(meta)
    if isinstance(copied.get("sampling_params"), dict):
        copied["sampling_params"] = dict(copied["sampling_params"])
    return copied


def as_token_ids(rendered: Any, *, what: str) -> list[int]:
    """Normalize ``apply_chat_template(tokenize=True)`` to a flat ``list[int]``.

    ``transformers`` >= 5 returns a ``BatchEncoding``, whose ``list()`` would
    silently yield its keys; some templates wrap ids in a batch of one.
    Anything that is not a flat list of ints is rejected.
    """
    ids = rendered
    if isinstance(ids, Mapping):  # transformers>=5 BatchEncoding
        if "input_ids" not in ids:
            raise TokenStreamError(f"{what}: tokenizer returned a mapping without 'input_ids': {sorted(ids)}")
        ids = ids["input_ids"]
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes, bytearray)):
        raise TokenStreamError(f"{what}: tokenizer returned {type(ids).__name__}, expected a sequence of int token IDs")
    if len(ids) and isinstance(ids[0], (list, tuple)):  # batch of conversations
        if len(ids) != 1:
            raise TokenStreamError(f"{what}: tokenizer returned {len(ids)} conversations, expected 1")
        ids = ids[0]
    for index, token_id in enumerate(ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TokenStreamError(
                f"{what}: tokenizer returned {type(token_id).__name__} at index "
                f"{index}, expected int token IDs"
            )
    return list(ids)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize message content fields for ``apply_chat_template``.

    Handles:
    * Text-only multi-part ``content`` lists -> plain strings.
    * ``content: null`` in assistant messages with ``tool_calls``.
    * ``tool`` role messages are passed through with ``tool_call_id``.
    """
    if not isinstance(messages, list):
        raise TokenStreamError("messages must be a list of message objects")
    normalized = []
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise TokenStreamError(f"messages[{index}] must be an object")
        msg = dict(msg)
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise TokenStreamError(f"messages[{index}] has an unsupported role")
        if content is not None and not isinstance(content, (str, list)):
            raise TokenStreamError(f"messages[{index}].content must be text, text parts, or null")

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                else:
                    raise TokenStreamError(
                        f"messages[{index}]: only text content parts are supported; "
                        "image, audio, file, and unknown parts cannot be discarded"
                    )
            msg["content"] = "\n".join(text_parts)

        if role == "assistant" and "tool_calls" in msg and msg.get("content") is None:
            msg["content"] = ""

        normalized.append(msg)
    return normalized


def validate_tools(tools: Any) -> None:
    """Accept function definitions without claiming constrained decoding."""
    if tools is None:
        return
    if not isinstance(tools, list):
        raise TokenStreamError("tools must be a list of function definitions or null")
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise TokenStreamError("Only function tools are supported")
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"].strip():
            raise TokenStreamError("Each tool must name a function")
        if function.get("parameters") is not None and not isinstance(function["parameters"], dict):
            raise TokenStreamError("Tool parameters must be a JSON schema object")
        if function.get("strict") is not None and function["strict"] is not False:
            raise TokenStreamError("Strict tool schemas are not enforced by token-ID transports")


def _same_message(prev: Mapping[str, Any], cur: Mapping[str, Any]) -> bool:
    """Whether two messages are the same turn of a conversation."""
    return prev.get("role") == cur.get("role") and prev.get("content") == cur.get("content")


def _extends_conversation(previous: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> bool:
    """Whether ``candidate`` strictly extends ``previous``, using the same
    message equality as the inter-turn delta check."""
    return len(candidate) > len(previous) and all(
        _same_message(prev, cur) for prev, cur in zip(previous, candidate)
    )


#: How many times one committed turn may be re-served to a request with
#: matching normalized messages. A lost-response retry needs one, a retried
#: retry two; past that the caller is treated as asking to re-sample and
#: would otherwise be handed the byte-identical completion forever, with
#: no error to break the loop on.  See :meth:`TokenStreamManager.cached_turn`.
MAX_TURN_REPLAYS = 4


@dataclass(frozen=True)
class ParsedCompletion:
    """Canonical response fields cached with a committed sampled turn."""

    content: str | None
    tool_calls: list[dict[str, Any]] | None
    finish_reason: str
    reasoning_content: str | None = None

    @property
    def message_content(self) -> str | None:
        """OpenAI represents a tool-only response with null content."""
        return ((self.content or "").strip() or None) if self.tool_calls else self.content


@dataclass
class CommittedTurn:
    """Replay data whose lifetime is one committed turn of a token stream."""

    completion_len: int
    log_probs: list[float]
    finish_reason: str
    engine_meta: dict[str, Any]
    parsed_completion: ParsedCompletion | None = None
    tools: list[dict[str, Any]] | None = None


class _SessionState:
    """Authoritative token stream and message history of one session."""

    __slots__ = ("stream", "messages", "last_turn", "replays", "tools", "reasoning_effort")

    def __init__(
        self,
        stream: list[int],
        messages: list[dict[str, Any]],
        last_turn: CommittedTurn | None = None,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str | None = None,
    ):
        self.stream = stream        # prompt + sampled completion tokens
        self.messages = messages    # normalized messages of the last request
        self.last_turn = last_turn  # replay data for cached_turn
        # How often this turn has been re-served for matching normalized messages.
        self.replays = 0
        self.tools = deepcopy(tools or [])
        self.reasoning_effort = reasoning_effort


class TokenStreamManager:
    """Builds strictly token-in-token-out prompts across turns of a session.

    Usage (one instance per tokenizer, shared across sessions)::

        manager = TokenStreamManager(tokenizer)
        cached = manager.cached_turn(session_id, messages)
        if cached is not None:
            ...  # lost-response retry: re-serve, do not sample again
        else:
            prompt_ids = manager.build_prompt(session_id, messages, tools)
            ...  # feed prompt_ids to the engine, sample completion_ids
            manager.commit(session_id, messages, prompt_ids, completion_ids,
                           log_probs, finish_reason,   # these two enable cached_turn
                           engine_meta, tools=tools)   # metadata and tools are cached too

    When selecting a reasoning effort, pass the same ``reasoning_effort`` to
    ``cached_turn``, ``build_prompt``, and ``commit``. The model registry passes
    the profile's ``generation_prompt`` and ``assistant_turn_end``.

    ``tokenizer`` needs ``apply_chat_template(messages, tools=...,
    add_generation_prompt=..., tokenize=True)`` (see :func:`as_token_ids`) and
    ``encode(text, add_special_tokens=False)`` or plain-call tokenization.
    """

    # Roles allowed in the inter-turn gap.
    _GAP_ROLES = ("tool", "user", "system")

    def __init__(
        self,
        tokenizer: Any,
        max_sessions: int = 4096,
        generation_prompt: str = GENERATION_PROMPT,
        max_turn_replays: int = MAX_TURN_REPLAYS,
        assistant_turn_end: str = ASSISTANT_TURN_END,
        continuation_headers: tuple[str, ...] = (),
        assistant_turn_endings: tuple[str, ...] = (),
        prepare_delta_messages: Any = None,
        default_reasoning_effort: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_sessions = max_sessions
        self.max_turn_replays = max_turn_replays
        self.generation_prompt = generation_prompt
        self.assistant_turn_end = assistant_turn_end
        self.continuation_headers = continuation_headers
        self.assistant_turn_endings = assistant_turn_endings
        self.prepare_delta_messages = prepare_delta_messages
        self.default_reasoning_effort = default_reasoning_effort
        self._sessions: OrderedDict[str, _SessionState] = OrderedDict()
        self._state_lock = threading.Lock()
        # Stream ids dropped by LRU eviction, kept (bounded, value-less) only
        # so their later requests can say *why* they can no longer be served —
        # otherwise an evicted stream fails continuation requests with a
        # message about assistant content that reads like the agent's fault.
        self._evicted: OrderedDict[str, None] = OrderedDict()
        self._sys_ids: list[int] | None = None  # lazy; [] if absent or unmeasurable
        self._sys_ids_measured = True           # False if prefix probes fail to isolate it
        self._gen_ids: list[int] | None = None  # lazy tokenization of generation_prompt
        self._end_ids: list[int] | None = None  # lazy tokenization of assistant_turn_end
        self._header_ids: list[int] | None = None  # lazy; [] if unmeasurable

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        session_id: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        reasoning_effort: str | None = None,
    ) -> list[int]:
        """Build prompt IDs, requiring the committed tools on continuations.

        Opening prompts can supply new tools. Pass the same definitions to
        :meth:`commit` after generation succeeds to pin them to that stream.
        """
        if session_id is None:
            raise TokenStreamError("Strict token-in-token-out requires a session_id")

        normalized = normalize_messages(messages)
        validate_tools(tools)
        with self._state_lock:
            state = self._sessions.get(session_id)
            evicted = session_id in self._evicted
            if state is not None:
                self._sessions.move_to_end(session_id)

        first_assistant = next(
            (i for i, msg in enumerate(normalized) if msg.get("role") == "assistant"),
            None,
        )
        if first_assistant is None:
            # Opening request: the full render defines the stream start and
            # replaces any cached stream on commit.
            if state is not None:
                if _extends_conversation(state.messages, normalized):
                    raise TokenStreamError(
                        f"Session {session_id}: request extends the previous "
                        f"conversation but carries no assistant message at "
                        f"position {len(state.messages)} — the sampled "
                        f"assistant turn is missing from the history, so the "
                        f"request cannot be served token-in-token-out.  Echo "
                        f"back every assistant message the proxy returned, "
                        f"including tool-call-only turns whose content is null."
                    )
                logger.info(
                    "Session %s: request carries no assistant message; "
                    "starting a new conversation (replacing the previous "
                    "%d-message stream)",
                    session_id, len(state.messages),
                )
            return self._full_render(normalized, tools, reasoning_effort)

        if state is None:
            if evicted:
                raise TokenStreamError(
                    f"Stream {session_id}: its tokens were evicted (LRU) "
                    f"before this turn, so this conversation cannot continue. "
                    f"More than max_sessions={self.max_sessions} agent streams "
                    f"were retained in this manager (shared per model profile): "
                    f"delete finished rollouts (DELETE /sessions/{{id}}) to "
                    f"release all of their agents' slots, or raise the cap where "
                    f"TokenStreamManager is built."
                )
            raise TokenStreamError(
                f"Session {session_id}: no token stream exists but the "
                f"request carries an assistant message at position "
                f"{first_assistant}; refusing to re-tokenize assistant content"
            )

        if (tools or []) != state.tools:
            raise TokenStreamError(
                f"Session {session_id}: tool definitions cannot change within a conversation; "
                "repeat the opening tools or start a new conversation"
            )
        self._check_reasoning_effort(session_id, state, reasoning_effort)
        return state.stream + self._delta_ids(session_id, state, normalized)

    @staticmethod
    def _check_reasoning_effort(session_id: str, state: _SessionState, reasoning_effort: str | None) -> None:
        if reasoning_effort != state.reasoning_effort:
            raise TokenStreamError(
                f"Session {session_id}: reasoning_effort cannot change within a conversation or retry; "
                "repeat the opening effort or start a new conversation"
            )

    def cached_turn(
        self,
        session_id: str | None,
        messages: list[dict[str, Any]],
        reasoning_effort: str | None = None,
    ) -> tuple[list[int], list[int], list[float], str, dict[str, Any]] | None:
        """The last committed turn, iff the normalized request messages match.

        **Consuming**: a hit spends one of the turn's :data:`MAX_TURN_REPLAYS`
        replays, so call this once per turn served. Past the cap the repeat is
        refused rather than handing the caller the same completion forever.

        Returns:
            ``(prompt_ids, completion_ids, log_probs, finish_reason, engine_meta)``
            of the cached turn, or ``None`` when this is not such a repeat (or the
            committer supplied no logprobs/finish_reason to cache).
        """
        if session_id is None:
            return None
        with self._state_lock:
            state = self._sessions.get(session_id)
        if state is None or state.last_turn is None:
            return None
        normalized = normalize_messages(messages)
        if normalized != state.messages:
            return None
        self._check_reasoning_effort(session_id, state, reasoning_effort)
        with self._state_lock:
            if state.replays >= self.max_turn_replays:
                raise TokenStreamError(
                    f"Session {session_id}: this request has already been "
                    f"re-served {state.replays} times from the cached turn and is "
                    f"being repeated with matching normalized messages again; "
                    f"the proxy cannot re-sample it without forking the session's "
                    f"token stream.  Continue the conversation (echo the assistant "
                    f"turn and append the new messages) or open a new session to sample afresh."
                )
            state.replays += 1
            if self._sessions.get(session_id) is state:
                self._sessions.move_to_end(session_id)
        turn = state.last_turn
        return (
            state.stream[:-turn.completion_len],
            state.stream[-turn.completion_len:],
            list(turn.log_probs),
            turn.finish_reason,
            _copy_engine_meta(turn.engine_meta),
        )

    def committed_turn(self, session_id: str | None, completion_ids: list[int]) -> CommittedTurn | None:
        """Return matching replay state for parsing, without consuming a replay."""
        with self._state_lock:
            state = self._sessions.get(session_id)
        if state is None or state.last_turn is None:
            return None
        turn = state.last_turn
        if len(completion_ids) != turn.completion_len or state.stream[-turn.completion_len:] != completion_ids:
            return None
        return turn

    def commit(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        prompt_ids: list[int],
        completion_ids: list[int],
        log_probs: list[float] | None = None,
        finish_reason: str | None = None,
        engine_meta: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        """Record the sampled completion as the new authoritative stream.

        ``log_probs`` and ``finish_reason`` are optional; when both are
        supplied the turn is also cached so an exact retry of this request —
        a lost-response recovery — can be re-served via :meth:`cached_turn`.
        ``engine_meta`` (what the engine reported about the turn) is cached
        alongside them, so a re-serve reports the engine state the tokens were
        sampled under rather than the current one.
        Pass the same ``tools`` used by :meth:`build_prompt`; they are pinned
        for continuations and retained for parsing a recovered completion.
        """
        last_turn = None
        if completion_ids and log_probs is not None and finish_reason is not None:
            last_turn = CommittedTurn(len(completion_ids), list(log_probs), finish_reason,
                                      _copy_engine_meta(engine_meta or {}), tools=deepcopy(tools))
        state = _SessionState(
            stream=list(prompt_ids) + list(completion_ids),
            messages=normalize_messages(messages),
            last_turn=last_turn,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )
        evicted_sessions = []
        with self._state_lock:
            self._sessions[session_id] = state
            self._sessions.move_to_end(session_id)
            self._evicted.pop(session_id, None)  # committing revives an evicted id
            while len(self._sessions) > self.max_sessions:
                evicted, _ = self._sessions.popitem(last=False)
                self._evicted[evicted] = None
                if len(self._evicted) > self.max_sessions:  # one in, at most one out
                    self._evicted.popitem(last=False)
                evicted_sessions.append(evicted)
        for evicted in evicted_sessions:
            logger.warning(
                "TokenStreamManager evicted agent stream %s (LRU, "
                "max_sessions=%d streams per manager, shared per model profile); "
                "continuations of this conversation will be rejected — delete "
                "finished rollouts to release their agents' slots, or raise the "
                "cap where TokenStreamManager is built",
                evicted, self.max_sessions,
            )

    def drop(self, session_id: str) -> None:
        """Forget a session's token stream (call on session deletion)."""
        with self._state_lock:
            self._sessions.pop(session_id, None)
            self._evicted.pop(session_id, None)

    def has_session(self, session_id: str) -> bool:
        """Whether this manager currently holds a token stream for ``session_id``."""
        with self._state_lock:
            return session_id in self._sessions

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _full_render(self, normalized: list[dict[str, Any]], tools: list[dict] | None,
                     reasoning_effort: str | None = None) -> list[int]:
        # add_generation_prompt=False + the manager's own header — never the
        # template's generation prompt (see GENERATION_PROMPT).
        return as_token_ids(
            self.tokenizer.apply_chat_template(
                normalized,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                **({"reasoning_effort": reasoning_effort}
                   if reasoning_effort is not None and reasoning_effort != self.default_reasoning_effort else {}),
            ),
            what="first-turn prompt",
        ) + self._generation_prompt_ids()

    def _delta_ids(
        self,
        session_id: str,
        state: _SessionState,
        normalized: list[dict[str, Any]],
    ) -> list[int]:
        """Tokenize only the new inter-turn messages.

        Expected shape of the incoming conversation::

            state.messages + [assistant echo] + new tool/user/system msgs

        The assistant echo is never re-tokenized (the sampled tokens cached
        in ``state.stream`` are authoritative), so its content is not
        compared — only its role.
        """
        n_prev = len(state.messages)
        if len(normalized) < n_prev + 2:
            raise TokenStreamError(
                f"Session {session_id}: request does not extend the previous "
                f"conversation ({len(normalized)} messages, need at least {n_prev + 2})"
            )

        for i, (prev, cur) in enumerate(zip(state.messages, normalized)):
            if not _same_message(prev, cur):
                raise TokenStreamError(
                    f"Session {session_id}: conversation history was "
                    f"modified at message {i} (role={cur.get('role')!r})"
                )

        if normalized[n_prev].get("role") != "assistant":
            raise TokenStreamError(
                f"Session {session_id}: expected an assistant echo at "
                f"position {n_prev}, got role={normalized[n_prev].get('role')!r}"
            )

        new_msgs = normalized[n_prev + 1:]
        bad_roles = [m.get("role") for m in new_msgs if m.get("role") not in self._GAP_ROLES]
        if bad_roles:
            raise TokenStreamError(
                f"Session {session_id}: inter-turn messages may only have "
                f"roles {self._GAP_ROLES}, got {bad_roles}"
            )

        if self.prepare_delta_messages is not None:
            assistant = normalized[n_prev]
            if state.last_turn is not None and state.last_turn.parsed_completion is not None:
                # Tool-result recipients must follow the response we served,
                # not a potentially modified assistant echo from the client.
                parsed = state.last_turn.parsed_completion
                assistant = {"role": "assistant", "content": parsed.message_content, "tool_calls": parsed.tool_calls}
            new_msgs = self.prepare_delta_messages(new_msgs, assistant)

        delta = as_token_ids(
            self.tokenizer.apply_chat_template(
                new_msgs,
                add_generation_prompt=False,
                tokenize=True,
            ),
            what=f"session {session_id}: inter-turn delta",
        )
        sys_ids = self._get_sys_ids()
        if sys_ids:
            if delta[: len(sys_ids)] == sys_ids:
                delta = delta[len(sys_ids):]
            elif self._sys_ids_measured:
                # E.g. Llama renders a mid-stream system message as its dated
                # top-of-conversation block; splicing that preamble is off-format.
                raise TokenStreamError(
                    f"Session {session_id}: the inter-turn delta for new "
                    f"messages {[m.get('role') for m in new_msgs]} does not "
                    f"begin with the template's measured per-render prefix, so "
                    f"the template renders them as part of its "
                    f"top-of-conversation block (bos / default system preamble) "
                    f"rather than a mid-stream turn.  This is usually a system "
                    f"message mid-conversation, which this template cannot place "
                    f"token-in-token-out; keep system messages in the opening turn."
                )
        self._reject_unstripped_system_prefix(session_id, new_msgs, delta)
        # GLM can sample the next role header as its stop token. Preserve
        # that sampled token and append only the remainder of the new turn.
        for header in self.continuation_headers:
            header_ids = self._tokenize(header, what="continuation header")
            if header_ids and state.stream[-len(header_ids):] == header_ids:
                if delta[:len(header_ids)] != header_ids:
                    raise TokenStreamError(
                        f"Session {session_id}: sampled continuation header {header!r} "
                        "does not match the next message role"
                    )
                delta = delta[len(header_ids):]
                break
        return self._turn_gap_ids(state.stream) + delta + self._generation_prompt_ids()

    def _turn_gap_ids(self, stream: list[int]) -> list[int]:
        """The tokens that must sit between ``stream`` and the next render.

        Use the profile's ``assistant_turn_end``, unless the stream already
        ends with a complete ``assistant_turn_endings`` marker (Harmony).
        Return the suffix of the ending the stream does not already have.
        For ChatML, a sampled ``<|im_end|>`` needs only the trailing newline;
        a truncated turn receives the missing ending.
        """
        for ending in self.assistant_turn_endings:
            ids = self._tokenize(ending, what="completed assistant ending")
            if ids and stream[-len(ids):] == ids:
                return []
        end = self._turn_end_ids()
        for have in range(len(end), 0, -1):
            if stream[-have:] == end[:have]:
                return end[have:]
        return list(end)

    def _reject_unstripped_system_prefix(self, session_id: str, new_msgs: list[dict[str, Any]], delta: list[int]) -> None:
        """Refuse a delta that still carries the template's implicit system prefix.

        :meth:`_get_sys_ids` compares one- and two-user-message renders to
        isolate a fixed prefix. If those probes fail or the message renders
        differently by position, the prefix remains unmeasured. A delta that
        opens with the template's system header while its first new message
        is not a system message would inject an implicit system block into
        the authoritative stream: fail the turn instead.
        """
        if self._sys_ids_measured or (new_msgs and new_msgs[0].get("role") == "system"):
            return
        header = self._system_header_ids()
        if header and delta[: len(header)] == header:
            raise TokenStreamError(
                f"Session {session_id}: the inter-turn delta opens with the "
                f"chat template's system header, but the new messages start "
                f"with role={new_msgs[0].get('role')!r} — the template adds an "
                f"implicit system prefix to every render and this one could "
                f"not be measured (the user-message probes failed or could not "
                f"isolate a fixed prefix), so appending the delta would inject a second "
                f"system prompt mid-stream.  Refusing rather than corrupting "
                f"the session's token stream."
            )

    def _get_sys_ids(self) -> list[int]:
        """The implicit prefix every render carries, measured once.

        Some templates prepend a fixed block to *every* render — a ``bos``
        token (Llama), a default dated system message (Llama 3.1), or both —
        that belongs at the start of the stream but must not be repeated in
        every inter-turn delta.  It is measured position-independently (see
        :meth:`_measure_implicit_prefix`), without rendering an empty
        conversation. A template whose prefix cannot be measured leaves it
        *unmeasured* (``_sys_ids_measured=False``, empty) so :meth:`_reject_unstripped_system_prefix`
        guards deltas instead of the stream being silently corrupted.
        """
        if self._sys_ids is None:
            self._sys_ids = self._measure_implicit_prefix()
        return self._sys_ids

    def _measure_implicit_prefix(self) -> list[int]:
        """Measure the fixed prefix the template adds to every render.

        Rendering one user message yields ``prefix + user_block`` and two
        yields ``prefix + user_block + user_block`` (the prefix appears once,
        the message once per occurrence), so the prefix is the first
        ``2*len(one) - len(two)`` tokens — provided the two renders line up
        that way (``two == one + one[n:]``).  Rendering ``[]`` is *not* used:
        the templates that carry the most prefix (Llama) refuse it.

        Returns the prefix, or ``[]`` with ``_sys_ids_measured=False`` when the
        probes fail (the tokenizer has no real template) or the user turn
        renders differently by position (the measurement cannot isolate the
        prefix) — either way the delta-side guard takes over.
        """
        probe = [{"role": "user", "content": _PREFIX_PROBE}]
        try:
            one = as_token_ids(
                self.tokenizer.apply_chat_template(probe, add_generation_prompt=False, tokenize=True),
                what="implicit prefix (one message)",
            )
            two = as_token_ids(
                self.tokenizer.apply_chat_template(probe + probe, add_generation_prompt=False, tokenize=True),
                what="implicit prefix (two messages)",
            )
        except Exception as e:
            self._sys_ids_measured = False
            logger.info(
                "Cannot render the implicit-prefix probes (%s: %s); the "
                "template's per-render prefix — if any — is left unmeasured and "
                "inter-turn deltas are checked for an unstripped system header "
                "instead", type(e).__name__, e,
            )
            return []
        n = 2 * len(one) - len(two)
        if n < 0 or two != one + one[n:]:
            self._sys_ids_measured = False
            logger.info(
                "The template renders a user turn differently by position, so "
                "its implicit per-render prefix cannot be isolated; inter-turn "
                "deltas are checked for an unstripped system header instead",
            )
            return []
        if n:
            logger.info(
                "Measured the template's implicit per-render prefix: %d tokens "
                "(stripped from every inter-turn delta)", n,
            )
        return one[:n]

    def _system_header_ids(self) -> list[int]:
        """Token IDs the template writes *before* a system message's content.

        Derived by rendering a sentinel system message and keeping what
        precedes the sentinel, so no chat format is hardcoded.  ``[]`` when
        the probe cannot render or the sentinel is not recoverable from it.
        """
        if self._header_ids is None:
            self._header_ids = []
            try:
                rendered = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": _HEADER_PROBE}],
                    add_generation_prompt=False,
                    tokenize=False,
                )
                head, sep, _ = str(rendered).partition(_HEADER_PROBE)
                if sep and head:
                    self._header_ids = self._tokenize(head, what="system header")
            except Exception as e:
                logger.debug("Cannot measure the template's system header: %s", e)
        return self._header_ids

    def _turn_end_ids(self) -> list[int]:
        """Token IDs of the profile's ``assistant_turn_end`` (tokenized once)."""
        if self._end_ids is None:
            self._end_ids = self._tokenize(self.assistant_turn_end, what="assistant turn end")
        return list(self._end_ids)

    def _generation_prompt_ids(self) -> list[int]:
        """Token IDs of the manager's generation prompt (tokenized once)."""
        if self._gen_ids is None:
            self._gen_ids = self._tokenize(self.generation_prompt, what="generation prompt")
        return list(self._gen_ids)

    def _tokenize(self, text: str, *, what: str) -> list[int]:
        """Tokenize plain text (no special tokens added by the tokenizer).

        Uses ``tokenizer.encode(text, add_special_tokens=False)``, falling
        back to calling the tokenizer itself — the two spellings every HF
        (and HF-compatible) tokenizer offers for plain-text tokenization.
        """
        try:
            if hasattr(self.tokenizer, "encode"):
                rendered = self.tokenizer.encode(text, add_special_tokens=False)
            else:
                rendered = self.tokenizer(text, add_special_tokens=False)
        except TypeError as e:
            raise TokenStreamError(
                f"tokenizer cannot tokenize the {what} {text!r}: it needs "
                f"encode(text, add_special_tokens=False) or plain-call "
                f"tokenization ({e})"
            ) from e
        return as_token_ids(rendered, what=what)
