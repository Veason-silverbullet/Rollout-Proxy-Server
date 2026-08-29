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
the template's assistant-message ending (:data:`ASSISTANT_TURN_END`) the
sampled completion did not itself contain — the trailing newline after a
natural stop, the whole ``<|im_end|>\n`` after a truncated turn.  Without it
the stream is one token short of the template's own render at every turn boundary.

The echoed assistant message in the request is ignored for tokenization — the
cached sampled tokens are authoritative.

The delta scheme intentionally mirrors trainer-side reconstruction (e.g.
verl's ``RemoteAgentLoop._reconstruct_output`` / ``_tokenize_tool_messages``):
render the new messages alone and strip the template's implicit system
prefix (``apply_chat_template([])``) if present.  Renders use
``add_generation_prompt=False`` and the manager appends its own generation
prompt (:data:`GENERATION_PROMPT`) instead of the template's — the trainer
must reconstruct with the same header for byte-identical token streams.

TITO is **enforced**, never approximated: any request that cannot be built
as a strict extension of the session's token stream — edited history,
unexpected roles, a missing ``session_id``, or sampled assistant content
arriving for a session with no cached stream — raises
:class:`TokenStreamError` instead of falling back to re-tokenization.
Callers surface the error to the agent; a failed rollout is recoverable,
a silently corrupted one is not.

One case is *not* a violation: a request carrying **no assistant message at
all**.  Such a conversation cannot be a continuation (a strict extension
always includes the previous turn's assistant echo) and contains nothing
that would need re-tokenizing, so it opens a **new conversation** on the
session — the full render starts a fresh stream, replacing any cached one on
commit.  Agents that run several sequential conversations under one session
id are served without ever re-tokenizing sampled content.

One more shape is *recovered* rather than rejected: a request that repeats
the session's previous request **verbatim**.  It cannot be a new turn (a
continuation appends at least the assistant echo plus one new message) — it
is exactly what an agent, or its SDK's automatic retry, resends when the
previous turn was generated and committed but the response was lost in
transit (an agent- or proxy-side timeout, a dropped connection).  The sampled
tokens are already part of the authoritative stream, so callers consult
:meth:`TokenStreamManager.cached_turn` before
:meth:`~TokenStreamManager.build_prompt` and re-serve the cached turn,
making lost-response retries idempotent instead of stranding the rollout.
This includes the session's opening request, which carries no assistant
message: only a *different* assistant-free request opens a fresh
conversation, because re-rendering a byte-identical one would sample the
(usually longest) first turn a second time and record a branch the agent
never saw.  Deliberate re-sampling of the same opening prompt belongs in a
new session; the replay cap says so once it is reached.

Concurrency: turns of one session must be sequential — concurrent requests
for the *same* session are not supported and may interleave state
arbitrarily (the rollout provider enforces this with a per-session lock
held across its generate flow).  Requests for *different* sessions may run
concurrently, and callers may offload ``build_prompt`` to a worker thread
(the rollout provider does, to keep tokenization off the event loop):
per-session state is only touched by that session's sequential turns, and
the shared bookkeeping consists of single CPython-atomic dict operations.
"""

from __future__ import annotations
import logging
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

#: Default generation prompt the manager appends to every engine prompt, in
#: place of the chat template's own (renders happen with
#: ``add_generation_prompt=False``).  Appending a fixed header rather than the
#: template's ``add_generation_prompt=True`` output matters when a template
#: pre-fills the start of the assistant turn (e.g. a reasoning opener the
#: served checkpoint must sample itself): the manager gives the engine the
#: header only and lets it sample the rest, and the recorded completion stays
#: free of forced-format header tokens.  The per-model value is **stated by
#: the model's profile** (:mod:`proxyserver.tokenization`); this ChatML value
#: is only the default for direct/test construction.  Why Qwen states exactly
#: this — a *bare* header, not the template's ``<think>``-prefilled one — is
#: documented on ``proxyserver/tokenization/qwen3_5.py``.
GENERATION_PROMPT = "<|im_start|>assistant\n"

#: Default marker closing an assistant message: how the template ends the
#: turn (here ChatML's ``<|im_end|>`` the model samples, then a newline).  The
#: engine stops *at* the sampled EOS, so a trailing newline the template
#: writes after it is never part of a completion — and the inter-turn delta
#: renders the new messages from scratch, so it does not supply it either.
#: Without appending it, every continuation prompt reads
#: ``...<|im_end|><|im_start|>user`` where the template writes
#: ``...<|im_end|>\n<|im_start|>user``: the engine prompt, and the training
#: stream reconstructed from the record, drift one token off the format the
#: checkpoint was trained on at *every* turn boundary.
#: :meth:`TokenStreamManager._turn_gap_ids` appends whichever suffix of this
#: the stream is still missing, so a turn truncated at ``max_tokens`` (which
#: sampled no EOS at all) is closed the way the template would close it.  The
#: per-model value is **stated by the model's profile**; this ChatML value is
#: only the default for direct/test construction.
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


def as_token_ids(rendered: Any, *, what: str) -> list[int]:
    """Normalize ``apply_chat_template(tokenize=True)`` to a flat ``list[int]``.

    The return type is not stable across tokenizers: ``transformers`` >= 5
    answers with a ``BatchEncoding`` (a ``Mapping``), <= 4 with a plain
    ``list[int]``, and a remote/HTTP tokenizer facade with whatever it parsed
    out of JSON.  Some templates wrap the ids in a batch of one.

    Passing a ``Mapping`` on unchecked is the dangerous case: ``list()`` of it
    silently yields its *keys* (``["input_ids", "attention_mask"]``), which
    would enter the session's authoritative token stream as two bogus tokens
    and corrupt every later prompt instead of raising.  So convert
    explicitly, and reject anything that is not a flat list of ints.
    """
    ids = rendered
    if isinstance(ids, Mapping):  # transformers>=5 BatchEncoding
        if "input_ids" not in ids:
            raise TokenStreamError(f"{what}: tokenizer returned a mapping without 'input_ids': {sorted(ids)}")
        ids = ids["input_ids"]
    if len(ids) and isinstance(ids[0], (list, tuple)):  # batch of conversations
        if len(ids) != 1:
            raise TokenStreamError(f"{what}: tokenizer returned {len(ids)} conversations, expected 1")
        ids = ids[0]
    if len(ids) and not isinstance(ids[0], int):
        raise TokenStreamError(f"{what}: tokenizer returned {type(ids[0]).__name__} elements, expected int token IDs")
    return list(ids)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize message content fields for ``apply_chat_template``.

    Handles:
    * Multi-part ``content`` lists (OpenAI vision format) -> plain strings.
    * ``content: null`` in assistant messages with ``tool_calls``.
    * ``tool`` role messages are passed through with ``tool_call_id``.
    """
    normalized = []
    for msg in messages:
        msg = dict(msg)
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif "text" in part:
                        text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            msg["content"] = "\n".join(text_parts) if text_parts else ""

        if role == "assistant" and "tool_calls" in msg:
            if msg.get("content") is None:
                msg["content"] = ""

        normalized.append(msg)
    return normalized


def _same_message(prev: Mapping[str, Any], cur: Mapping[str, Any]) -> bool:
    """Whether two messages are the same turn of a conversation."""
    return prev.get("role") == cur.get("role") and prev.get("content") == cur.get("content")


def _extends_conversation(previous: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> bool:
    """Whether ``candidate`` strictly extends ``previous``.

    Used by :meth:`TokenStreamManager.build_prompt` to spot a continuation
    that dropped its assistant echo.  It shares :func:`_same_message` with
    the inter-turn delta (which compares pairwise itself, to name the
    offending index): answering "is this a continuation?" with two
    *different* equality tests is what let a request carrying one extra
    message key slip past this check and silently reset the token stream.
    """
    return len(candidate) > len(previous) and all(
        _same_message(prev, cur) for prev, cur in zip(previous, candidate)
    )


#: How many times one committed turn may be re-served to an exact repeat of
#: its request.  A lost-response retry needs one, a retried retry two; past
#: that the caller is re-issuing the request on purpose (hoping to re-sample)
#: and would otherwise be handed the byte-identical completion forever, with
#: no error to break the loop on.  See :meth:`TokenStreamManager.cached_turn`.
MAX_TURN_REPLAYS = 4


class _SessionState:
    """Authoritative token stream and message history of one session."""

    __slots__ = ("stream", "messages", "last_turn", "replays")

    def __init__(
        self,
        stream: list[int],
        messages: list[dict[str, Any]],
        last_turn: tuple[int, list[float], str, dict[str, Any]] | None = None,
    ):
        self.stream = stream        # prompt + sampled completion tokens
        self.messages = messages    # normalized messages of the last request
        # (completion length, logprobs, finish_reason, engine_meta) of the last
        # committed turn, when the committer supplied them — lets an exact
        # retry of that request be re-served instead of rejected (see
        # cached_turn).  engine_meta is cached with the rest so a re-served
        # turn still reports the engine state it was actually sampled under.
        self.last_turn = last_turn
        # How often this turn has already been re-served to an exact repeat.
        self.replays = 0


class TokenStreamManager:
    """Builds strictly token-in-token-out prompts across turns of a session.

    Usage (one instance per tokenizer, shared across sessions)::

        manager = TokenStreamManager(tokenizer)
        cached = manager.cached_turn(session_id, messages)
        if cached is not None:
            ...  # lost-response retry: re-serve, do not sample again
        prompt_ids = manager.build_prompt(session_id, messages, tools)
        ...  # feed prompt_ids to the engine, sample completion_ids
        manager.commit(session_id, messages, prompt_ids, completion_ids,
                       log_probs, finish_reason,   # these two enable cached_turn
                       engine_meta)                # cached with them, re-served with them

    Messages are always rendered with ``add_generation_prompt=False`` and the
    generation prompt is appended by the manager itself — the template file
    stays stock while the engine never sees the template's ``<think>\n``
    pre-fill, which the served checkpoint cannot continue from (see
    :data:`GENERATION_PROMPT`).  The generation prompt and the assistant turn
    end (``generation_prompt`` / ``assistant_turn_end``) are **stated by the
    caller** — the model registry passes what the model's profile declares
    (:mod:`proxyserver.tokenization`), so ChatML (Qwen: ``<|im_start|>`` /
    ``<|im_end|>``) and Llama (``<|start_header_id|>`` / ``<|eot_id|>``) are
    each served correctly; the ChatML defaults serve direct/test construction.
    Any implicit per-render prefix the template prepends (a bos token, a
    default system block) is measured and stripped from inter-turn deltas
    (see :meth:`_measure_implicit_prefix`).

    ``tokenizer`` needs two methods::

        apply_chat_template(messages, tools=..., add_generation_prompt=...,
                            tokenize=True) -> list[int] | BatchEncoding
        encode(text, add_special_tokens=False) -> list[int]
        # (or being callable: tokenizer(text, add_special_tokens=False)
        #  returning a mapping with "input_ids")

    Both return shapes are accepted (``transformers`` >= 5 answers with a
    ``BatchEncoding``); see :func:`as_token_ids`.

    Raises :class:`TokenStreamError` whenever a prompt cannot be built as a
    strict extension of the session's token stream. Note that the strict
    first-turn rule — no assistant messages before a stream exists — also
    rejects few-shot assistant examples in the opening request; that is
    deliberate, as the manager cannot distinguish them from previously
    sampled content that would have to be re-tokenized.

    A request with no assistant message is always treated as the opening
    turn of a (possibly new) conversation: when the session already holds a
    stream, that stream is replaced on the next :meth:`commit` — see the
    module docstring.
    """

    # Roles allowed in the inter-turn gap (mirrors the trainer-side filter).
    _GAP_ROLES = ("tool", "user", "system")

    def __init__(
        self,
        tokenizer: Any,
        max_sessions: int = 4096,
        generation_prompt: str = GENERATION_PROMPT,
        max_turn_replays: int = MAX_TURN_REPLAYS,
        assistant_turn_end: str = ASSISTANT_TURN_END,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_sessions = max_sessions
        self.max_turn_replays = max_turn_replays
        # The two markers are model-format-specific and **stated** by the
        # caller — the model registry passes the values the model's profile
        # declares (see proxyserver.tokenization).  The ChatML defaults serve
        # direct/test construction; the implicit per-render prefix the template
        # prepends (a bos token, a default system block) is still measured and
        # stripped from deltas (self-validating; see _measure_implicit_prefix).
        self.generation_prompt = generation_prompt
        self.assistant_turn_end = assistant_turn_end
        self._sessions: OrderedDict[str, _SessionState] = OrderedDict()
        # Session ids dropped by LRU eviction, kept (bounded, value-less) only
        # so their later requests can say *why* they can no longer be served —
        # otherwise an evicted rollout fails every remaining turn with a
        # message about assistant content that reads like the agent's fault.
        self._evicted: OrderedDict[str, None] = OrderedDict()
        self._sys_ids: list[int] | None = None  # lazy; [] if template rejects []
        self._sys_ids_measured = True           # False once the [] probe failed
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
    ) -> list[int]:
        """Build the prompt token IDs for the next completion of a session."""
        if session_id is None:
            raise TokenStreamError("Strict token-in-token-out requires a session_id")

        normalized = normalize_messages(messages)
        state = self._sessions.get(session_id)

        first_assistant = next(
            (i for i, msg in enumerate(normalized) if msg.get("role") == "assistant"),
            None,
        )
        if first_assistant is None:
            # Opening request of a conversation: with no assistant message
            # anywhere, nothing would need re-tokenizing, and a strict
            # continuation always carries the previous turn's assistant echo
            # — so the full render *defines* the stream start.  With cached
            # state this means the agent opened a NEW conversation on the
            # same session (e.g. the next phase of a multi-phase task, or a
            # retried first turn); the fresh stream replaces the old one on
            # commit.
            if state is not None:
                if _extends_conversation(state.messages, normalized):
                    # ...except when it extends the previous request exactly:
                    # a new conversation shares no history, so this is a
                    # continuation that *lost* its assistant echo, and
                    # re-rendering it would silently drop the sampled tokens
                    # of the turn it is answering.  Refuse rather than
                    # corrupt: the rollout is recoverable, the stream is not.
                    # (The usual cause is an agent dropping the echo because
                    # a tool-call-only turn has content: null.)
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
                self._sessions.move_to_end(session_id)
            return self._full_render(normalized, tools)

        if state is None:
            # Sampled-looking content for a session with no cached stream
            # would have to be re-tokenized from text, which TITO forbids.
            if session_id in self._evicted:
                raise TokenStreamError(
                    f"Session {session_id}: its token stream was evicted "
                    f"(LRU) before this turn, so the rollout cannot continue "
                    f"— every remaining turn of this session will fail the "
                    f"same way.  More than max_sessions={self.max_sessions} "
                    f"sessions held streams at once: delete finished sessions "
                    f"(DELETE /sessions/{{id}}) so they release their slots, "
                    f"or raise the cap where TokenStreamManager is built."
                )
            raise TokenStreamError(
                f"Session {session_id}: no token stream exists but the "
                f"request carries an assistant message at position "
                f"{first_assistant}; refusing to re-tokenize assistant content"
            )

        self._sessions.move_to_end(session_id)
        return state.stream + self._delta_ids(session_id, state, normalized)

    def cached_turn(
        self,
        session_id: str | None,
        messages: list[dict[str, Any]],
    ) -> tuple[list[int], list[int], list[float], str, dict[str, Any]] | None:
        """The last committed turn, iff ``messages`` repeats its request verbatim.

        **Consuming**: a hit spends one of the turn's
        :data:`MAX_TURN_REPLAYS` replays, so call this once per turn served,
        not as a side-effect-free predicate.

        A request that exactly equals the session's previous request cannot
        be a new turn — a strict continuation appends at least the assistant
        echo plus one new message — but it is exactly what an agent (or its
        SDK's automatic retry) resends when the previous turn was generated
        and committed yet its response was lost in transit.  Rejecting it
        would strand the rollout, because the sampled tokens are already part
        of the authoritative stream; callers re-serve the cached turn
        instead, making lost-response retries idempotent.

        This covers the session's **opening** request too, which carries no
        assistant message: a fresh conversation is what a *different*
        assistant-free request opens, but a byte-identical repeat of the one
        just served is a lost first turn — the longest and most expensive of
        most rollouts — and re-rendering it would sample it a second time and
        record a branch the agent never saw.  A caller that means to
        re-sample the same opening prompt opens a new session (the replay cap
        below says so once it is reached).

        Re-serving is capped at :data:`MAX_TURN_REPLAYS` per committed turn.
        A lost-response retry needs one replay; a caller that keeps re-issuing
        the identical request is not recovering a lost response but asking to
        re-sample, and answering it from the cache forever would hand it the
        same completion — including the same truncated one it is trying to get
        past — with no error to break the loop on.  Past the cap the repeat is
        refused like any other request that cannot extend the stream.

        Returns:
            ``(prompt_ids, completion_ids, log_probs, finish_reason,
            engine_meta)`` of the cached turn, or ``None`` when this is not
            such a repeat (or the committer supplied no logprobs/finish_reason
            to cache).
        """
        if session_id is None:
            return None
        state = self._sessions.get(session_id)
        if state is None or state.last_turn is None:
            return None
        normalized = normalize_messages(messages)
        if normalized != state.messages:
            return None
        if state.replays >= self.max_turn_replays:
            raise TokenStreamError(
                f"Session {session_id}: this request has already been "
                f"re-served {state.replays} times from the cached turn and is "
                f"being repeated verbatim again; the proxy cannot re-sample it "
                f"without forking the session's token stream.  Continue the "
                f"conversation (echo the assistant turn and append the new "
                f"messages) or open a new session to sample afresh."
            )
        state.replays += 1
        completion_len, log_probs, finish_reason, engine_meta = state.last_turn
        self._sessions.move_to_end(session_id)
        return (
            state.stream[:-completion_len],
            state.stream[-completion_len:],
            list(log_probs),
            finish_reason,
            dict(engine_meta),
        )

    def commit(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        prompt_ids: list[int],
        completion_ids: list[int],
        log_probs: list[float] | None = None,
        finish_reason: str | None = None,
        engine_meta: dict[str, Any] | None = None,
    ) -> None:
        """Record the sampled completion as the new authoritative stream.

        ``log_probs`` and ``finish_reason`` are optional; when both are
        supplied the turn is also cached so an exact retry of this request —
        a lost-response recovery — can be re-served via :meth:`cached_turn`.
        ``engine_meta`` (what the engine reported about the turn) is cached
        alongside them, so a re-serve reports the engine state the tokens were
        sampled under rather than the current one.
        """
        last_turn = None
        if completion_ids and log_probs is not None and finish_reason is not None:
            last_turn = (len(completion_ids), list(log_probs), finish_reason, dict(engine_meta or {}))
        self._sessions[session_id] = _SessionState(
            stream=list(prompt_ids) + list(completion_ids),
            messages=normalize_messages(messages),
            last_turn=last_turn,
        )
        self._sessions.move_to_end(session_id)
        self._evicted.pop(session_id, None)  # committing revives an evicted id
        while len(self._sessions) > self.max_sessions:
            evicted, _ = self._sessions.popitem(last=False)
            self._evicted[evicted] = None
            if len(self._evicted) > self.max_sessions:  # one in, at most one out
                self._evicted.popitem(last=False)
            logger.warning(
                "TokenStreamManager evicted session %s (LRU, max_sessions=%d); "
                "every further turn of it will be rejected — delete finished "
                "sessions so they release their slots, or raise the cap where "
                "TokenStreamManager is built",
                evicted, self.max_sessions,
            )

    def drop(self, session_id: str) -> None:
        """Forget a session's token stream (call on session deletion)."""
        self._sessions.pop(session_id, None)
        self._evicted.pop(session_id, None)

    def has_session(self, session_id: str) -> bool:
        """Whether this manager currently holds a token stream for ``session_id``."""
        return session_id in self._sessions

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _full_render(self, normalized: list[dict[str, Any]], tools: list[dict] | None) -> list[int]:
        # add_generation_prompt=False + the manager's own header — never the
        # template's generation prompt (see GENERATION_PROMPT).
        return as_token_ids(
            self.tokenizer.apply_chat_template(
                normalized,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
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

        # Render only the new messages and strip the template's implicit
        # system prefix — same scheme as the trainer's
        # _tokenize_tool_messages, so both sides build identical streams.
        # The generation prompt is the manager's own header, appended after
        # the strip (see GENERATION_PROMPT).
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
                # The template prepends a fixed boilerplate prefix (a bos
                # token, a default system block, ...) to every render, measured
                # once and normally stripped here — but this delta does not
                # begin with it.  That happens when a new inter-turn message is
                # one the template folds into its *top-of-conversation* block
                # instead of rendering as a plain mid-stream turn: Llama renders
                # a system message as its dated top system block, bos included,
                # so the measured (empty-system) prefix is no longer a clean
                # prefix.  Splicing that boilerplate mid-stream would put a bos
                # / date preamble inside the authoritative stream, off the
                # format the checkpoint was trained on — refuse instead.
                raise TokenStreamError(
                    f"Session {session_id}: the inter-turn delta for new "
                    f"messages {[m.get('role') for m in new_msgs]} does not "
                    f"begin with the template's measured per-render prefix, so "
                    f"the template renders them as part of its "
                    f"top-of-conversation block (bos / default system preamble) "
                    f"rather than a mid-stream turn.  This is usually a system "
                    f"message mid-conversation, which this template cannot place "
                    f"token-in-token-out; keep system messages in the opening "
                    f"turn."
                )
        self._reject_unstripped_system_prefix(session_id, new_msgs, delta)
        # The template closes the assistant message with "<|im_end|>\n"; the
        # engine stopped at the sampled "<|im_end|>" (or was truncated before
        # even that), so the manager supplies whatever of it is missing —
        # otherwise the turn boundary reads "<|im_end|><|im_start|>user"
        # instead of the template's "<|im_end|>\n<|im_start|>user" (see
        # ASSISTANT_TURN_END).
        return self._turn_gap_ids(state.stream) + delta + self._generation_prompt_ids()

    def _turn_gap_ids(self, stream: list[int]) -> list[int]:
        """The tokens that must sit between ``stream`` and the next render.

        The chat template ends every assistant message with
        :data:`ASSISTANT_TURN_END`; the stream ends wherever generation
        stopped.  Return the suffix of that ending the stream does not
        already have, so a natural stop (which sampled the ``<|im_end|>``)
        contributes only the trailing newline while a truncated turn is
        closed in full.
        """
        end = self._turn_end_ids()
        for have in range(len(end), 0, -1):
            if stream[-have:] == end[:have]:
                return end[have:]
        return list(end)

    def _reject_unstripped_system_prefix(self, session_id: str, new_msgs: list[dict[str, Any]], delta: list[int]) -> None:
        """Refuse a delta that still carries the template's implicit system prefix.

        :meth:`_get_sys_ids` measures that prefix by rendering an empty
        conversation, which some templates refuse outright (the bundled
        Qwen3.5 among them).  It then assumes there is no prefix — right for
        Qwen3.5, but a template that both refuses ``[]`` *and* prepends a
        system block would splice a second system prompt into the middle of
        the authoritative stream, silently, on every tool turn.  A delta that
        opens with the template's system header while its first new message
        is not a system message is exactly that case: fail the turn instead.
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
                f"not be measured (apply_chat_template([]) is unsupported by "
                f"this template), so appending the delta would inject a second "
                f"system prompt mid-stream.  Refusing rather than corrupting "
                f"the session's token stream."
            )

    def _get_sys_ids(self) -> list[int]:
        """The implicit prefix every render carries, measured once.

        Some templates prepend a fixed block to *every* render — a ``bos``
        token (Llama), a default dated system message (Llama 3.1), or both —
        that belongs at the start of the stream but must not be repeated in
        every inter-turn delta.  It is measured position-independently (see
        :meth:`_measure_implicit_prefix`), which works even for the templates
        that reject an empty conversation (Qwen3.5 raises; Llama would inject
        its default system block).  A template whose prefix cannot be measured
        leaves it *unmeasured* (``_sys_ids_measured=False``, empty) so
        :meth:`_reject_unstripped_system_prefix` guards deltas instead of the
        stream being silently corrupted.
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
        """Token IDs of :data:`ASSISTANT_TURN_END` (tokenized once)."""
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
