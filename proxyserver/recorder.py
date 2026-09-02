"""Thread-safe session data recorder for the LLM proxy."""

from __future__ import annotations
import base64
import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any
from .rollout_sessions import CompletionRecord, RoutedExpertsBlob, SessionRecord

logger = logging.getLogger(__name__)

# Default root for persisted session records: ./proxyserver/sessions
DEFAULT_SESSION_ROOT = Path(__file__).resolve().parent / "sessions"

# How long a deleted session's id stays tombstoned.  Deletion races the
# session's own in-flight turns: the driver deletes a cancelled trial while
# a turn is still generating, and the turn-preserving delivery paths (the
# relay's orphan recording, local mode's detached pipelines) then hand that
# turn to the recorder minutes later — un-tombstoned it would silently
# re-create the just-deleted record, which nothing ever deletes again.
_TOMBSTONE_TTL = 3600.0
# Bound on remembered tombstones, matching the relay's expired-table cap.
# Oldest fall off first — their in-flight turns are long finished.
_MAX_TOMBSTONES = 4096


class SessionStore:
    """Persists :class:`SessionRecord` JSON to disk, one file per session.

    Mirrors the layout of the per-session error logs: all sessions of one
    proxy run share a directory keyed by the proxy's operating-mode label
    (``relay``, ``slime``, ``direct``, ...) and stamped with the server
    start time:
    ``{root}/{mode}/{yyyy}-{mm}-{dd}-{hh}-{mm}-{ss}/{session_id}.json``.

    A session is rewritten in full after every recorded turn, so a rollout
    that crashes — or whose driver never calls ``/sessions/{id}/complete``
    — still leaves its turns on disk.
    """

    def __init__(self, root: Path, mode: str):
        self._run_dir = root / mode / time.strftime("%Y-%m-%d-%H-%M-%S")
        self._lock = threading.Lock()
        # Highest version written per session; stale flushes are skipped so
        # concurrent writers can never regress the file on disk.
        self._latest_version: dict[str, int] = {}

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def save(self, session_id: str, payload: str, version: int = 0) -> None:
        """Write one session's JSON.  Never raises — persistence failures
        must not break request serving.

        ``version`` is the caller's per-session monotonic snapshot counter
        (see :meth:`SessionRecorder._snapshot`); a payload older than one
        already written for the session is dropped instead of overwriting
        the newer state."""
        try:
            with self._lock:
                if version < self._latest_version.get(session_id, 0):
                    return
                self._latest_version[session_id] = version
                self._run_dir.mkdir(parents=True, exist_ok=True)
                path = self._run_dir / f"{session_id}.json"
                tmp = path.with_name(f"{session_id}.json.tmp")
                tmp.write_text(payload, encoding="utf-8")
                tmp.replace(path)
        except OSError as e:
            logger.warning("Cannot save session record for %s: %s", session_id, e)


def _conversation_extent(turns: list[CompletionRecord]) -> tuple[int, int]:
    """Extent of the recorded conversation the next turn must extend:
    ``(non-assistant message count of the previous request, token-stream
    length after the previous turn)``.

    ``turns`` are the recorded turns of **one agent** (streams are
    per-agent; the caller filters the session's interleaved record by
    ``agent_id``).  Turns store deltas (see :class:`CompletionRecord`), so
    both values are sums over the current conversation — the turns since
    the last ``new_conversation`` marker: each request's message count
    is the previous one plus its delta, and the stream after each turn
    grows by exactly its prompt delta plus its sampled completion.

    Messages are counted **excluding assistant messages** on both sides of
    every comparison: a continuation's stored delta omits the assistant echo
    (it is the previous turn's completion — see :class:`CompletionRecord`),
    so the stored counts only determine the request's non-assistant extent.
    That is enough: under strict TITO a continuation's non-assistant
    messages strictly extend the previous request's, so the count both
    locates the delta and spots a duplicate delivery.
    """
    start = 0
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].new_conversation:
            start = i
            break
    n_messages = 0
    stream_len = 0
    for turn in turns[start:]:
        n_messages += sum(1 for m in turn.request_messages if m.get("role") != "assistant")
        stream_len += len(turn.prompt_token_ids) + len(turn.completion_token_ids)
    return n_messages, stream_len


class SessionRecorder:
    """Manages session records for the LLM proxy.

    Turns are stored **delta-only**: ``record_completion`` receives each
    request's full conversation and full engine prompt but keeps only what
    the request appended beyond the previous turn (full history per turn
    would grow the record quadratically) — see :class:`CompletionRecord`
    for the reconstruction scheme.  Deltas are computed **per agent**: a
    multi-agent rollout's agents each own an independent token stream, and
    their turns interleave in one session record tagged with ``agent_id``.

    Thread-safe, and actually used from threads: the proxy offloads
    recording, completion, and dumping to worker threads because they
    serialize the full session (which grows every turn) and must not stall
    the event loop.  All state access synchronizes on one lock, and flushes
    carry per-session monotonic versions (:meth:`_snapshot`) so a slow
    flush can never overwrite a newer one on disk.

    When a :class:`SessionStore` is supplied, every mutation of a session
    is also flushed to disk.  ``save_logprobs=False`` omits
    ``completion_logprobs`` from those disk snapshots only — the in-memory
    record (and everything served from it, e.g. ``GET /sessions/{id}``)
    always carries the logprobs.  ``save_routed_experts`` gates the
    session's ``routed_experts`` blobs the same way, but defaults **off**:
    the store rewrites the full session JSON after every turn, and a blob
    covering a long stream re-written per turn is gigabytes of write
    amplification for data the trainer fetches over ``GET /sessions/{id}``
    anyway.
    """

    def __init__(
        self,
        store: SessionStore | None = None,
        save_logprobs: bool = True,
        save_routed_experts: bool = False,
    ) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()
        self._store = store
        self._save_logprobs = save_logprobs
        self._save_routed_experts = save_routed_experts
        # Per-session snapshot counter (bumped under the lock) so flushes
        # that happen outside the lock cannot land out of order on disk.
        self._versions: dict[str, int] = {}
        # session_id -> monotonic deletion time, insertion order = deletion
        # order.  Turns delivered after deletion are dropped instead of
        # resurrecting the record — see delete_session.
        self._deleted: OrderedDict[str, float] = OrderedDict()

    def ensure_session(self, session_id: str, model_name: str | None = None) -> None:
        """Create the session only if it does not exist yet.

        The agent's first completion request implicitly opens the session;
        an existing record is left untouched so later turns of the same
        session never reset it.  ``model_name`` (the request's validated
        model) is stamped once, by the first request that carries it.

        A tombstoned id (see :meth:`delete_session`) is not re-created: the
        session was deleted and whoever is asking is a straggler of a
        rollout that is already over.
        """
        with self._lock:
            if self._tombstoned(session_id):
                return
            session = self._sessions.get(session_id)
            if session is None:
                self._sessions[session_id] = SessionRecord(session_id=session_id, model_name=model_name)
            elif session.model_name is None and model_name:
                session.model_name = model_name

    def record_completion(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        completion_text: str,
        token_ids: list[int],
        logprobs: list[float],
        finish_reason: str | None = None,
        prompt_token_ids: list[int] | None = None,
        agent_id: str | None = None,
        content: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        weight_version: str | None = None,
        routed_experts: dict[str, Any] | None = None,
    ) -> None:
        """Record a single LLM completion for a session.

        ``agent_id`` names the agent (of a multi-agent rollout) whose turn
        this is. The proxy's HTTP API always supplies a real agent_id now
        (keyed api_keys require one); ``None`` remains only as this
        method's default for direct/programmatic callers that don't
        distinguish agents. Streams are per-agent, so the delta baseline
        and the duplicate check below consider only this agent's turns —
        the session's record interleaves all agents' turns in arrival order.

        ``messages`` / ``prompt_token_ids`` arrive as the request's full
        conversation and full engine prompt, but only their delta beyond the
        agent's previous turn is stored — and the assistant echo that opens
        a continuation's delta is dropped too, being just the previous
        turn's recorded completion — see :class:`CompletionRecord`.

        A duplicate delivery of the agent's **last recorded turn** is
        dropped, whether that turn opened a conversation or continued one: a
        lost-response retry is re-served the already-committed turn
        (``token_stream.cached_turn``), and a late relay delivery of the
        original response can race it (``server._record_orphan_response``)
        — both funnel here, and the turn must be recorded exactly once.

        A turn for a tombstoned session (see :meth:`delete_session`) is
        dropped entirely: the session was deleted while the turn was in
        flight — typically a cancelled trial — and recording it would
        re-create a record nobody will ever fetch or delete again.

        ``routed_experts`` is the turn's repacked capture blob (see
        :class:`~proxyserver.rollout_sessions.RoutedExpertsBlob`), stored
        **per agent** — never per turn — and only forward.  A delta
        (``start`` = the rows already stored, same ``cols``, on a turn that
        extends the stream) is appended; a whole-stream blob (``start`` 0)
        supersedes when it is at least as wide as what is stored; anything
        else — a late delivery of an older turn racing a newer one, a delta
        against a stream that was reset — is dropped with a warning rather
        than corrupting the stored coverage.  A duplicate delivery
        early-returns above without touching it.
        """
        full_prompt_ids = list(prompt_token_ids or [])
        with self._lock:
            if self._tombstoned(session_id):
                logger.info(
                    "Session %s was deleted; dropping a turn delivered after "
                    "deletion (%d completion tokens) instead of re-creating "
                    "the record",
                    session_id, len(token_ids),
                )
                return
            session = self._sessions.get(session_id)
            if session is None:
                logger.warning("Session %s not found, auto-creating for recording", session_id)
                session = SessionRecord(session_id=session_id)
                self._sessions[session_id] = session

            # This agent's own turns: its stream is independent of the other
            # agents', whose turns interleave arbitrarily in session.turns.
            agent_turns = [t for t in session.turns if t.agent_id == agent_id]

            # A request with no assistant message opens a (new) conversation
            # — the same rule the TITO layer applies — and stores its full
            # render; a continuation stores only what it appended.  A
            # continuation with no recorded predecessor (auto-created
            # session) is also stored in full so nothing is lost.
            new_conversation = (
                not any(msg.get("role") == "assistant" for msg in messages)
                or not agent_turns
            )
            if new_conversation and agent_turns:
                # An opening turn stores its request in full, so a duplicate
                # delivery of one is spotted by plain equality.  It needs its
                # own check: the continuation dedup below never runs for these,
                # and without this a re-served first-turn retry (or a late
                # orphan racing it) lands in the record twice — a phantom
                # branch the agent never saw, trained on as if it had.
                last = agent_turns[-1]
                if (
                    last.new_conversation
                    and messages == last.request_messages
                    and token_ids == last.completion_token_ids
                    and full_prompt_ids == last.prompt_token_ids
                ):
                    logger.info(
                        "Session %s: skipping duplicate delivery of agent %s's "
                        "last recorded opening turn (%d completion tokens)",
                        session_id, agent_id, len(token_ids),
                    )
                    return
            elif not new_conversation:
                n_prev, stream_len = _conversation_extent(agent_turns)
                # Assistant messages are not stored on continuation turns:
                # the echo a continuation replays is the previous turn's
                # completion (completion_text / tool_calls), so storing it
                # again would duplicate data every turn.  Counting and
                # slicing therefore consider only non-assistant messages —
                # see _conversation_extent.
                non_assistant = [m for m in messages if m.get("role") != "assistant"]
                last = agent_turns[-1]
                if (
                    len(non_assistant) == n_prev
                    and token_ids == last.completion_token_ids
                    and len(full_prompt_ids) == stream_len - len(last.completion_token_ids)
                ):
                    # Duplicate delivery of the agent's last recorded turn:
                    # same request messages, same completion, same engine
                    # prompt.  A legitimate new turn can never match (a
                    # continuation appends at least one non-assistant message
                    # beyond the echo, and its prompt covers the agent's full
                    # recorded stream).
                    logger.info(
                        "Session %s: skipping duplicate delivery of agent %s's "
                        "last recorded turn (%d completion tokens)",
                        session_id, agent_id, len(token_ids),
                    )
                    return
                if len(non_assistant) < n_prev or len(full_prompt_ids) < stream_len:
                    # Numerically cannot extend what was recorded (e.g. a
                    # worker that reported no prompt ids) — store the turn in
                    # full rather than a corrupt slice.
                    logger.warning(
                        "Session %s (agent %s): turn does not extend the "
                        "recorded stream (%d non-assistant messages / %d "
                        "prompt tokens vs %d / %d already recorded); storing "
                        "the turn in full",
                        session_id, agent_id, len(non_assistant), len(full_prompt_ids), n_prev, stream_len,
                    )
                    new_conversation = True
                else:
                    messages = non_assistant[n_prev:]
                    full_prompt_ids = full_prompt_ids[stream_len:]
            if routed_experts is not None and agent_id is not None:
                blob = RoutedExpertsBlob.model_validate(routed_experts)
                prev = session.routed_experts.get(agent_id)
                if (
                    blob.start and prev is not None and blob.start == prev.rows
                    and blob.cols == prev.cols and not new_conversation
                ):
                    merged = base64.b64decode(prev.data) + base64.b64decode(blob.data)
                    session.routed_experts[agent_id] = RoutedExpertsBlob(
                        data=base64.b64encode(merged).decode("ascii"),
                        rows=prev.rows + blob.rows, cols=prev.cols, dtype=prev.dtype,
                    )
                elif blob.start == 0 and (prev is None or blob.rows >= prev.rows):
                    session.routed_experts[agent_id] = blob
                else:
                    logger.warning(
                        "Session %s (agent %s): dropping a routed-experts blob that "
                        "neither extends nor supersedes the stored one (start %d, "
                        "rows %d, cols %d vs stored rows %s, cols %s)",
                        session_id, agent_id, blob.start, blob.rows, blob.cols,
                        prev.rows if prev else None, prev.cols if prev else None,
                    )
            record = CompletionRecord(
                agent_id=agent_id,
                request_messages=messages,
                prompt_token_ids=full_prompt_ids,
                new_conversation=new_conversation,
                completion_text=completion_text,
                completion_token_ids=token_ids,
                completion_logprobs=logprobs,
                finish_reason=finish_reason,
                weight_version=weight_version,
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
            )
            session.turns.append(record)
            payload, version = self._snapshot(session)
        self._save(session_id, payload, version)

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Retrieve session data. Returns None if not found."""
        with self._lock:
            return self._sessions.get(session_id)

    def dump_session(self, session_id: str) -> dict[str, Any] | None:
        """Serialize a session to a plain dict under the lock — a consistent
        snapshot even while another thread records a turn.  None if unknown."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return session.model_dump()

    def mark_completed(self, session_id: str) -> None:
        """Mark a session as completed."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.completed = True
            payload, version = self._snapshot(session)
        self._save(session_id, payload, version)

    def delete_session(self, session_id: str) -> None:
        """Remove a session, free its memory, and tombstone its id.

        Any persisted JSON is kept: the record on disk outlives the
        in-memory one, which is the point of saving it.

        The tombstone exists because deletion races the session's own
        in-flight turns.  The driver deletes a cancelled trial while a turn
        is still generating, and the turn-preserving delivery paths — the
        relay's orphan recording, local mode's detached pipelines — would
        then hand that turn to :meth:`record_completion` minutes later,
        silently re-creating the record; the driver deletes exactly once,
        so nothing would ever free it again.  While tombstoned (see
        :meth:`is_deleted`), such turns are dropped and the id's requests
        are refused at the endpoint.  The tombstone expires after
        ``_TOMBSTONE_TTL``, so an id deliberately re-created later starts
        fresh; within the window a re-created id is refused loudly rather
        than leak a record per cancelled trial.
        """
        with self._lock:
            self._sessions.pop(session_id, None)
            self._deleted[session_id] = time.monotonic()
            self._deleted.move_to_end(session_id)
            self._prune_tombstones()
            # _versions is deliberately kept: the store's stale-write guard
            # is monotonic per session_id, so a re-created id must keep
            # counting where it left off (costs one int per id ever seen).

    def is_deleted(self, session_id: str) -> bool:
        """Whether the session was deleted within the tombstone window.

        True means the id belongs to a deleted (typically cancelled)
        rollout whose stragglers may still be landing: its turns must be
        dropped and its new requests refused.
        """
        with self._lock:
            return self._tombstoned(session_id)

    def _tombstoned(self, session_id: str) -> bool:
        """Live-tombstone check; call under the lock."""
        deleted_at = self._deleted.get(session_id)
        return deleted_at is not None and time.monotonic() - deleted_at < _TOMBSTONE_TTL

    def _prune_tombstones(self) -> None:
        """Drop expired tombstones and bound the table; call under the lock.

        Insertion order is deletion order, so the expiry scan stops at the
        first still-live entry."""
        cutoff = time.monotonic() - _TOMBSTONE_TTL
        while self._deleted:
            deleted_at = next(iter(self._deleted.values()))
            if deleted_at >= cutoff:
                break
            self._deleted.popitem(last=False)
        while len(self._deleted) > _MAX_TOMBSTONES:
            self._deleted.popitem(last=False)

    # -- persistence -------------------------------------------------------

    def _snapshot(self, session: SessionRecord) -> tuple[str | None, int]:
        """Serialize a session while the lock is held (consistent snapshot)
        and stamp it with a per-session monotonic version, so the store can
        drop a stale flush that lands after a newer one.

        Serialized via ``model_dump()`` + ``json.dumps`` (rather than
        ``model_dump_json``) so the logprobs opt-out below can prune the
        dump without widening the record model; ``ensure_ascii=False`` and
        compact separators keep the on-disk shape pydantic produced
        before.  A turn's ``content`` / ``reasoning_content`` /
        ``tool_calls`` is dropped when None — the null carries no
        information, so the disk record omits the key (the in-memory
        record, like the logprobs, always carries all fields)."""
        if self._store is None:
            return None, 0
        version = self._versions.get(session.session_id, 0) + 1
        self._versions[session.session_id] = version
        data = session.model_dump()
        if not self._save_routed_experts or not data["routed_experts"]:
            # The blobs are large and re-written with every turn's full
            # rewrite; the empty dict of a capture-off session carries no
            # information either way.
            data.pop("routed_experts", None)
        for turn_data in data["turns"]:
            if not self._save_logprobs:
                turn_data.pop("completion_logprobs", None)
            for key in ("content", "reasoning_content", "tool_calls"):
                if turn_data[key] is None:
                    del turn_data[key]
        return json.dumps(data, ensure_ascii=False, separators=(",", ":")), version

    def _save(self, session_id: str, payload: str | None, version: int) -> None:
        """Flush a serialized session to disk, outside the lock."""
        if payload is not None and self._store is not None:
            self._store.save(session_id, payload, version)
