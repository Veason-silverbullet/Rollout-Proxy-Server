"""Thread-safe session data recorder for the LLM proxy."""

from __future__ import annotations
import base64
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from pydantic import ValidationError
from .rollout_sessions import CompletionRecord, RoutedExpertsBlob, SessionRecord
from .token_stream import normalize_messages

logger = logging.getLogger(__name__)

# Deleted ids are tombstoned so late requests and deliveries cannot recreate
# them; the oldest fall off first once the bound is reached.
_TOMBSTONE_TTL = 3600.0
_MAX_TOMBSTONES = 8192


@dataclass
class _FailedTurn:
    """An unresolved recording error and the position an exact retry can repair."""

    messages: list[dict[str, Any]]
    prompt_token_ids: list[int]
    token_ids: list[int]
    recorded_turns: int


class SessionStore:
    """Persists :class:`SessionRecord` JSON to
    ``{root}/{server-start-time}/{session_id}.json``, rewritten after every turn.
    """

    def __init__(self, root: Path):
        self._run_dir = root.expanduser().resolve() / time.strftime("%Y-%m-%d-%H-%M-%S")
        self._lock = threading.Lock()
        self._latest_version: dict[str, int] = {}

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def save(self, session_id: str, payload: str, version: int = 0) -> None:
        """Write one session's JSON, skipping payloads older than ``version``
        already written. Never raises."""
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
    """``(non-assistant message count, token-stream length)`` of one agent's
    current conversation, summed over its turns since the last
    ``new_conversation``. Assistant echoes are not stored, so they are not
    counted."""
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


def _merge_routed_experts(
    session: SessionRecord,
    session_id: str,
    agent_id: str,
    routed_experts: dict[str, Any],
) -> None:
    """Fold one turn's routed-experts blob into the agent's stored coverage.

    A delta whose ``start`` matches the stored rows is appended; a whole-stream
    blob supersedes when it covers at least as many rows. Anything else is dropped
    with a warning: the turn is already committed, and R3 conversion fails later
    on the missing coverage. Call under the recorder lock, after clearing prior
    coverage for a new conversation.
    """
    try:
        blob = RoutedExpertsBlob.model_validate(routed_experts)
    except ValidationError as e:
        logger.warning(
            "Session %s (agent %s): dropping a malformed routed-experts blob (%s); the turn itself is still recorded",
            session_id, agent_id, e,
        )
        return
    prev = session.routed_experts.get(agent_id)
    if blob.start and prev is not None and blob.start == prev.rows and blob.cols == prev.cols:
        try:
            merged = base64.b64decode(prev.data) + base64.b64decode(blob.data)
        except ValueError as e:
            logger.warning(
                "Session %s (agent %s): dropping an undecodable routed-experts "
                "delta (%s); keeping the stored coverage",
                session_id, agent_id, e,
            )
            return
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


class SessionRecorder:
    """Thread-safe, per-agent delta records of every session.

    All state access synchronizes on one lock; disk flushes carry per-session
    versions so a slow flush cannot overwrite a newer one. The ``save_*``
    flags only trim disk snapshots; the in-memory record keeps every field.
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
        self._versions: dict[str, int] = {}
        # session_id -> monotonic deletion time, in deletion order.
        self._deleted: OrderedDict[str, float] = OrderedDict()
        # First unresolved recording error per agent. An exact retry can
        # repair it only before that agent records another turn; otherwise
        # later deltas may already include the missing completion as prompt.
        self._recording_failed: dict[str, dict[str | None, _FailedTurn]] = {}

    def ensure_session(self, session_id: str, model_name: str | None = None) -> None:
        """Create the session unless it exists or is tombstoned; stamp the
        first validated ``model_name``."""
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
        sampling_params: dict[str, float | int] | None = None,
    ) -> None:
        """Record one completion, storing only its delta beyond the agent's
        previous turn (see :class:`CompletionRecord`).

        Duplicate deliveries of the agent's last recorded turn (lost-response
        retries) and turns for tombstoned sessions are dropped. Routing blobs
        are merged per agent by :func:`_merge_routed_experts`.
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

            agent_turns = [t for t in session.turns if t.agent_id == agent_id]
            failed = self._recording_failed.get(session_id, {}).get(agent_id)
            repairs_failure = (
                failed is not None
                and len(agent_turns) == failed.recorded_turns
                and full_prompt_ids == failed.prompt_token_ids
                and token_ids == failed.token_ids
                and normalize_messages(messages) == failed.messages
            )

            # Same rule as the TITO layer; a continuation without a recorded
            # predecessor is also stored in full.
            new_conversation = not any(msg.get("role") == "assistant" for msg in messages) or not agent_turns
            if new_conversation and agent_turns:
                # A re-served opening retry must not be recorded twice.
                last = agent_turns[-1]
                if (
                    last.new_conversation
                    and normalize_messages(messages) == normalize_messages(last.request_messages)
                    and token_ids == last.completion_token_ids
                    and full_prompt_ids == last.prompt_token_ids
                ):
                    logger.info(
                        "Session %s: skipping duplicate delivery of agent %s's "
                        "last recorded opening turn (%d completion tokens)",
                        session_id, agent_id, len(token_ids),
                    )
                    if repairs_failure:
                        self._clear_recording_failure(session_id, agent_id)
                    return
            elif not new_conversation:
                n_prev, stream_len = _conversation_extent(agent_turns)
                non_assistant = [m for m in messages if m.get("role") != "assistant"]
                last = agent_turns[-1]
                if (
                    len(non_assistant) == n_prev
                    and token_ids == last.completion_token_ids
                    and len(full_prompt_ids) == stream_len - len(last.completion_token_ids)
                ):
                    # Duplicate delivery; a real continuation always appends
                    # at least one non-assistant message.
                    logger.info(
                        "Session %s: skipping duplicate delivery of agent %s's "
                        "last recorded turn (%d completion tokens)",
                        session_id, agent_id, len(token_ids),
                    )
                    if repairs_failure:
                        self._clear_recording_failure(session_id, agent_id)
                    return
                if len(non_assistant) < n_prev or len(full_prompt_ids) < stream_len:
                    logger.warning(
                        "Session %s (agent %s): turn does not extend the recorded "
                        "stream (%d non-assistant messages / %d prompt tokens vs "
                        "%d / %d already recorded); storing the turn in full",
                        session_id, agent_id, len(non_assistant), len(full_prompt_ids), n_prev, stream_len,
                    )
                    new_conversation = True
                else:
                    messages = non_assistant[n_prev:]
                    full_prompt_ids = full_prompt_ids[stream_len:]
            if new_conversation and agent_id is not None:
                # A new conversation replaces the stream, so its stored capture is
                # stale even when capture is currently disabled.
                session.routed_experts.pop(agent_id, None)
            if routed_experts is not None and agent_id is not None:
                _merge_routed_experts(session, session_id, agent_id, routed_experts)
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
                sampling_params=sampling_params,
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
            )
            session.turns.append(record)
            if repairs_failure:
                self._clear_recording_failure(session_id, agent_id)
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
            if session_id in self._recording_failed:
                raise RuntimeError(
                    f"Session {session_id} has an unresolved recording failure: "
                    "a committed turn may be missing. Retry the failed turn before "
                    "advancing its agent, or retry the trial on a fresh session."
                )
            return session.model_dump()

    def finalize_session(self, session_id: str) -> None:
        """Mark a drained recording complete, checking failures atomically.

        The server owns admission/draining.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise RuntimeError("Session disappeared during finalization")
            if session_id in self._recording_failed:
                raise RuntimeError("Session has an unresolved recording failure; retry the trial on a fresh session")
            session.completed = True
            payload, version = self._snapshot(session)
        self._save(session_id, payload, version)

    def mark_recording_failed(
        self,
        session_id: str,
        *,
        agent_id: str | None = None,
        messages: list[dict[str, Any]],
        prompt_token_ids: list[int],
        token_ids: list[int],
    ) -> None:
        """Block collection until this committed turn is recorded or verified.

        Only an exact retry repairs it; a different turn's delta may already
        contain the missing tokens. The first failure per agent is kept.
        """
        with self._lock:
            if self._tombstoned(session_id):
                return
            failures = self._recording_failed.setdefault(session_id, {})
            if agent_id not in failures:
                session = self._sessions.get(session_id)
                failures[agent_id] = _FailedTurn(
                    messages=normalize_messages(messages),
                    prompt_token_ids=list(prompt_token_ids),
                    token_ids=list(token_ids),
                    recorded_turns=sum(t.agent_id == agent_id for t in session.turns) if session else 0,
                )

    def _clear_recording_failure(self, session_id: str, agent_id: str | None) -> None:
        """Resolve only this agent's verified retry; call under the lock."""
        failures = self._recording_failed[session_id]
        del failures[agent_id]
        if not failures:
            del self._recording_failed[session_id]

    def delete_session(self, session_id: str) -> None:
        """Remove a session from memory (its disk snapshot is kept) and
        tombstone its id."""
        with self._lock:
            self._sessions.pop(session_id, None)
            self._recording_failed.pop(session_id, None)
            self._deleted[session_id] = time.monotonic()
            self._deleted.move_to_end(session_id)
            self._prune_tombstones()
            # _versions is kept: the store's stale-write guard is monotonic per id.

    def is_deleted(self, session_id: str) -> bool:
        """Whether the session was deleted within the tombstone window."""
        with self._lock:
            return self._tombstoned(session_id)

    def _tombstoned(self, session_id: str) -> bool:
        """Live-tombstone check; call under the lock."""
        deleted_at = self._deleted.get(session_id)
        return deleted_at is not None and time.monotonic() - deleted_at < _TOMBSTONE_TTL

    def _prune_tombstones(self) -> None:
        """Drop expired tombstones and bound the table; call under the lock."""
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
        """Serialize the disk snapshot under the lock, stamped with a
        per-session monotonic version; disabled fields and null parsed views
        are omitted."""
        if self._store is None:
            return None, 0
        version = self._versions.get(session.session_id, 0) + 1
        self._versions[session.session_id] = version
        exclude: dict[str, Any] = {}
        if not self._save_routed_experts or not session.routed_experts:
            exclude["routed_experts"] = True
        turn_exclude: set[str] = set()
        if not self._save_logprobs:
            turn_exclude.add("completion_logprobs")
        if turn_exclude:
            exclude["turns"] = {"__all__": turn_exclude}
        data = session.model_dump(exclude=exclude)
        for turn_data in data["turns"]:
            for key in ("content", "reasoning_content", "tool_calls"):
                if turn_data.get(key) is None:
                    turn_data.pop(key, None)
        try:
            return json.dumps(data, ensure_ascii=False, separators=(",", ":")), version
        except (TypeError, ValueError) as e:
            # The disk snapshot is never read back; keep the in-memory record.
            logger.error(
                "Cannot serialize the disk snapshot of session %s (%s); "
                "skipping the write — the in-memory record stays complete",
                session.session_id, e,
            )
            return None, version

    def _save(self, session_id: str, payload: str | None, version: int) -> None:
        """Flush a serialized session to disk, outside the lock."""
        if payload is not None and self._store is not None:
            self._store.save(session_id, payload, version)
