"""Offline test of :class:`SessionRecorder` delta storage and duplicate dedup.

The dedup this pins: a lost-response retry is re-served the already-committed
turn (``rollout_provider`` via ``token_stream.cached_turn``), and a late
relay delivery of the original response can race it
(``server._record_orphan_response``).  Both funnel into
``record_completion``, which must record the turn exactly once — recording
the duplicate would append a bogus full-stored copy of the turn, and the
trainer's delta reconstruction would double the stream.

Run:
    python test/test-recorder.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver.recorder import _TOMBSTONE_TTL, SessionRecorder  # noqa: E402
from offline_common import check  # noqa: E402

U1 = {"role": "user", "content": "hi"}
A1 = {"role": "assistant", "content": "hello"}
U2 = {"role": "user", "content": "more"}
A2 = {"role": "assistant", "content": "t2"}
U3 = {"role": "user", "content": "next"}

P1 = [1, 2, 3]          # full engine prompt of turn 1
C1 = [900, 901]         # completion of turn 1
D2 = [4, 5]             # prompt delta of turn 2
C2 = [910, 911, 912]    # completion of turn 2
P2 = P1 + C1 + D2       # full engine prompt of turn 2
D3 = [6, 7]             # prompt delta of turn 3
P3 = P2 + C2 + D3       # full engine prompt of turn 3


def record_turn1(rec: SessionRecorder) -> None:
    rec.ensure_session("s", model_name="m")
    rec.record_completion("s", [U1], "hello", C1, [-0.1, -0.2],
                          finish_reason="stop", prompt_token_ids=P1)


def record_turn2(rec: SessionRecorder) -> None:
    rec.record_completion("s", [U1, A1, U2], "t2", C2, [-0.3, -0.4, -0.5],
                          finish_reason="stop", prompt_token_ids=P2)


def test_delta_storage() -> None:
    print("Turns are stored delta-only")
    rec = SessionRecorder()
    record_turn1(rec)
    record_turn2(rec)
    turns = rec.dump_session("s")["turns"]
    check("two turns recorded", len(turns) == 2)
    check("turn 1 stores the full render",
          turns[0]["new_conversation"] and turns[0]["prompt_token_ids"] == P1)
    check("turn 2 stores only its delta, without the assistant echo",
          not turns[1]["new_conversation"]
          and turns[1]["prompt_token_ids"] == D2
          and [m["role"] for m in turns[1]["request_messages"]] == ["user"])


def test_duplicate_delivery_dedup() -> None:
    print("\nA duplicate delivery of the last turn is recorded exactly once")
    rec = SessionRecorder()
    record_turn1(rec)
    record_turn2(rec)

    # The re-served retry (or the late orphan delivery racing it) hands the
    # recorder an identical payload again — it must not append a second copy.
    record_turn2(rec)
    turns = rec.dump_session("s")["turns"]
    check("duplicate of the last turn is dropped", len(turns) == 2)

    # A genuinely new turn still extends the record afterwards.
    rec.record_completion("s", [U1, A1, U2, A2, U3], "t3", [920], [-0.6],
                          finish_reason="stop", prompt_token_ids=P3)
    turns = rec.dump_session("s")["turns"]
    check("a following turn still lands as a delta",
          len(turns) == 3
          and not turns[2]["new_conversation"]
          and turns[2]["prompt_token_ids"] == D3
          and [m["role"] for m in turns[2]["request_messages"]] == ["user"])

    # Same shape but a different completion (a real re-sample, e.g. after a
    # new_conversation reset) is NOT treated as a duplicate.
    rec2 = SessionRecorder()
    record_turn1(rec2)
    record_turn2(rec2)
    rec2.record_completion("s", [U1, A1, U2], "t2'", [990, 991, 992],
                           [-0.7, -0.8, -0.9], finish_reason="stop",
                           prompt_token_ids=P2)
    check("a different completion for the same request is kept (stored full)",
          len(rec2.dump_session("s")["turns"]) == 3)


def test_duplicate_opening_turn_dedup() -> None:
    print("\nA duplicate delivery of an *opening* turn is recorded once too")
    # The opening request carries no assistant message, so it is stored in
    # full rather than as a delta — and the continuation dedup never runs for
    # it.  Its retry is the most expensive turn to lose, and both the
    # re-served retry and a late orphan delivery funnel here: without a check
    # of its own the first turn lands twice, a phantom branch the agent never
    # saw, trained on as if it had.
    rec = SessionRecorder()
    record_turn1(rec)
    record_turn1(rec)
    turns = rec.dump_session("s")["turns"]
    check("duplicate of the opening turn is dropped", len(turns) == 1)

    # A genuinely new conversation on the same session still lands.
    rec.record_completion("s", [{"role": "user", "content": "phase 2"}], "p2",
                          [930], [-0.9], finish_reason="stop", prompt_token_ids=[8, 9])
    turns = rec.dump_session("s")["turns"]
    check("a different opening request still opens a new conversation",
          len(turns) == 2 and turns[1]["new_conversation"])

    # ...and so does a re-sample of the same opening prompt that produced
    # different tokens (nothing about it is a duplicate delivery).
    rec2 = SessionRecorder()
    record_turn1(rec2)
    rec2.record_completion("s", [U1], "hello", [990, 991], [-0.1, -0.2],
                           finish_reason="stop", prompt_token_ids=P1)
    check("a different completion for the same opening request is kept",
          len(rec2.dump_session("s")["turns"]) == 2)

    # The continuation after a deduped opening turn still lands as a delta —
    # dropping the duplicate must not disturb the delta baseline.
    rec3 = SessionRecorder()
    record_turn1(rec3)
    record_turn1(rec3)
    record_turn2(rec3)
    turns = rec3.dump_session("s")["turns"]
    check("the following turn still lands as a delta on the deduped opening turn",
          len(turns) == 2 and not turns[1]["new_conversation"]
          and turns[1]["prompt_token_ids"] == D2)


def test_multi_agent_interleaving() -> None:
    print("\nAgents' turns interleave in one record; deltas and dedup are per agent")
    B1 = {"role": "user", "content": "b task"}
    B2 = [B1, {"role": "assistant", "content": "b1"},
          {"role": "user", "content": "b more"}]
    rec = SessionRecorder()
    rec.ensure_session("s", model_name="m")
    # Agent a opens, agent b opens in between, agent a continues: b's
    # opener must not reset a's delta baseline.
    rec.record_completion("s", [U1], "hello", C1, [-0.1, -0.2],
                          finish_reason="stop", prompt_token_ids=P1, agent_id="a")
    rec.record_completion("s", [B1], "b1", [800], [-0.3],
                          finish_reason="stop", prompt_token_ids=[50, 51], agent_id="b")
    rec.record_completion("s", [U1, A1, U2], "t2", C2, [-0.3, -0.4, -0.5],
                          finish_reason="stop", prompt_token_ids=P2, agent_id="a")
    turns = rec.dump_session("s")["turns"]
    check("turns land in arrival order, tagged per agent",
          [t["agent_id"] for t in turns] == ["a", "b", "a"])
    check("b's interleaved opener stores its full render",
          turns[1]["new_conversation"] and turns[1]["prompt_token_ids"] == [50, 51])
    check("a's continuation is a delta over a's own turns",
          not turns[2]["new_conversation"] and turns[2]["prompt_token_ids"] == D2)

    # A duplicate delivery of a's last turn lands AFTER b recorded another
    # turn: the dedup must still spot it among a's turns, not compare it
    # against the session's overall last turn.
    rec.record_completion("s", B2, "b2", [810, 811], [-0.5, -0.6],
                          finish_reason="stop", prompt_token_ids=[50, 51, 800, 60],
                          agent_id="b")
    rec.record_completion("s", [U1, A1, U2], "t2", C2, [-0.3, -0.4, -0.5],
                          finish_reason="stop", prompt_token_ids=P2, agent_id="a")
    turns = rec.dump_session("s")["turns"]
    check("duplicate of a's last turn is dropped despite b's turn in between",
          [t["agent_id"] for t in turns] == ["a", "b", "a", "b"])
    check("b's continuation stayed a delta of b's own stream",
          not turns[3]["new_conversation"] and turns[3]["prompt_token_ids"] == [60])

    # The default agent (an old-style key, agent_id None) is independent of
    # the named agents.
    rec.record_completion("s", [U1], "hello", [990], [-0.1],
                          finish_reason="stop", prompt_token_ids=P1)
    turns = rec.dump_session("s")["turns"]
    check("an agent-less turn opens its own stream",
          len(turns) == 5 and turns[4]["agent_id"] is None
          and turns[4]["new_conversation"])


def test_extracted_completion_fields() -> None:
    print("\nDumps carry the extracted content/reasoning_content, and no assistant echo in deltas")
    # The caller (server._message_fields) extracts the agent-visible
    # content and reasoning_content from completion_text; the recorder
    # stores them verbatim alongside the raw completion_text.
    rec = SessionRecorder()
    rec.ensure_session("q", model_name="Qwen3.5-9B")
    rec.record_completion("q", [U1], "<think>weighing it</think>hello", C1, [-0.1, -0.2],
                          finish_reason="stop", prompt_token_ids=P1,
                          content="hello", reasoning_content="weighing it")
    rec.record_completion("q", [U1, A1, U2], "<think>\n\n</think>t2", C2, [-0.3, -0.4, -0.5],
                          finish_reason="stop", prompt_token_ids=P2, content="t2")
    turns = rec.dump_session("q")["turns"]
    check("a turn carries the extracted content and reasoning_content",
          turns[0]["content"] == "hello" and turns[0]["reasoning_content"] == "weighing it")
    check("reasoning_content is None when the model emitted none",
          turns[1]["content"] == "t2" and turns[1]["reasoning_content"] is None)
    check("no stored message carries reasoning_content",
          all("reasoning_content" not in m for t in turns for m in t["request_messages"]))
    check("the continuation's assistant echo is not stored",
          [m["role"] for m in turns[1]["request_messages"]] == ["user"])

    # A new_conversation request with pre-existing assistant history keeps
    # it: no recorded turn could restore those messages.
    rec2 = SessionRecorder()
    rec2.ensure_session("h", model_name="m")
    rec2.record_completion("h", [U1, A1, U2], "t2", C2, [-0.3, -0.4, -0.5],
                           finish_reason="stop", prompt_token_ids=P2)
    turns = rec2.dump_session("h")["turns"]
    check("an opening turn keeps its pre-existing assistant history",
          turns[0]["new_conversation"]
          and [m["role"] for m in turns[0]["request_messages"]] == ["user", "assistant", "user"])


def test_weight_version_recorded_per_turn() -> None:
    print("\nEach turn records the engine's policy weight version")
    # An RL trainer that updates weights between rollout steps needs this to
    # prove a session was on-policy: turns carrying two versions straddled an
    # update, and the samples they yield silently mix two policies.
    rec = SessionRecorder()
    rec.ensure_session("w", model_name="m")
    rec.record_completion("w", [U1], "hello", C1, [-0.1, -0.2],
                          finish_reason="stop", prompt_token_ids=P1,
                          weight_version="step-7")
    rec.record_completion("w", [U1, A1, U2], "t2", C2, [-0.3, -0.4, -0.5],
                          finish_reason="stop", prompt_token_ids=P2,
                          weight_version="step-8")
    rec.record_completion("w", [U1, A1, U2, A2, U3], "t3", [920], [-0.6],
                          finish_reason="stop", prompt_token_ids=P3)
    turns = rec.dump_session("w")["turns"]
    check("the version rides on each turn",
          [t["weight_version"] for t in turns] == ["step-7", "step-8", None])
    check("a session that straddled an update is detectable from the record",
          len({t["weight_version"] for t in turns if t["weight_version"]}) == 2)
    check("an engine that reports none records None, not a mismatch",
          turns[2]["weight_version"] is None)


def test_deleted_session_tombstone() -> None:
    print("\nA deleted session's late turns are dropped, not resurrected")
    # Deletion races the session's own in-flight turns: the driver deletes a
    # cancelled trial mid-generation, and the turn-preserving delivery paths
    # (relay orphan recording, local-mode detached pipelines) hand the turn
    # to the recorder afterwards.  Recording it would re-create the record —
    # and the driver deletes exactly once, so nothing would ever free it.
    rec = SessionRecorder()
    record_turn1(rec)
    rec.delete_session("s")
    check("deleted session is tombstoned", rec.is_deleted("s"))
    check("the record is gone", rec.dump_session("s") is None)

    record_turn2(rec)
    check("a turn delivered after deletion does not re-create the record",
          rec.dump_session("s") is None)
    rec.ensure_session("s", model_name="m")
    check("ensure_session does not re-create a tombstoned session",
          rec.dump_session("s") is None)

    # A trial cancelled during its very first turn has stragglers too: the
    # delete may land before any turn was recorded, and must still tombstone.
    rec.delete_session("never-served")
    check("deleting a never-served session still tombstones its id",
          rec.is_deleted("never-served"))
    check("an unrelated live session is not tombstoned", not rec.is_deleted("other"))

    # Past the TTL the id is free again — a deliberately re-created session
    # id (the case the _versions comment preserves) starts fresh.
    rec._deleted["s"] -= 2 * _TOMBSTONE_TTL
    check("the tombstone expires", not rec.is_deleted("s"))
    record_turn1(rec)
    check("an expired id records again",
          rec.dump_session("s") is not None and len(rec.dump_session("s")["turns"]) == 1)


def blob(rows: int) -> dict:
    """A routed-experts blob as the slime transport packs it (R3)."""
    return {"data": "AAECAwQF", "rows": rows, "cols": 6, "dtype": "uint8"}


def test_routed_experts_latest_supersedes() -> None:
    print("\nRouted-experts blobs are kept per agent, latest coverage wins")
    # Each turn's capture covers the agent's whole stream so far, so the
    # record stores one blob per agent (superseding), never per turn — and
    # never shrinks: a late out-of-order delivery (an orphaned turn racing a
    # newer one) must not replace wider coverage with narrower.
    rec = SessionRecorder()
    rec.ensure_session("s", model_name="m")
    rec.record_completion("s", [U1], "hello", C1, [-0.1, -0.2],
                          finish_reason="stop", prompt_token_ids=P1, agent_id="a",
                          routed_experts=blob(len(P1) + len(C1) - 1))
    rec.record_completion("s", [U1, A1, U2], "t2", C2, [-0.3, -0.4, -0.5],
                          finish_reason="stop", prompt_token_ids=P2, agent_id="a",
                          routed_experts=blob(len(P2) + len(C2) - 1))
    dump = rec.dump_session("s")
    check("one blob per agent, not per turn", list(dump["routed_experts"]) == ["a"])
    check("the newer, wider capture superseded turn 1's",
          dump["routed_experts"]["a"]["rows"] == len(P2) + len(C2) - 1)

    rec.record_completion("s", [U1], "hello", C1, [-0.1, -0.2],
                          finish_reason="stop", prompt_token_ids=P1, agent_id="b",
                          routed_experts=blob(len(P1) + len(C1) - 1))
    dump = rec.dump_session("s")
    check("agents keep independent blobs",
          dump["routed_experts"]["b"]["rows"] == len(P1) + len(C1) - 1
          and dump["routed_experts"]["a"]["rows"] == len(P2) + len(C2) - 1)

    # A late delivery of turn 1's payload: shorter coverage, dropped.
    rec.record_completion("s", [U1], "hello", C1, [-0.1, -0.2],
                          finish_reason="stop", prompt_token_ids=P1, agent_id="a",
                          routed_experts=blob(len(P1) + len(C1) - 1))
    check("an out-of-order delivery cannot shrink the stored coverage",
          rec.dump_session("s")["routed_experts"]["a"]["rows"] == len(P2) + len(C2) - 1)

    # A turn recorded without a blob (capture toggled off mid-run) keeps the
    # stored one — the record never loses data it already holds.
    rec.record_completion("s", [U1, A1, U2, A2, U3], "t3", [920], [-0.6],
                          finish_reason="stop", prompt_token_ids=P3, agent_id="a")
    check("a blob-less turn leaves the stored blob in place",
          rec.dump_session("s")["routed_experts"]["a"]["rows"] == len(P2) + len(C2) - 1)


def test_routed_experts_snapshot_gating() -> None:
    print("\nBlobs stay out of the disk snapshots unless asked for")
    # The store rewrites the full session JSON after every turn; a blob
    # covering a long stream re-written per turn is write amplification for
    # data the trainer fetches over GET /sessions/{id} anyway.
    import json
    import tempfile
    from proxyserver.recorder import SessionStore

    with tempfile.TemporaryDirectory() as root:
        store = SessionStore(Path(root), mode="test")
        rec = SessionRecorder(store=store, save_routed_experts=False)
        rec.ensure_session("s", model_name="m")
        rec.record_completion("s", [U1], "hello", C1, [-0.1, -0.2],
                              finish_reason="stop", prompt_token_ids=P1, agent_id="a",
                              routed_experts=blob(len(P1) + len(C1) - 1))
        on_disk = json.loads((store.run_dir / "s.json").read_text())
        check("the persisted JSON omits the blobs by default", "routed_experts" not in on_disk)
        check("the in-memory record (GET /sessions/{id}) always carries them",
              rec.dump_session("s")["routed_experts"]["a"]["rows"] == len(P1) + len(C1) - 1)

    with tempfile.TemporaryDirectory() as root:
        store = SessionStore(Path(root), mode="test")
        rec = SessionRecorder(store=store, save_routed_experts=True)
        rec.ensure_session("s", model_name="m")
        rec.record_completion("s", [U1], "hello", C1, [-0.1, -0.2],
                              finish_reason="stop", prompt_token_ids=P1, agent_id="a",
                              routed_experts=blob(len(P1) + len(C1) - 1))
        on_disk = json.loads((store.run_dir / "s.json").read_text())
        check("save_routed_experts=True persists them",
              on_disk["routed_experts"]["a"]["rows"] == len(P1) + len(C1) - 1)

        # Capture off: the empty dict carries no information — the disk
        # record omits the key so an R3-less run's files keep their shape.
        rec.ensure_session("bare", model_name="m")
        rec.record_completion("bare", [U1], "hello", C1, [-0.1, -0.2],
                              finish_reason="stop", prompt_token_ids=P1, agent_id="a")
        on_disk = json.loads((store.run_dir / "bare.json").read_text())
        check("a capture-off session's file omits the empty key", "routed_experts" not in on_disk)


def main() -> None:
    test_delta_storage()
    test_duplicate_delivery_dedup()
    test_duplicate_opening_turn_dedup()
    test_multi_agent_interleaving()
    test_extracted_completion_fields()
    test_weight_version_recorded_per_turn()
    test_deleted_session_tombstone()
    test_routed_experts_latest_supersedes()
    test_routed_experts_snapshot_gating()
    print("\n" + "=" * 70)
    print("PASS: SessionRecorder stores turns delta-only and records a turn\n"
          "      exactly once — opening turns included — even when a re-served\n"
          "      retry and a late orphan delivery both hand it the same payload;\n"
          "      a multi-agent session's interleaved turns keep per-agent\n"
          "      deltas and per-agent dedup; dumps carry the extracted\n"
          "      content/reasoning_content but never the assistant echo a\n"
          "      delta replays, and each turn carries the engine's weight\n"
          "      version so an off-policy session is detectable; routed-experts\n"
          "      blobs are per-agent, latest-coverage-wins, and stay out of\n"
          "      the per-turn disk rewrites unless save_routed_experts asks.")
    print("=" * 70)


if __name__ == "__main__":
    main()
