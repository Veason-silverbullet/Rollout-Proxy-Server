"""Training conversion and result contracts, without framework dependencies."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proxyserver.training import record_to_streams, TrialResult, validate_training_results


ROLES = {"planner": "planner", "worker": "agent"}
POLICY = {"temperature": 1.0, "top_p": 1.0}


def turn(agent, prompt, tokens, *, reset=False):
    return {
        "agent_id": agent, "prompt_token_ids": prompt,
        "completion_token_ids": tokens, "completion_logprobs": [-.25] * len(tokens),
        "new_conversation": reset, "sampling_params": dict(POLICY),
        "weight_version": "7", "finish_reason": "stop",
    }


def record(sid="trial_readable"):
    return {"session_id": sid, "turns": [
        turn("planner", [1, 2], [3], reset=True),
        turn("worker", [10], [11, 12], reset=True),
        turn("planner", [4], [5, 6]),
    ]}


def rejects(fn, *args, match=None, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        if match is not None:
            assert match in str(exc), str(exc)
    else:
        raise AssertionError("Accepted invalid training data")


def test_conversion():
    data = record()
    original = deepcopy(data)
    planner, worker = record_to_streams(data, roles=ROLES)
    assert planner.tokens == [1, 2, 3, 4, 5, 6]
    assert planner.loss_mask == [0, 0, 1, 0, 1, 1]
    assert planner.logprobs == [0., 0., -.25, 0., -.25, -.25]
    assert [(c.start, c.end) for c in planner.completions] == [(2, 3), (4, 6)]
    assert planner.completion_ends == [3, 6]
    assert [c.weight_version for c in planner.completions] == ["7", "7"]
    assert worker.tokens == [10, 11, 12] and worker.role == "agent"
    assert data == original
    planner.tokens[0] = 99
    planner.completions[0].sampling_params["top_p"] = .5
    assert data == original
    json.dumps(asdict(worker), allow_nan=False)
    assert record_to_streams({"session_id": "empty", "turns": []}, roles={}) == []
    rejects(record_to_streams, data, roles={"planner": "planner"}, match="Unregistered")
    for field, value in [("completion_token_ids", [True]), ("completion_logprobs", []),
                         ("completion_logprobs", [float("inf")]), ("weight_version", 7),
                         ("sampling_params", {"top_p": True})]:
        bad = record()
        bad["turns"][0][field] = value
        rejects(record_to_streams, bad, roles=ROLES)
    bad = record()
    bad["turns"][0]["prompt_token_ids"] = []
    rejects(record_to_streams, bad, roles=ROLES, match="leading prompt")
    bad = record()
    bad["turns"][0].pop("completion_logprobs")
    rejects(record_to_streams, bad, roles=ROLES, match="logprobs")
    legacy = {"session_id": "legacy", "turns": [turn(None, [1], [2])]}
    legacy["turns"][0].pop("weight_version")
    legacy["turns"][0].pop("sampling_params")
    stream, = record_to_streams(legacy, roles={None: "agent"})
    assert stream.agent_id is None and stream.completions[0].weight_version is None
    assert stream.completions[0].sampling_params is None


def test_segments_and_capture():
    data = record()
    data["turns"].append(turn("planner", [20], [21], reset=True))
    rejects(record_to_streams, data, roles=ROLES, match="reset")
    streams = record_to_streams(data, roles=ROLES, allow_segments=True)
    assert [(s.agent_id, s.segment_index) for s in streams] == [
        ("planner", 0), ("worker", 0), ("planner", 1)]
    assert streams[0].tokens == [1, 2, 3, 4, 5, 6]
    assert streams[2].tokens == [20, 21]
    # A real agent's name cannot collide with a synthetic segment identity.
    data["turns"].append(turn("planner.segment1", [30], [31], reset=True))
    assert len(record_to_streams(data, roles={**ROLES, "planner.segment1": "agent"}, allow_segments=True)) == 4
    data["routed_experts"] = {"planner": {"rows": 1, "cols": 1, "data": "AQ=="}}
    rejects(record_to_streams, data, roles=ROLES, allow_segments=True, match="reset")

    data = {"session_id": "capture", "turns": [turn("worker", [10], [11, 12])],
            "routed_experts": {"worker": {"rows": 2, "cols": 1, "data": "AQI="}}}
    stream, = record_to_streams(data, roles=ROLES, require_routed_experts=True)
    assert stream.routed_experts == data["routed_experts"]["worker"]
    stream.routed_experts["rows"] = 1
    assert data["routed_experts"]["worker"]["rows"] == 2
    rejects(stream.validate, match="byte count")
    data["routed_experts"]["worker"].update(rows=1, data="AQ==")
    rejects(record_to_streams, data, roles=ROLES, match="incomplete")
    data.pop("routed_experts")
    rejects(record_to_streams, data, roles=ROLES, require_routed_experts=True)


def test_result_contract():
    allocations = {"task": [["trial_first", "trial_readable"], ["trial_surplus"]]}
    first = TrialResult("task", 0, 1, "trial_readable", record_to_streams(record(), roles=ROLES), scores={"reward": 1.0})
    surplus = TrialResult("task", 1, 0, "trial_surplus", record_to_streams(record("trial_surplus"), roles=ROLES), error="failed")
    options = dict(allocations=allocations, roles=ROLES, trained_roles=["agent"],
                   expected_sampling=POLICY, expected_weight_version="7")
    results = [first, surplus]
    validate_training_results(results, **options)
    rejects(validate_training_results, [first], **options, match="Missing allocated")
    rejects(validate_training_results, [first, first], **options, match="duplicate trial")
    for field, value in [("slot", True), ("slot", -1), ("slot", 2), ("attempt", True),
                         ("attempt", -1), ("attempt", 2), ("attempt", 0),
                         ("session_id", "wrong"), ("task_id", "wrong"),
                         ("scores", {"reward": float("nan")})]:
        bad = deepcopy(results)
        setattr(bad[0], field, value)
        rejects(validate_training_results, bad, **options)
    bad = deepcopy(results)
    bad[1].streams[1].completions[0].weight_version = "stale"
    rejects(validate_training_results, bad, **options, match="weight version")
    bad[1].error = None  # Successful surplus trials must be checked too.
    rejects(validate_training_results, bad, **options, match="weight version")
    for params in (None, {"temperature": 1.0}, {"temperature": 1.0, "top_p": .9}):
        bad = deepcopy(results)
        bad[1].streams[1].completions[0].sampling_params = params
        rejects(validate_training_results, bad, **options, match="sampling policy")
    bad = deepcopy(results)
    bad[0].streams[0].completions[0].weight_version = None  # Untrained planner.
    validate_training_results(bad, **options)
    bad[0].streams[1].role = "planner"  # Cannot relabel trained agent to escape checks.
    rejects(validate_training_results, bad, **options, match="mismatched role")
    bad = deepcopy(results)
    bad[0].streams.append(deepcopy(bad[0].streams[0]))
    rejects(validate_training_results, bad, **options, match="duplicate stream")
    bad = deepcopy(results)
    bad[0].streams[0].session_id = "wrong"
    rejects(validate_training_results, bad, **options, match="Wrong-session")
    bad = deepcopy(results)
    bad[0].streams[0].segment_index = 1
    rejects(validate_training_results, bad, **options, match="Missing conversation")
    # Even untrained streams must have structurally valid data.
    bad = deepcopy(results)
    bad[0].streams[0].loss_mask[0] = 1
    rejects(validate_training_results, bad, **options, match="boundaries")
    empty = TrialResult("empty", 0, 0, "no_tokens", error="engine failed")
    validate_training_results([empty], allocations={"empty": [["no_tokens"]]}, roles={})
    validate_training_results([], allocations={}, roles={})
    for allocation in ({"task": [["same"], ["same"]]}, {"task": []},
                       {"task": [[]]}, {"task": "sid"}, {"task": [[""]]}):
        rejects(validate_training_results, [], allocations=allocation, roles=ROLES)
    rejects(validate_training_results, results, **{**options, "trained_roles": ["unknown"]})
    rejects(validate_training_results, results, **{**options, "expected_sampling": {"top_p": True}})


def test_capture_requirements_and_segment_policy():
    data = record()
    data["routed_experts"] = {"worker": {"rows": 2, "cols": 1, "data": "AQI="}}
    streams = record_to_streams(data, roles=ROLES)
    result = TrialResult("task", 0, 0, data["session_id"], streams)
    options = dict(allocations={"task": [[data["session_id"]]]}, roles=ROLES,
                   require_routed_experts=True)
    validate_training_results([result], trained_roles=["agent"], **options)
    rejects(validate_training_results, [result], **options)  # Planner has no capture.
    # Segment policy checks visit each conversation, not only the final one.
    data.pop("routed_experts")
    data["turns"].append(turn("worker", [20], [21], reset=True))
    result.streams = record_to_streams(data, roles=ROLES, allow_segments=True)
    options = dict(allocations={"task": [[data["session_id"]]]}, roles=ROLES,
                   expected_weight_version="7", expected_sampling=POLICY)
    validate_training_results([result], **options)
    result.streams[1].completions[0].weight_version = None
    rejects(validate_training_results, [result], **options, match="weight version")


def test_stream_tampering_and_empty_turns():
    stream = record_to_streams(record(), roles=ROLES)[0]
    for mutate in (
        lambda s: setattr(s.completions[0], "start", True),
        lambda s: setattr(s.completions[1], "start", 2),
        lambda s: setattr(s.completions[1], "end", 5),
        lambda s: s.logprobs.__setitem__(0, -.1),
        lambda s: s.tokens.__setitem__(0, True),
        lambda s: setattr(s, "agent_id", []),
        lambda s: setattr(s, "segment_index", True),
    ):
        bad = deepcopy(stream)
        mutate(bad)
        rejects(bad.validate)
    data = {"session_id": "empty_turn", "turns": [turn("worker", [10], []),
                                                    turn("worker", [], [11])]}
    data["turns"][0].update(weight_version=None, sampling_params=None)
    stream, = record_to_streams(data, roles=ROLES)
    assert stream.tokens == [10, 11] and stream.completion_ends == [2]
    validate_training_results([TrialResult("task", 0, 0, "empty_turn", [stream])],
                              allocations={"task": [["empty_turn"]]}, roles=ROLES,
                              expected_sampling=POLICY, expected_weight_version="7")


def test_recorder_integration():
    from proxyserver.recorder import SessionRecorder
    recorder = SessionRecorder()
    recorder.ensure_session("integration")
    messages = [{"role": "user", "content": "hello"}]
    for prompt, tokens, history in (
        ([1, 2], [3], messages),
        ([1, 2, 3, 4], [5], messages + [{"role": "assistant", "content": "answer"},
                                       {"role": "user", "content": "continue"}]),
    ):
        recorder.record_completion("integration", agent_id="worker", messages=history,
            completion_text="answer", token_ids=tokens, logprobs=[-.5], prompt_token_ids=prompt,
            sampling_params=POLICY, weight_version="7")
    recorder.finalize_session("integration")
    stream, = record_to_streams(recorder.dump_session("integration"), roles=ROLES)
    assert stream.tokens == [1, 2, 3, 4, 5]
    assert stream.loss_mask == [0, 0, 1, 0, 1]
    assert stream.completion_ends == [3, 5]


if __name__ == "__main__":
    subprocess.run([sys.executable, "-c", "import sys; import proxyserver.training; "
        "assert not {'transformers', 'numpy', 'pydantic', 'fastapi', "
        "'proxyserver.service'} & sys.modules.keys()"], check=True)
    for test in (test_conversion, test_segments_and_capture, test_result_contract,
                 test_capture_requirements_and_segment_policy,
                 test_stream_tampering_and_empty_turns, test_recorder_integration):
        test()
        print(f"PASS {test.__name__}")
