"""Exercise optional policy checks on collected JSON and session models."""

import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver.rollout_sessions import SessionRecord
from proxyserver.validation import validate_session_record
from sandbox.utils import get_session
from offline_common import check


def record():
    return {"session_id": "collection", "turns": [
        {"agent_id": agent, "request_messages": [], "completion_text": "answer",
         "completion_token_ids": [index], "weight_version": version,
         "sampling_params": {"temperature": 1.0, "top_p": top_p}}
        for index, (agent, version, top_p) in enumerate([
            ("trained", "7", 1.0), ("other", "3", 0.8), ("trained", "7", 1.0),
        ], 1)
    ]}


def rejects(data, message, **kwargs):
    try:
        validate_session_record(data, **kwargs)
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"Accepted a record that should fail: {message}")


def test_policy_checks():
    data = record()
    original = deepcopy(data)
    policy = {"expected_sampling": {"temperature": 1.0, "top_p": 1.0},
              "expected_weight_version": "7", "trained_agent_ids": {"trained"}}
    validate_session_record(data, **policy)
    validate_session_record(SessionRecord.model_validate(data), **policy)
    assert data == original
    rejects(data, "sampling policy", expected_sampling={"top_p": 1.0})
    # A later turn, including an opening turn of a fresh conversation, must
    # satisfy the same policy. Checking only the session's first turn is unsafe.
    for field, value, message in [
        ("sampling_params", None, "sampling policy"),
        ("sampling_params", {}, "sampling policy"),
        ("sampling_params", {"temperature": 1.0, "top_p": 0.9}, "sampling policy"),
        ("sampling_params", {"temperature": 1.0, "top_p": True}, "sampling policy"),
        ("sampling_params", {"temperature": 1.0, "top_p": float("nan")}, "sampling policy"),
        ("weight_version", None, "weight version"),
        ("weight_version", "8", "weight version"),
    ]:
        changed = record()
        changed["turns"][-1][field] = value
        changed["turns"][-1]["new_conversation"] = True
        rejects(changed, message, **policy)
    # Debug/legacy records and unversioned direct results stay readable.
    legacy = record()
    for turn in legacy["turns"]:
        turn.pop("sampling_params")
        turn["weight_version"] = None
    validate_session_record(legacy)
    legacy_model = SessionRecord.model_validate(legacy)
    assert all(turn.sampling_params is None for turn in legacy_model.turns)
    validate_session_record(legacy_model)
    direct = record()
    for turn in direct["turns"]:
        turn["weight_version"] = None
    validate_session_record(direct, expected_sampling={"top_p": 1.0}, trained_agent_ids={"trained"})
    validate_session_record({"session_id": "failed", "turns": []}, **policy)
    check("JSON/models, every selected turn, mixed policies, legacy and direct records", True)


def test_validation_arguments_and_shape():
    for bad in ({}, {"max_tokens": 8}, {"top_p": True}, {"top_p": float("inf")}, {"top_p": 10 ** 999}):
        rejects(record(), "expected_sampling", expected_sampling=bad)
    for bad in ("", 7):
        rejects(record(), "expected_weight_version", expected_weight_version=bad)
    for bad in ("trained", [7]):
        rejects(record(), "trained_agent_ids", trained_agent_ids=bad)
    for malformed, message in [
        ({}, "turns"), ({"turns": None}, "turns"),
        ({"turns": [None]}, "turn 0"),
        ({"turns": [{"agent_id": [], "completion_token_ids": [1]}]}, "agent_id"),
        ({"turns": [{"agent_id": "trained"}]}, "completion_token_ids"),
    ]:
        rejects(malformed, message, expected_sampling={"top_p": 1.0})
    check("malformed records and invalid policy expectations fail clearly", True)


def test_sandbox_collection_boundary():
    response = httpx.Response(200, json=record(), request=httpx.Request("GET", "http://proxy/sessions/collection"))
    with patch("proxyserver.client.httpx.get", return_value=response):
        result = get_session("http://proxy", "collection", expected_sampling={"top_p": 1.0},
                             expected_weight_version="7", trained_agent_ids={"trained"})
        assert result == record()
        try:
            get_session("http://proxy", "collection", expected_sampling={"top_p": 0.5})
        except ValueError as error:
            assert "sampling policy" in str(error)
        else:
            raise AssertionError("The collector returned mismatched samples")
        assert get_session("http://proxy", "collection") == record()
    missing = httpx.Response(404, request=httpx.Request("GET", "http://proxy/sessions/missing"))
    with patch("proxyserver.client.httpx.get", return_value=missing):
        try:
            get_session("http://proxy", "missing", expected_sampling={"top_p": 1.0})
        except httpx.HTTPStatusError as error:
            assert error.response.status_code == 404
        else:
            raise AssertionError("The collector accepted a missing session")
    check("sandbox collection rejects mismatches before returning samples", True)


def test_strict_collection_is_opt_in():
    data = record()
    data["turns"][0]["completion_token_ids"] = [-1]
    validate_session_record(data)
    rejects(data, "nonnegative", strict=True)
    response = httpx.Response(200, json=data, request=httpx.Request("GET", "http://proxy/sessions/collection"))
    with patch("proxyserver.client.httpx.get", return_value=response):
        assert get_session("http://proxy", "collection") == data
        try:
            get_session("http://proxy", "collection", strict=True)
        except ValueError as error:
            assert "nonnegative" in str(error)
        else:
            raise AssertionError("Strict collection returned invalid tokens")
    data = record()
    validate_session_record(data, strict=True)  # saved logprobs are optional
    validate_session_record(SessionRecord.model_validate(data), strict=True)
    supplied_empty = deepcopy(data)
    supplied_empty['turns'][0]['completion_logprobs'] = []
    rejects(supplied_empty, 'misaligned', strict=True)
    rejects(SessionRecord.model_validate(supplied_empty), 'misaligned', strict=True)
    rejects(data, "leading prompt", require_logprobs=True)
    for turn in data["turns"]:
        turn["prompt_token_ids"] = [0]
        turn["completion_logprobs"] = [-0.1]
    validate_session_record(data, require_logprobs=True)
    validate_session_record(SessionRecord.model_validate(data), require_logprobs=True)
    check("strict JSON/model checks and collector opt-in are wired", True)


if __name__ == "__main__":
    test_strict_collection_is_opt_in()
    test_policy_checks()
    test_validation_arguments_and_shape()
    test_sandbox_collection_boundary()
