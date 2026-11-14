"""Optional policy checks at the rollout collection boundary."""

from __future__ import annotations
from collections.abc import Collection, Mapping
from typing import Any
from .model_registry import validate_sampling_overrides
from .protocol import finite
from .record_validation import validate_record_structure
from .rollout_sessions import CompletionRecord, SessionRecord


def validate_session_record(
    record: SessionRecord | Mapping[str, Any],
    *,
    expected_sampling: Mapping[str, float | int] | None = None,
    expected_weight_version: str | None = None,
    trained_agent_ids: Collection[str | None] | None = None,
    strict: bool = False,
    require_logprobs: bool = False,
    require_routed_experts: bool = False,
) -> None:
    """Reject sampled turns that do not match the collector's policy.

    Accept a session model or the JSON object from ``GET /sessions/{id}``.
    Every nonempty completion of the selected agents is checked, across all
    conversations. Pass failed and surplus trial records through this check
    before filtering them. Empty sessions/turns can represent failed trials
    and contain no sampled tokens to validate.

    strict checks raw record integrity without requiring optional fields.
    require_logprobs and require_routed_experts imply strict validation and
    require those fields for selected agents. Saved JSON may intentionally
    omit logprobs/routing blobs; live GET records retain them. Validate JSON
    before model construction to reject coercible invalid types.

    Checks are opt-in: ``expected_sampling`` requires recorded values for
    every supplied key (supply all seven distribution knobs for a complete
    policy check). ``expected_weight_version`` requires that exact version
    on every sampled turn. Leave it unset for direct engines that report no
    version. ``trained_agent_ids=None`` selects every agent; an explicit
    collection filters out agents whose samples will not be trained on.
    """
    sampling = None
    if expected_sampling is not None:
        if not isinstance(expected_sampling, Mapping) or not expected_sampling:
            raise ValueError("expected_sampling must be a nonempty mapping of distribution parameters")
        sampling = validate_sampling_overrides(dict(expected_sampling), "expected_sampling")
    if expected_weight_version is not None and (
        not isinstance(expected_weight_version, str) or not expected_weight_version
    ):
        raise ValueError("expected_weight_version must be a nonempty string or None")
    if trained_agent_ids is not None and (
        isinstance(trained_agent_ids, (str, bytes))
        or not isinstance(trained_agent_ids, Collection)
        or any(agent is not None and not isinstance(agent, str) for agent in trained_agent_ids)
    ):
        raise ValueError("trained_agent_ids must be a collection of agent IDs or None")
    selected = set(trained_agent_ids) if trained_agent_ids is not None else None
    if strict or require_logprobs or require_routed_experts:
        validate_record_structure(
            record.model_dump(exclude_unset=True) if isinstance(record, SessionRecord) else record,
            selected=selected, require_logprobs=require_logprobs,
            require_routed_experts=require_routed_experts,
        )
    if not isinstance(record, (SessionRecord, Mapping)):
        raise ValueError("Session record must be a SessionRecord or mapping")
    session_id = record.session_id if isinstance(record, SessionRecord) else record.get("session_id")
    turns = record.turns if isinstance(record, SessionRecord) else record.get("turns")
    if not isinstance(turns, list):
        raise ValueError(f"Session {session_id!r}: turns must be a list")
    for index, turn in enumerate(turns):
        if isinstance(turn, CompletionRecord):
            agent_id = turn.agent_id
            tokens = turn.completion_token_ids
            actual_sampling = turn.sampling_params
            actual_version = turn.weight_version
        elif isinstance(turn, Mapping):
            agent_id = turn.get("agent_id")
            tokens = turn.get("completion_token_ids")
            actual_sampling = turn.get("sampling_params")
            actual_version = turn.get("weight_version")
        else:
            raise ValueError(f"Session {session_id!r}: turn {index} must be an object")
        if agent_id is not None and not isinstance(agent_id, str):
            raise ValueError(f"Session {session_id!r}: turn {index} has an invalid agent_id")
        if selected is not None and agent_id not in selected:
            continue
        where = f"Session {session_id!r}, agent {agent_id!r}, turn {index}"
        if not isinstance(tokens, list):
            raise ValueError(f"{where}: completion_token_ids must be a list")
        if not tokens:
            continue
        if sampling is not None:
            if not isinstance(actual_sampling, Mapping):
                raise ValueError(f"{where}: missing sampling policy metadata")
            for key, expected in sampling.items():
                actual = actual_sampling.get(key)
                if not finite(actual) or actual != expected:
                    raise ValueError(
                        f"{where}: mismatched or missing sampling policy parameter "
                        f"{key!r}: expected {expected!r}, got {actual!r}"
                    )
        if expected_weight_version is not None and actual_version != expected_weight_version:
            raise ValueError(
                f"{where}: stale, mixed or missing weight version: "
                f"expected {expected_weight_version!r}, got {actual_version!r}"
            )
