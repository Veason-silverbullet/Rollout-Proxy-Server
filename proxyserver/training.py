"""Optional, framework-independent conversion and rollout contract checks.

Only the standard library and raw record integrity checks are imported. These
utilities do not participate in request serving. Finalize before collecting;
neither a snapshot nor its ``completed`` flag proves finalization occurred.
"""

from __future__ import annotations
from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from .protocol import finite
from .record_validation import validate_record_structure


def _name(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _roles(roles: Mapping[str | None, str]) -> None:
    if not isinstance(roles, Mapping):
        raise ValueError("roles must map agent IDs to explicit roles")
    for agent, role in roles.items():
        if agent is not None:
            _name(agent, "agent_id")
        _name(role, "role")


@dataclass
class TrainingCompletion:
    """One recorded turn's [start, end) completion span in a training stream.

    Empty completions retain their turn metadata with start == end.
    """

    start: int
    end: int
    weight_version: str | None = None
    sampling_params: dict[str, float | int] | None = None
    finish_reason: str | None = None


@dataclass
class TrainingStream:
    """Exact tokens for one agent conversation, with unshifted training masks.

    Prompt/gap tokens have loss_mask=0 and placeholder logprobs=0.0; sampled tokens
    have loss_mask=1 and their recorded logprobs. The trainer owns causal shifting.
    Identity is (session_id, agent_id, segment_index), never a synthesized agent name.
    """

    session_id: str
    agent_id: str | None
    role: str
    segment_index: int = 0
    tokens: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    completions: list[TrainingCompletion] = field(default_factory=list)
    routed_experts: dict[str, Any] | None = None

    @property
    def completion_ends(self) -> list[int]:
        """Exclusive ends of nonempty completions, in stream order."""
        return [c.end for c in self.completions if c.end > c.start]

    def validate(self, *, require_routed_experts: bool = False) -> None:
        _name(self.session_id, "session_id")
        if self.agent_id is not None:
            _name(self.agent_id, "agent_id")
        _name(self.role, "role")
        if type(self.segment_index) is not int or self.segment_index < 0:
            raise ValueError("segment_index must be a nonnegative integer")
        if any(not isinstance(v, list) for v in (
            self.tokens, self.loss_mask, self.logprobs, self.completions,
        )) or not (len(self.tokens) == len(self.loss_mask) == len(self.logprobs)):
            raise ValueError("Tokens, loss masks and logprobs must be aligned lists")
        if any(type(m) is not int or m not in (0, 1) for m in self.loss_mask):
            raise ValueError("Loss masks must contain integer zeros and ones")
        if not all(finite(p) for p in self.logprobs):
            raise ValueError("Logprobs must be finite numbers")
        cursor, turns = 0, []
        for completion in self.completions:
            if (not isinstance(completion, TrainingCompletion)
                    or type(completion.start) is not int or type(completion.end) is not int
                    or not cursor <= completion.start <= completion.end <= len(self.tokens)):
                raise ValueError("Invalid or overlapping completion boundaries")
            start, end = completion.start, completion.end
            if (any(self.loss_mask[cursor:start])
                    or any(m != 1 for m in self.loss_mask[start:end])
                    or any(p != 0 for p in self.logprobs[cursor:start])):
                raise ValueError("Masks/logprobs disagree with prompt and completion boundaries")
            if completion.weight_version is not None:
                _name(completion.weight_version, "weight_version")
            if completion.finish_reason is not None:
                _name(completion.finish_reason, "finish_reason")
            turns.append({
                "agent_id": self.agent_id,
                "prompt_token_ids": self.tokens[cursor:start],
                "completion_token_ids": self.tokens[start:end],
                "completion_logprobs": self.logprobs[start:end],
                "sampling_params": completion.sampling_params,
            })
            cursor = end
        if cursor != len(self.tokens):
            raise ValueError("Completion boundaries do not cover the recorded stream")
        blobs = {} if self.routed_experts is None else {self.agent_id: self.routed_experts}
        validate_record_structure(
            {"turns": turns, "routed_experts": blobs}, require_logprobs=True,
            # Never expose an attached partial capture as ready for training.
            require_routed_experts=require_routed_experts or self.routed_experts is not None,
        )


def record_to_streams(
    record: Mapping[str, Any], *, roles: Mapping[str | None, str],
    allow_segments: bool = False,
    require_routed_experts: bool = False,
) -> list[TrainingStream]:
    """Convert raw session JSON, validating it before any type coercion.

    Pass SessionRecord.model_dump() for model instances. All recorded agents must
    have a registered role; no naming heuristics or filtering are applied. Missing
    generated-token logprobs are rejected. Missing sampling/version attestations
    remain None, for optional policy checks at the result boundary.

    Resets are rejected unless allow_segments=True. Routing across resets is always
    rejected because records retain only the final conversation's capture. An empty
    diagnostic record returns []; callers decide whether it is trainable. Returned
    lists and metadata are independent copies of the record.
    """
    _roles(roles)
    validate_record_structure(record, require_logprobs=True, require_routed_experts=require_routed_experts)
    session_id = record.get("session_id")
    _name(session_id, "session_id")
    streams: list[TrainingStream] = []
    current: dict[str | None, TrainingStream] = {}
    blobs = record.get("routed_experts", {})
    for turn in record["turns"]:
        agent = turn.get("agent_id")
        if agent not in roles:
            raise ValueError(f"Unregistered recorded agent: {agent!r}")
        stream = current.get(agent)
        reset = stream is not None and turn.get("new_conversation", False)
        if reset and (not allow_segments or agent in blobs):
            raise ValueError("Conversation reset requires separate segments without routing capture")
        if stream is None or reset:
            stream = TrainingStream(session_id, agent, roles[agent],
                                    0 if stream is None else stream.segment_index + 1)
            current[agent] = stream
            streams.append(stream)
        prompt, completion = turn.get("prompt_token_ids", []), turn["completion_token_ids"]
        start = len(stream.tokens) + len(prompt)
        stream.tokens.extend(prompt)
        stream.tokens.extend(completion)
        stream.loss_mask.extend([0] * len(prompt) + [1] * len(completion))
        stream.logprobs.extend([0.0] * len(prompt) + list(turn.get("completion_logprobs", [])))
        stream.completions.append(TrainingCompletion(
            start, len(stream.tokens), weight_version=turn.get("weight_version"),
            sampling_params=deepcopy(turn.get("sampling_params")),
            finish_reason=turn.get("finish_reason"),
        ))
    for stream in streams:
        stream.routed_experts = deepcopy(blobs.get(stream.agent_id))
        stream.validate(require_routed_experts=require_routed_experts)
    return streams


@dataclass
class TrialResult:
    """One terminal result per allocated task slot, including failed trials.

    attempt is the zero-based index into that slot's allocated session IDs. Earlier
    attempt diagnostics can be validated in separate batches with their own explicit
    allocations. Error fields never exempt available samples from validation.
    """

    task_id: str
    slot: int
    attempt: int
    session_id: str
    streams: list[TrainingStream] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    cleanup_error: str | None = None


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def validate_training_results(
    results: Sequence[TrialResult], *, allocations: Mapping[str, Sequence[Sequence[str]]],
    roles: Mapping[str | None, str], trained_roles: Collection[str] | None = None,
    expected_sampling: Mapping[str, float | int] | None = None,
    expected_weight_version: str | None = None,
    require_routed_experts: bool = False,
) -> None:
    """Validate all trial results before filtering failures or surplus slots.

    allocations maps task -> slots -> session IDs ordered by retry attempt. IDs are
    opaque nonempty strings, not UUIDs. Exactly one terminal result is required per
    slot; allocated session IDs must be globally unique. roles is authoritative:
    results cannot relabel an agent to bypass policy validation.

    Structural checks apply to every stream. Policy and required capture checks
    apply to trained_roles (None selects all registered roles). Expectations are
    opt-in, including weight versions for unversioned direct engines. Supply all
    distribution parameters for a complete policy check. Empty diagnostic results
    are permitted; task success and minimum trainable data are caller decisions.
    """
    _roles(roles)
    if trained_roles is None:
        selected = set(roles.values())
    else:
        if (not isinstance(trained_roles, Collection) or isinstance(trained_roles, (str, bytes))
                or any(not isinstance(r, str) or r not in roles.values() for r in trained_roles)):
            raise ValueError("trained_roles must contain registered roles")
        selected = set(trained_roles)
    if expected_sampling is not None and (
        not isinstance(expected_sampling, Mapping) or not expected_sampling
        or any(not isinstance(k, str) or not k or not finite(v)
               for k, v in expected_sampling.items())
    ):
        raise ValueError("expected_sampling must be a nonempty finite numeric mapping")
    if expected_weight_version is not None:
        _name(expected_weight_version, "expected_weight_version")
    if not isinstance(allocations, Mapping) or not _sequence(results):
        raise ValueError("Expected an allocation mapping and a sequence of TrialResult objects")
    expected, sessions = {}, set()
    for task, slots in allocations.items():
        _name(task, "task_id")
        if not _sequence(slots) or not slots:
            raise ValueError("Each task requires allocated slots")
        for slot, attempts in enumerate(slots):
            if not _sequence(attempts) or not attempts:
                raise ValueError("Each slot requires allocated attempts")
            for sid in attempts:
                _name(sid, "allocated session_id")
                if sid in sessions:
                    raise ValueError("Duplicate allocated session ID")
                sessions.add(sid)
            expected[task, slot] = attempts
    seen = set()
    for result in results:
        if not isinstance(result, TrialResult):
            raise ValueError("Results must be TrialResult objects")
        _name(result.task_id, "task_id")
        _name(result.session_id, "session_id")
        if type(result.slot) is not int or type(result.attempt) is not int or result.attempt < 0:
            raise ValueError("Slot and attempt must be nonnegative integers")
        key = (result.task_id, result.slot)
        if key not in expected or key in seen:
            raise ValueError("Unallocated or duplicate trial slot")
        attempts = expected[key]
        if result.attempt >= len(attempts) or attempts[result.attempt] != result.session_id:
            raise ValueError("Trial returned an unallocated attempt/session")
        seen.add(key)
        if not isinstance(result.scores, Mapping) or any(
            not isinstance(k, str) or not finite(v) for k, v in result.scores.items()
        ):
            raise ValueError("Scores must be a finite numeric mapping")
        if any(v is not None and not isinstance(v, str) for v in (result.error, result.cleanup_error)):
            raise ValueError("Trial errors must be strings or None")
        if not isinstance(result.streams, list):
            raise ValueError("Trial streams must be a list")
        identities = set()
        segments: dict[str | None, list[int]] = {}
        for stream in result.streams:
            if not isinstance(stream, TrainingStream):
                raise ValueError("Trial streams must be TrainingStream objects")
            trained = isinstance(stream.role, str) and stream.role in selected
            stream.validate(require_routed_experts=require_routed_experts and trained)
            identity = (stream.agent_id, stream.segment_index)
            if stream.session_id != result.session_id or identity in identities:
                raise ValueError("Wrong-session or duplicate stream identity")
            identities.add(identity)
            segments.setdefault(stream.agent_id, []).append(stream.segment_index)
            if stream.agent_id not in roles or stream.role != roles[stream.agent_id]:
                raise ValueError("Unregistered agent or mismatched role")
            if stream.role not in selected:
                continue
            for completion in stream.completions:
                if completion.start == completion.end:
                    continue
                if expected_weight_version is not None and completion.weight_version != expected_weight_version:
                    raise ValueError("Stale, mixed or missing weight version")
                if expected_sampling is not None and (
                    completion.sampling_params is None
                    or any(completion.sampling_params.get(k) != v for k, v in expected_sampling.items())
                ):
                    raise ValueError("Mismatched or missing sampling policy")
        for indices in segments.values():
            if sorted(indices) != list(range(len(indices))):
                raise ValueError("Missing conversation segment")
    if seen != set(expected):
        raise ValueError("Missing allocated trial results")
