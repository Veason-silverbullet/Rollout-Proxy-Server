"""Dependency-free integrity checks for raw proxy recording dictionaries."""
from collections.abc import Mapping
import base64
from .protocol import finite


def validate_record_structure(record, *, selected=None, require_logprobs=False, require_sampling_masks=False, require_routed_experts=False):
    """Validate raw dictionaries before any model coercion.

    Integrity checks apply to all turns. Required training fields apply only
    to selected agents. Empty diagnostic records remain valid. Expert blobs
    describe the last conversation per agent, never concatenated resets.
    """
    if not isinstance(record, Mapping) or not isinstance(record.get("turns"), list):
        raise ValueError("Session record must contain a turns list")
    lengths, conversations = {}, {}

    def ids(value, name):
        if not isinstance(value, list) or any(type(t) is not int or t < 0 for t in value):
            raise ValueError(f"{name} must contain nonnegative integer token IDs")

    for index, turn in enumerate(record["turns"]):
        if not isinstance(turn, Mapping):
            raise ValueError(f"turn {index} must be an object")
        agent = turn.get("agent_id")
        if agent is not None and not isinstance(agent, str):
            raise ValueError(f"turn {index} has an invalid agent_id")
        required = selected is None or agent in selected
        prompt, completion = turn.get("prompt_token_ids", []), turn.get("completion_token_ids")
        ids(prompt, "prompt_token_ids")
        ids(completion, "completion_token_ids")
        reset = turn.get("new_conversation", False)
        if type(reset) is not bool:
            raise ValueError("new_conversation must be a bool")
        if agent not in lengths or reset:
            lengths[agent] = 0
            conversations[agent] = conversations.get(agent, 0) + 1
        if required and require_logprobs and completion and not prompt and not lengths[agent]:
            raise ValueError("A training stream requires a leading prompt")
        lengths[agent] += len(prompt) + len(completion)
        probs = turn.get("completion_logprobs", [])
        if not isinstance(probs, list) or not all(finite(p) for p in probs):
            raise ValueError("completion_logprobs must contain finite numbers")
        if ("completion_logprobs" in turn or required and require_logprobs) and len(probs) != len(completion):
            raise ValueError("Missing or misaligned completion_logprobs")
        sampling = turn.get("sampling_params")
        if sampling is not None and (not isinstance(sampling, Mapping) or any(
            not isinstance(k, str) or not finite(v) for k, v in sampling.items()
        )):
            raise ValueError("sampling_params must be a finite numeric mapping")
        support, offsets = turn.get("rollout_top_p_token_ids"), turn.get("rollout_top_p_token_offsets")
        if support is None and offsets is None:
            if required and require_sampling_masks and completion:
                raise ValueError("Missing sampling masks")
            continue
        ids(support, "rollout_top_p_token_ids")
        if (not isinstance(offsets, list) or len(offsets) != len(completion) + 1
                or any(type(o) is not int or o < 0 for o in offsets)
                or offsets[0] != 0 or offsets[-1] != len(support)
                or any(a >= b for a, b in zip(offsets, offsets[1:]))):
            raise ValueError("Invalid sampling mask offsets")
        for i, token in enumerate(completion):
            row = support[offsets[i]:offsets[i + 1]]
            if len(set(row)) != len(row):
                raise ValueError("Sampling mask contains duplicate token IDs")
            if token not in row:
                raise ValueError("Sampling mask excludes the generated token")

    blobs = record.get("routed_experts", {})
    if not isinstance(blobs, Mapping):
        raise ValueError("routed_experts must be a mapping")
    for agent, blob in blobs.items():
        if agent not in lengths or not isinstance(blob, Mapping):
            raise ValueError("Invalid routed_experts agent or blob")
        rows, cols, start = blob.get("rows"), blob.get("cols"), blob.get("start", 0)
        if (type(rows) is not int or rows < 0 or type(cols) is not int or cols <= 0
                or type(start) is not int or start != 0 or blob.get("dtype", "uint8") != "uint8"
                or rows > max(0, lengths[agent] - 1)):
            raise ValueError("Invalid routed_experts dimensions or coverage")
        try:
            data = blob["data"]
            if not isinstance(data, str):
                raise ValueError("Expected base64 string")
            decoded = base64.b64decode(data, validate=True)
        except (KeyError, ValueError) as exc:
            raise ValueError("Invalid routed_experts base64 data") from exc
        if len(decoded) != rows * cols:
            raise ValueError("routed_experts byte count does not match dimensions")
    if require_routed_experts:
        for agent, length in lengths.items():
            if (selected is not None and agent not in selected) or not length:
                continue
            if conversations[agent] > 1:
                raise ValueError("Routing across conversation resets requires per-segment capture")
            if agent not in blobs or blobs[agent]["rows"] != max(0, length - 1):
                raise ValueError("Missing or incomplete routed_experts coverage")
