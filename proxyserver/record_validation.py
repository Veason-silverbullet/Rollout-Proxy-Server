"""Dependency-free integrity checks for raw proxy recording dictionaries."""
import base64
from collections.abc import Mapping
from .protocol import finite


def validate_record_structure(record, *, selected=None, require_logprobs=False, require_routed_experts=False):
    """Validate raw dictionaries before any model coercion.

    Integrity checks apply to all turns. Required training fields apply only
    to selected agents. Empty diagnostic records remain valid. Expert blobs
    describe independent segments.
    """
    if not isinstance(record, Mapping) or not isinstance(record.get("turns"), list):
        raise ValueError("Session record must contain a turns list")
    lengths, segment_lengths = {}, {}
    current_segments = {}

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
        expected = current_segments.get(agent, -1) + int(reset or agent not in current_segments)
        segment = turn.get("segment_index")
        if type(segment) is not int or segment < 0 or segment != expected:
            raise ValueError("Invalid segment_index or conversation boundary")
        current_segments[agent] = segment
        if turn.get("purpose", "primary") not in ("primary", "compaction", "auxiliary"):
            raise ValueError("Invalid recorded request purpose")
        if agent not in lengths or reset:
            lengths[agent] = 0
        if required and require_logprobs and completion and not prompt and not lengths[agent]:
            raise ValueError("A training stream requires a leading prompt")
        lengths[agent] += len(prompt) + len(completion)
        segment_lengths[agent, segment] = lengths[agent]
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

    captures = record.get("segment_routed_experts", {})
    if not isinstance(captures, Mapping):
        raise ValueError("segment_routed_experts must be a mapping")
    segment_blobs = {}
    for agent, segments in captures.items():
        if not isinstance(segments, Mapping):
            raise ValueError("Per-agent segment captures must be a mapping")
        for index, blob in segments.items():
            if isinstance(index, str) and index.isdecimal() and str(int(index)) == index:
                index = int(index)
            if type(index) is not int or (agent, index) not in segment_lengths:
                raise ValueError("Unknown routed-experts segment")
            segment_blobs[agent, index] = blob
    for key, blob in segment_blobs.items():
        length = segment_lengths[key]
        if not isinstance(blob, Mapping):
            raise ValueError("Invalid routed_experts agent or blob")
        rows, cols, start = blob.get("rows"), blob.get("cols"), blob.get("start", 0)
        if (type(rows) is not int or rows < 0 or type(cols) is not int or cols <= 0
                or type(start) is not int or start != 0 or blob.get("dtype", "uint8") != "uint8"
                or rows > max(0, length - 1)):
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
        for (agent, segment), length in segment_lengths.items():
            if (selected is not None and agent not in selected) or not length:
                continue
            blob = segment_blobs.get((agent, segment))
            if blob is None or blob["rows"] != max(0, length - 1):
                raise ValueError("Missing or incomplete routed_experts coverage")
