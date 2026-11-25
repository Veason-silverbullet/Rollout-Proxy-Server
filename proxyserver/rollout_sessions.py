"""Data models for the LLM proxy session recording."""

from __future__ import annotations
import time
from typing import Any
from pydantic import BaseModel, Field


class CompletionRecord(BaseModel):
    """One completion and its prompt delta within an agent's segment.

    A segment opening stores the full prompt and request messages, including
    any retained assistant history. On continuations, reconstruct the prompt
    and canonical messages in turn order within (agent_id, segment_index)::

        prompt = previous_prompt + previous_completion_ids + prompt_token_ids
        messages = previous_messages + [previous_assistant] + request_messages

    Rebuild previous_assistant from the recorded content, reasoning_content,
    and tool_calls. The caller's assistant echo may differ from this message.
    """

    agent_id: str | None = Field(
        default=None,
        description=(
            "Agent this turn belongs to — the agent_id part of the keyed "
            "api_key.  Each agent owns an independent token stream, so "
            "delta reconstruction concatenates only turns with the same agent_id"
        ),
    )
    request_messages: list[dict[str, Any]] = Field(
        description=(
            "Non-assistant messages this request appended beyond the "
            "previous turn (the new tool/user/system messages — the "
            "assistant echo is not stored, being the previous turn's "
            "completion); the request's full messages array when "
            "new_conversation is true"
        )
    )
    prompt_token_ids: list[int] = Field(
        default_factory=list,
        description=(
            "Prompt token IDs this request appended to the session's token "
            "stream (token-in-token-out is enforced, so every prompt "
            "strictly extends the stream and the delta determines it); the "
            "full engine prompt when new_conversation is true"
        ),
    )
    new_conversation: bool = Field(
        default=False,
        description=(
            "True when this turn opened a (new) conversation on the session "
            "— an explicit segment opening; request_messages and prompt_token_ids "
            "then hold the full request, not a delta"
        ),
    )
    segment_index: int = Field(ge=0)
    purpose: str = "primary"
    completion_text: str = Field(description="Generated completion text")
    completion_token_ids: list[int] = Field(description="Token IDs of the generated completion")
    completion_logprobs: list[float] = Field(
        default_factory=list,
        description=(
            "Log probabilities for each generated token; always recorded "
            "in memory, but omitted from the persisted session JSON when "
            "the proxy runs with save_rollout_logprobs: false"
        ),
    )
    finish_reason: str | None = Field(
        default=None,
        description="Reason generation stopped: stop, tool_calls, length"
    )
    weight_version: str | None = Field(
        default=None,
        description=(
            "Policy weight version the engine reported for this turn. An RL "
            "trainer that updates weights between rollout steps needs it to "
            "prove a rollout was on-policy: a session whose turns carry more "
            "than one version straddled a weight update, and the samples it "
            "yields silently mix two policies. None when the backend does not "
            "report one (vLLM's OpenAI API, older rollout servers)"
        ),
    )
    sampling_params: dict[str, float | int] | None = Field(
        default=None,
        description=(
            "Distribution parameters sent to the engine after resolving model "
            "defaults, request values, and overrides. Retries retain the "
            "original parameters. None means no sampling metadata was recorded"
        ),
    )
    content: str | None = Field(
        default=None,
        description=(
            "Agent-visible content extracted from the completion — "
            "completion_text with reasoning and tool-call markup stripped, "
            "exactly as served in the OpenAI response's message.content "
            "(None on a tool-call turn with no surrounding text)"
        ),
    )
    reasoning_content: str | None = Field(
        default=None,
        description=(
            "Reasoning extracted from the completion's think block or Harmony "
            "analysis channels by the model-specific parser, as served in the "
            "OpenAI response's message.reasoning_content; None when the model "
            "emitted none"
        ),
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description="Parsed tool calls from the completion, if any"
    )


class RoutedExpertsBlob(BaseModel):
    """Per-token MoE expert selections for one agent's conversation segment.

    Produced by the sglang_generate and vllm_generate transports when routed-experts
    capture is enabled on both the proxy and the engine (R3). On the wire
    a turn's blob is either the **entire segment so far** (``start == 0``,
    the first turn or a fresh capture) or the rows added since the recorder's
    stored blob (``start == rows already stored``, requested with ``routed_experts_start_len``
    for SGLang or ``routed_experts_prompt_start`` for vLLM); the recorder appends
    deltas, so the stored blob covers the captured prefix with ``start == 0``
    — one blob per agent/segment, never per turn. Coverage can lag the segment
    when capture is disabled or an invalid extension is dropped.
    """

    data: str = Field(description="base64 of a C-order uint8 array of rows × cols expert ids")
    rows: int = Field(description="Tokens covered by this capture; may lag the current stream")
    start: int = Field(default=0, description="Row offset within the agent's stream; 0 once stored")
    cols: int = Field(
        description=(
            "Expert selections per token = num_layers × topk of the served "
            "model, as one number: the proxy holds no model config, so the "
            "trainer-side consumer validates the product and reshapes"
        )
    )
    dtype: str = Field(default="uint8", description="Element type of data; only uint8 is produced")


class SessionRecord(BaseModel):
    """All recorded data for a single session (one rollout).

    A rollout may run several agents (each with its own ``agent_id`` in the
    keyed api_key and its own token stream); their turns land in this one
    record, in recorder append order and tagged with their :attr:`CompletionRecord.agent_id`.
    This can differ from HTTP request arrival order.
    """

    session_id: str
    model_name: str | None = Field(
        default=None,
        description=(
            "The first validated model name recorded for this session; "
            "model pinning is per agent stream, so other agents may use another model"
        ),
    )
    turns: list[CompletionRecord] = Field(default_factory=list)
    segment_routed_experts: dict[str, dict[int, RoutedExpertsBlob]] = Field(
        default_factory=dict,
        description="Accumulated routing capture keyed by agent and segment; never shared across segments",
    )
    created_at: float = Field(default_factory=time.time)
    completed: bool = False
