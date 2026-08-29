"""Data models for the LLM proxy session recording."""

from __future__ import annotations
import time
from typing import Any
from pydantic import BaseModel, Field


class CompletionRecord(BaseModel):
    """Record of a single LLM completion call captured by the proxy.

    ``request_messages`` and ``prompt_token_ids`` store only this turn's
    **delta** — what the request appended beyond the previous recorded turn
    *of the same agent* — never the full history, which would grow the
    record quadratically in memory and on disk.  The assistant echo that
    opens a continuation's delta is not stored either: it is exactly the
    previous turn's recorded completion (``completion_text`` /
    ``tool_calls``), so a continuation's ``request_messages`` hold only the
    new non-assistant messages.  Strict token-in-token-out makes the delta
    lossless: every continuation prompt is a strict extension of its
    agent's token stream, so turn N's full values are rebuilt by
    concatenation **over the turns sharing this turn's ``agent_id``** (a
    session's agents each own an independent stream; their turns interleave
    in the record in arrival order)::

        full_messages_N = full_messages_{N-1} + [assistant_{N-1}] + request_messages_N
        full_prompt_N   = full_prompt_{N-1} + completion_token_ids_{N-1} + prompt_token_ids_N

    where ``assistant_{N-1}`` is the assistant message rebuilt from turn
    N-1's recorded completion (``content`` / ``reasoning_content`` /
    ``tool_calls`` — the extracted views of its ``completion_text``), and
    both streams reset at every turn whose ``new_conversation`` is true
    (such a turn stores the full request — including any pre-existing
    assistant history, which no recorded turn could restore — and
    *defines* a fresh stream).
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
            "— the session's first turn, or a request with no assistant "
            "message that started a fresh stream; request_messages and "
            "prompt_token_ids then hold the full request, not a delta"
        ),
    )
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
            "Chain-of-thought extracted from the completion's think block "
            "per the model profile's ReasoningParser, as served in the "
            "OpenAI response's message.reasoning_content; None when the "
            "model emitted none"
        ),
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description="Parsed tool calls from the completion, if any"
    )


class RoutedExpertsBlob(BaseModel):
    """Per-token MoE expert selections for one agent's whole token stream.

    Produced by the slime transport when routed-experts capture is on (R3,
    slime's ``--use-rollout-routing-replay``): the engine reports the
    selections for the **entire stream so far** on every turn, so one blob
    covers the agent's full stream minus one and each turn's blob supersedes
    the previous — the record never stores per-turn deltas (which would be
    quadratic, every delta carrying the whole prefix again).
    """

    data: str = Field(description="base64 of a C-order uint8 array of rows × cols expert ids")
    rows: int = Field(description="Tokens covered = the agent's stream length - 1")
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
    record, interleaved in arrival order and tagged with their
    :attr:`CompletionRecord.agent_id`.
    """

    session_id: str
    model_name: str | None = Field(
        default=None,
        description=(
            "The model the session is pinned to (the agent's validated 'model' claim); "
            "it selected the tokenizer every prompt of this session was built with"
        ),
    )
    turns: list[CompletionRecord] = Field(default_factory=list)
    routed_experts: dict[str, RoutedExpertsBlob] = Field(
        default_factory=dict,
        description=(
            "Latest routed-experts capture per agent_id — each blob covers "
            "that agent's whole recorded stream minus one and supersedes "
            "the previous turn's (see RoutedExpertsBlob).  Empty when "
            "capture is off; omitted from the persisted session JSON unless "
            "the proxy runs with save_rollout_routed_experts: true"
        ),
    )
    created_at: float = Field(default_factory=time.time)
    completed: bool = False
