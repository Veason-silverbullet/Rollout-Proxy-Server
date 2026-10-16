"""Shared gpt-oss-20b/120b Harmony profile.

The bundled HF tokenizer spells Harmony IDs 200002/200012 as <|return|> and
<|call|>. Both end an assistant action; <|end|> only ends a message and must
not stop an analysis-to-final generation. TITO retains all sampled reasoning
and endings, unlike upstream full-history inference rendering.
"""

from .base import ModelProfile
from .harmony import HarmonyReasoning
from .tool_parser import HarmonyToolParser
from ..token_stream import TokenStreamError


def prepare_delta_messages(messages: list[dict], assistant: dict) -> list[dict]:
    """Name tool-result speakers from the preceding assistant's call IDs.

    OpenAI tool results usually carry only tool_call_id. Harmony requires a
    function name in the speaker header. The manager supplies cached parsed
    calls when available, so a modified echo cannot redirect that speaker.
    Only new messages are copied/annotated; sampled history is never rendered.
    """
    calls = assistant.get("tool_calls") or []
    by_id = {call["id"]: call for call in calls if call.get("id")}
    result = []
    for message in messages:
        if message.get("role") not in ("user", "tool"):
            raise TokenStreamError("gpt-oss system/developer instructions belong in the opening request")
        if message["role"] == "tool":
            call_id = message.get("tool_call_id")
            call = by_id.get(call_id) if call_id else (calls[0] if len(calls) == 1 else None)
            name = (call.get("function") or call).get("name") if call else None
            if not isinstance(name, str) or not name:
                raise TokenStreamError("gpt-oss tool result must identify a preceding function call")
            if message.get("name") and message["name"] != name:
                raise TokenStreamError("gpt-oss tool result name disagrees with its tool_call_id")
            message = {**message, "name": name}
        result.append(message)
    return result


class GPTOSS(ModelProfile):
    name = "gpt_oss"
    tokenizer_dir = "GPT-OSS"
    generation_prompt = "<|start|>assistant"
    assistant_turn_end = "<|end|>"
    assistant_turn_endings = ("<|return|>", "<|call|>", "<|endoftext|>")
    prepare_delta_messages = staticmethod(prepare_delta_messages)
    stop_token_ids = (200002, 199999, 200012)
    tool_call_parser = HarmonyToolParser
    reasoning = HarmonyReasoning()


PROFILE = GPTOSS()
