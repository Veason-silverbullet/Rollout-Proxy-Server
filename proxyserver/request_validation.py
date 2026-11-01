"""Validate the supported text-chat contract before admitting session work."""

from typing import Any
from .model_registry import SAMPLING_PARAM_KEYS, validate_sampling_overrides
from .token_stream import normalize_messages, validate_tools


def validate_reasoning_effort(value: Any, supported: tuple[str, ...] = ()) -> None:
    if value is not None and (
        not isinstance(value, str)
        or value not in ("none", "minimal", "low", "medium", "high", "xhigh", "max")
    ):
        raise ValueError("reasoning_effort must be a supported effort string or null")
    if value is not None and supported and value not in supported:
        raise ValueError(f"Unsupported reasoning_effort {value!r} for this model; expected one of {', '.join(supported)}")


def validate_generation_parameters(params: dict[str, Any]) -> None:
    validate_sampling_overrides(
        {key: params[key] for key in SAMPLING_PARAM_KEYS if params.get(key) is not None},
        "sampling parameters",
    )
    for key in ("max_tokens", "max_completion_tokens"):
        value = params.get(key)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{key} must be a positive integer")
    seed = params.get("seed")
    if seed is not None and type(seed) is not int:
        raise ValueError("seed must be an integer")
    stops = params.get("stop_token_ids")
    if stops is not None and (
        not isinstance(stops, list) or any(type(token) is not int or token < 0 for token in stops)
    ):
        raise ValueError("stop_token_ids must be a list of nonnegative integers")
    stop = params.get("stop")
    if stop is not None:
        stops = [stop] if isinstance(stop, str) else stop
        if not isinstance(stops, list) or any(not isinstance(s, str) or not s for s in stops):
            raise ValueError("stop must be a nonempty string or a list of nonempty strings")


def validate_chat_request(body: Any) -> None:
    if not isinstance(body, dict):
        raise ValueError("Body must be a JSON object")
    if not isinstance(body.get("model"), str) or not body["model"].strip():
        raise ValueError("request names no model; set a nonempty string in the OpenAI 'model' field")
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise ValueError("messages must be a nonempty list of message objects")
    normalize_messages(body["messages"])
    for message in body["messages"]:
        if message.get("audio") is not None or message.get("function_call") is not None:
            raise ValueError("Audio messages and legacy function_call messages are not supported")
        calls = message.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list):
                raise ValueError("message.tool_calls must be a list")
            for call in calls:
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise ValueError("message.tool_calls must contain function calls")
                function = call.get("function")
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    raise ValueError("Each tool call must name a function")
                if not isinstance(function.get("arguments"), (str, dict)):
                    raise ValueError("Tool call arguments must be a JSON string or object")
    validate_tools(body.get("tools"))
    validate_generation_parameters(body)
    validate_reasoning_effort(body.get("reasoning_effort"))
    if "stream" in body and type(body["stream"]) is not bool:
        raise ValueError("stream must be a boolean")

    # Recognized defaults are harmless. Constraints we cannot honor must fail
    # before generation or cached replay, just like constrained tool_choice.
    if body.get("n") is not None and (type(body["n"]) is not int or body["n"] != 1):
        raise ValueError("Only n=1 is supported")
    if body.get("response_format") is not None and body["response_format"] != {"type": "text"}:
        raise ValueError("Only response_format={'type': 'text'} is supported; structured output is not enforced")
    if body.get("tool_choice") not in (None, "auto"):
        raise ValueError("Unsupported tool_choice: only 'auto' or an omitted/null value is supported")
    if body.get("parallel_tool_calls") is not None and body["parallel_tool_calls"] is not True:
        raise ValueError("parallel_tool_calls=false cannot be enforced")
    if body.get("logprobs") is not None and body["logprobs"] is not False:
        raise ValueError("Response logprobs are not supported; sampled logprobs are available in session records")
    if body.get("top_logprobs") is not None and (
        type(body["top_logprobs"]) is not int or body["top_logprobs"] != 0
    ):
        raise ValueError("top_logprobs is not supported")
    if body.get("logit_bias") is not None and body["logit_bias"] != {}:
        raise ValueError("logit_bias is not supported")
    options = body.get("stream_options")
    if options is not None and options != {} and (
        not isinstance(options, dict) or set(options) != {"include_usage"}
        or options["include_usage"] is not True
    ):
        raise ValueError("Only stream_options={'include_usage': true} is supported")

    supported = {
        "model", "messages", "tools", "stream", "stream_options", "n", "response_format",
        "tool_choice", "parallel_tool_calls", "logprobs", "top_logprobs", "logit_bias",
        "max_tokens", "max_completion_tokens", "stop", "stop_token_ids", "seed", "reasoning_effort",
        # These observational fields do not change the generation contract.
        "user", "metadata", "safety_identifier", *SAMPLING_PARAM_KEYS,
    }
    unsupported = sorted(key for key, value in body.items() if key not in supported and value is not None)
    if unsupported:
        raise ValueError(f"Unsupported request fields: {', '.join(unsupported)}")
