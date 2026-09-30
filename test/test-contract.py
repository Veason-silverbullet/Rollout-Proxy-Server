"""Live wire-contract checks for vLLM and SGLang HTTP endpoints.

Independent clients verify token-ID prompts, aligned sampled logprobs,
native stop/length reasons, and EOS inclusion. No proxy is started.
Run: python test/test-contract.py
"""

from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import Any
from common import EngineEndpoint, TokenOutput, live_test_engines, load_bundled_tokenizer, load_engine_endpoints, make_engine_client
from proxyserver.engines import LENGTH, SGLANG, STOP, VLLM, get_adapter
from proxyserver.model_registry import resolve_profile
from proxyserver.token_stream import as_token_ids

# Long enough that the model cannot finish, so truncation is guaranteed.
TRUNCATE_AT = 16
# Generous enough that a terse prompt finishes naturally. The model is a
# reasoning model, so its "one word" answers still cost a few hundred tokens.
STOP_BUDGET = 4096

TERSE_PROMPT = "Reply with exactly one word: OK"
LONG_PROMPT = "Count from 1 to 500, separated by commas."

# Native finish reasons expected from each engine's HTTP API.
EXPECTED_STOP_REASONS: dict[str, dict[str, str]] = {
    VLLM: {STOP: "stop", LENGTH: "length"},
    SGLANG: {STOP: "stop", LENGTH: "length"},         # authoritative
}


def _sampling_params(max_tokens: int) -> dict[str, Any]:
    """Deterministic test parameters in the proxy's vLLM-style format."""
    return {"temperature": 0.0, "top_p": 1.0, "max_tokens": max_tokens, "logprobs": True}


def expected_eos_ids(model: str, tokenizer: Any) -> set[int]:
    """Read known stop tokens independently of the engine's sampled output."""
    eos = tokenizer.eos_token_id
    path = Path(resolve_profile(model).tokenizer_path) / "generation_config.json"
    if path.is_file():
        eos = json.loads(path.read_text()).get("eos_token_id", eos)
    ids = eos if isinstance(eos, list) else [eos]
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError(f"No valid EOS token IDs configured for {model!r}")
    return set(ids)


async def check_engine(engine: str, endpoint: EngineEndpoint) -> None:
    adapter = get_adapter(engine)
    server = make_engine_client(endpoint)
    tokenizer = load_bundled_tokenizer(endpoint.model)
    print(f"\n[{engine}] {endpoint.base_url}  model={endpoint.model}")

    try:
        eos_ids = expected_eos_ids(endpoint.model, tokenizer)

        async def generate(prompt: str, max_tokens: int) -> TokenOutput:
            # as_token_ids: transformers>=5 returns a BatchEncoding here, the
            # engine needs a flat token-ID list.
            prompt_ids = as_token_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
                ),
                what="contract-test prompt",
            )
            output = await server.generate(prompt_ids, _sampling_params(max_tokens))

            # (1) token-ID prompt in, sampled token IDs out.
            assert output.token_ids, f"[{engine}] no token ids returned"
            assert all(isinstance(t, int) for t in output.token_ids), f"[{engine}] token ids are not ints"
            # (2) logprobs align one-to-one with the sampled tokens; the trainer
            #     pairs them positionally, so a mismatch is silent corruption.
            assert len(output.log_probs) == len(output.token_ids), (
                f"[{engine}] {len(output.log_probs)} logprobs for "
                f"{len(output.token_ids)} tokens — cannot be paired"
            )
            return output

        # ---- natural stop -------------------------------------------------
        stopped = await generate(TERSE_PROMPT, STOP_BUDGET)
        assert len(stopped.token_ids) < STOP_BUDGET, (
            f"[{engine}] the 'terse' prompt exhausted {STOP_BUDGET} tokens; "
            f"raise STOP_BUDGET, this test cannot tell stop from length"
        )
        expected = EXPECTED_STOP_REASONS[engine][STOP]
        assert stopped.stop_reason == expected, (
            f"[{engine}] natural stop reported stop_reason={stopped.stop_reason!r}, "
            f"expected {expected!r} — engines.py is out of date with this server"
        )
        # (4) the sampled EOS is included in the token IDs, so the recorded
        #     completion is exactly what the policy produced.
        assert stopped.token_ids[-1] in eos_ids, (
            f"[{engine}] natural stop omitted EOS: last token {stopped.token_ids[-1]}, "
            f"expected one of {sorted(eos_ids)}"
        )
        eos_text = tokenizer.decode([stopped.token_ids[-1]])
        print(f"  stop:   {len(stopped.token_ids):>5} tokens, stop_reason={stopped.stop_reason!r}, "
              f"last token {stopped.token_ids[-1]} = {eos_text!r}")

        # ---- truncation ---------------------------------------------------
        truncated = await generate(LONG_PROMPT, TRUNCATE_AT)
        assert len(truncated.token_ids) == TRUNCATE_AT, (
            f"[{engine}] truncated at {len(truncated.token_ids)} tokens, expected {TRUNCATE_AT}"
        )
        expected = EXPECTED_STOP_REASONS[engine][LENGTH]
        assert truncated.stop_reason == expected, (
            f"[{engine}] truncation reported stop_reason={truncated.stop_reason!r}, "
            f"expected {expected!r} — engines.py is out of date with this server"
        )
        print(f"  length: {len(truncated.token_ids):>5} tokens, stop_reason={truncated.stop_reason!r}")

        # ---- (3) the adapter turns both into the right OpenAI finish_reason.
        assert adapter.finish_reason(stopped.stop_reason, num_tokens=len(stopped.token_ids),
                                     max_tokens=STOP_BUDGET) == STOP
        assert adapter.finish_reason(truncated.stop_reason, num_tokens=len(truncated.token_ids),
                                     max_tokens=TRUNCATE_AT) == LENGTH

        # The long prompt must actually truncate, rather than naturally emit EOS.
        assert truncated.token_ids[-1] not in eos_ids, (
            f"[{engine}] the truncated response ends on the EOS token; the prompt "
            f"is not long enough to guarantee truncation"
        )
        print(f"  adapter[{engine}]: 'stop' and 'length' both normalized correctly")
    finally:
        await server.aclose()


async def main() -> None:
    endpoints = {engine: load_engine_endpoints(engine) for engine in live_test_engines()}

    print("engine contract test — asking the live servers, no proxy in the loop")
    for engine, rows in endpoints.items():
        for endpoint in rows:
            await check_engine(engine, endpoint)

    print("\n" + "=" * 70)
    print("PASS: selected live engines match the contract engines.py is built on —")
    print("      token-ID prompts accepted, logprobs aligned with sampled tokens,")
    print("      EOS included on a natural stop, and stop_reason reported in the")
    print("      native HTTP vocabulary ('stop'/'length'), normalized correctly.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
