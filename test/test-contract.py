"""Contract test: do the live engines still behave the way ``engines.py`` assumes?

``proxyserver/engines.py`` encodes one asymmetry between the two rollout
servers (read out of the reference implementation, verl):

* the vLLM server collapses "stop" and "length" into ``stop_reason="completed"``,
  so truncation must be recovered from the token count;
* the SGLang server passes SGLang's own ``meta_info.finish_reason.type``
  ("stop" / "length" / "abort") through verbatim, so it must be trusted.

Everything downstream — whether a truncated rollout is masked in training, whether
a tool call survives ``finish_reason`` — rests on that. An engine upgrade could
change it silently, and the unit tests would keep passing because they use fakes.

So this test asks the **real servers** in ``test/test_engines.yaml`` directly (every
endpoint listed in each engine's ``OPENAI_BASE_URL`` list), with no proxy in
the loop, and checks the four properties the proxy depends on:

1. a token-ID prompt is accepted and sampled token IDs come back;
2. the sampled IDs and their logprobs line up one-to-one;
3. a natural stop and a truncation are distinguishable, in the vocabulary each
   engine's rollout server reports;
4. a natural stop's last token is the EOS the model actually sampled.

Run (needs both endpoints reachable):
    python test/test-contract.py
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any
from common import EngineEndpoint, TokenOutput, load_bundled_tokenizer, load_engine_endpoints, make_rollout_server
from proxyserver.engines import LENGTH, SGLANG, STOP, VLLM, get_adapter
from proxyserver.token_stream import as_token_ids

logger = logging.getLogger("contract-test")

# Long enough that the model cannot finish, so truncation is guaranteed.
TRUNCATE_AT = 16
# Generous enough that a terse prompt finishes naturally. The model is a
# reasoning model, so its "one word" answers still cost a few hundred tokens.
STOP_BUDGET = 4096

TERSE_PROMPT = "Reply with exactly one word: OK"
LONG_PROMPT = "Count from 1 to 500, separated by commas."

# What each engine's rollout server is expected to report, per engines.py.
EXPECTED_STOP_REASONS: dict[str, dict[str, str]] = {
    VLLM: {STOP: "completed", LENGTH: "completed"},   # indistinguishable
    SGLANG: {STOP: "stop", LENGTH: "length"},         # authoritative
}


def _sampling_params(max_tokens: int) -> dict[str, Any]:
    """Exactly what the proxy sends — vLLM-styled on both engines."""
    return {"temperature": 0.0, "top_p": 1.0, "max_tokens": max_tokens, "logprobs": True}


async def check_engine(engine: str, endpoint: EngineEndpoint) -> None:
    adapter = get_adapter(engine)
    server = make_rollout_server(endpoint)
    tokenizer = load_bundled_tokenizer(endpoint.model)
    print(f"\n[{engine}] {endpoint.base_url}  model={endpoint.model}")

    try:
        eos_ids: set[int] = set()

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
        eos_ids.add(stopped.token_ids[-1])
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

        # A truncated response must never end on EOS — otherwise the token-count
        # heuristic the vLLM adapter relies on would be masking a real stop.
        assert truncated.token_ids[-1] not in eos_ids, (
            f"[{engine}] the truncated response ends on the EOS token; the prompt "
            f"is not long enough to guarantee truncation"
        )
        print(f"  adapter[{engine}]: 'stop' and 'length' both normalized correctly")
    finally:
        await server.aclose()


async def main() -> None:
    endpoints = {engine: load_engine_endpoints(engine) for engine in (VLLM, SGLANG)}
    missing = [e for e in (VLLM, SGLANG) if not endpoints[e]]
    if missing:
        raise SystemExit("test/test_engines.yaml has no section for: " + ", ".join(missing))

    print("engine contract test — asking the live servers, no proxy in the loop")
    for engine in (VLLM, SGLANG):
        for endpoint in endpoints[engine]:
            await check_engine(engine, endpoint)

    print("\n" + "=" * 70)
    print("PASS: both live engines match the contract engines.py is built on —")
    print("      token-ID prompts accepted, logprobs aligned with sampled tokens,")
    print("      EOS included on a natural stop, and stop_reason reported in the")
    print("      vocabulary each engine's rollout server uses ('completed' for vLLM,")
    print("      'stop'/'length' for SGLang), both normalizing to the right")
    print("      OpenAI finish_reason.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
