"""End-to-end test of the LLM proxy server in **local (direct) mode**, strict TITO.

Topology (one process, everything over real HTTP):

    50 mock agents               LLMProxyServer (local mode)
    (OpenAI SDK)   --HTTP-->     server.make_local_completion_handler
                                      |
                                      v
                                 DirectRolloutProvider (TokenStreamManager, strict TITO)
                                      |  HTTP: vLLM POST /v1/completions (token-ID prompt,
                                      |        return_tokens_as_token_ids), or
                                      |        SGLang POST /generate (input_ids, return_logprob)
                                      v
                                 the engine endpoints from test/test_engines.yaml
                                 (prompts tokenized locally from proxyserver/tokenization
                                 via mapping.json)

This is the counterpart of ``test-slime.py`` for ``transport_mode: direct``:
the proxy runs with the injected local completion handler and the real
:class:`DirectRolloutProvider`, which POSTs token-ID prompts straight to the
engines' own HTTP APIs — no training framework, no rollout-server stand-in.
Every endpoint listed for the engine under test is used, so the provider's
round-robin first-turn placement and sticky per-session bindings are
exercised whenever more than one instance is running.

Unlike slime (SGLang-only), direct supports both engines — the engine under
test comes from ``$INFERENCE_ENGINE`` (default ``vllm``), like the
relay/worker tests.  Under vLLM this also covers the raw ``finish_reason``
pass-through (``"length"`` trusted verbatim, ``"stop"`` double-checked
against the token count).

Per-session verification criteria are shared with the other e2e tests — see
``common.run_agent``.  Tampered history must be rejected with a 400 (the
same status the other modes return), never silently re-tokenized.

Run:
    python test/test-direct.py                          # vLLM
    INFERENCE_ENGINE=sglang python test/test-direct.py  # SGLang
"""

from __future__ import annotations
import asyncio
import logging
import httpx
from common import ENDPOINTS, ENGINE, KEY_DELIMITER, NUM_AGENTS, PROXY_API_KEY, MODEL_NAME, make_verifier, run_gauntlet
from proxyserver.rollout_provider import DirectRolloutProvider
from proxyserver.server import LLMProxyServer, make_local_completion_handler, stream_id

logger = logging.getLogger("direct-test")

MAX_TOKENS = 8192


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


async def main() -> None:
    model = MODEL_NAME
    verifier = make_verifier(model)

    # Wire the real local-mode stack, exactly as cli.py builds it
    # for `transport_mode: direct`.  Every fixture endpoint of the engine
    # under test is passed, so sticky round-robin placement is exercised too.
    provider = DirectRolloutProvider(
        base_urls=[ep.base_url for ep in ENDPOINTS],
        inference_engine=ENGINE,
        api_keys=[ep.api_key for ep in ENDPOINTS],
    )
    proxy = LLMProxyServer(
        host="127.0.0.1",
        port=0,
        completion_handler=make_local_completion_handler(provider),
        on_session_deleted=provider.release_session,
        api_key=PROXY_API_KEY,
        key_delimiter=KEY_DELIMITER,
        mode_label="direct",
    )
    assert proxy.mode == "local"
    proxy_url = await proxy.start()
    logger.info(
        "Proxy started at %s (mode=%s, %d %s endpoint(s): %s)",
        proxy_url, proxy.mode, len(provider.base_urls), ENGINE, ", ".join(provider.base_urls),
    )

    async with httpx.AsyncClient(timeout=30) as http:
        try:
            health = (await http.get(f"{proxy_url}/health")).json()
            assert health["mode"] == "local", f"unexpected health: {health}"

            # The shared gauntlet (common.run_gauntlet), plus the two
            # direct-specific seams injected below.
            agent_id_by_session: dict[str, str] = {}

            def sessions_bound_sticky(records: list) -> None:
                # Every session must have been bound sticky to one endpoint.
                agent_id_by_session.update({r["session_id"]: r["turns"][0]["agent_id"] for r in records})
                stream_ids = {stream_id(sid, aid) for sid, aid in agent_id_by_session.items()}
                assert set(provider._session_endpoints) >= stream_ids, "sessions missing endpoint bindings"

            def provider_dropped_session(sid: str) -> None:
                stream = stream_id(sid, agent_id_by_session[sid])
                assert not provider.models.has_session(stream), f"provider did not drop the token stream of deleted session {stream}"
                assert stream not in provider._session_endpoints, f"provider did not drop the endpoint binding of deleted session {stream}"

            await run_gauntlet(
                proxy_url, http,
                verifier=verifier, max_tokens=MAX_TOKENS,
                dump_dir=f"direct-{ENGINE}", proxy_api_key=PROXY_API_KEY,
                after_rollouts=sessions_bound_sticky,
                drop_check=provider_dropped_session,
            )

        finally:
            await proxy.stop()
            await provider.aclose()

    print("\n" + "=" * 70)
    print(f"PASS: local (direct) mode strict TITO OK — {NUM_AGENTS} agents x 2 turns via")
    print("      proxy -> local handler -> DirectRolloutProvider -> engine HTTP")
    print(f"      -> {ENGINE} ({model}): token-ID prompts in, sampled token")
    print("      IDs/logprobs out, sessions sticky to their endpoint, every")
    print("      turn's prompt strictly extends the session token stream,")
    print("      tampered history rejected with 400, keyed routing rejects")
    print("      malformed/unauthorized keys with 401, truncation is reported")
    print("      as finish_reason='length', and stream=true is honored as an")
    print("      SSE replay.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
