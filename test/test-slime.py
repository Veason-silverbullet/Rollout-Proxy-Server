"""End-to-end test of the LLM proxy server in **Slime transport**, strict TITO.

Topology (one process, everything over real HTTP):

    50 mock agents               LLMProxyServer
    (OpenAI SDK)   --HTTP-->     server.make_completion_handler
                                      |
                                      v
                                 SlimeRolloutProvider (TokenStreamManager, strict TITO)
                                      |  HTTP: POST /generate (input_ids, return_logprob)
                                      v
                                 SGLang endpoint (stands in for slime's sgl-router;
                                 prompts tokenized locally from proxyserver/tokenization 
                                 via mapping.json)

This test exercises the Slime transport:
the proxy runs with the injected completion handler and the real
:class:`SlimeRolloutProvider`.  slime fronts its pool of SGLang servers with
``sgl-router``, a *transparent* HTTP proxy — the router's ``/generate`` is
SGLang's ``/generate`` — so pointing the provider straight at the SGLang
endpoint from ``test/test_engines.yaml`` exercises exactly the wire protocol
production would speak to the router; the router only adds load balancing
across replicas.

slime is SGLang-only, so this test forces the ``sglang`` fixtures regardless
of ``$INFERENCE_ENGINE``.

Per-session verification criteria are shared with the direct tests — see
``common.run_agent``.  Tampered history must be rejected with a 400 (the
same status the other transports return), never silently re-tokenized.

Run:
    python test/test-slime.py
"""

from __future__ import annotations
import asyncio
import logging
import os
from tempfile import TemporaryDirectory

# slime is SGLang-only: pick the sglang fixtures/adapter before `common`
# resolves the engine under test from the environment.
os.environ["INFERENCE_ENGINE"] = "sglang"

import httpx  # noqa: E402
from common import ENDPOINT, ENGINE, KEY_DELIMITER, NUM_AGENTS,  PROXY_API_KEY, MODEL_NAME, make_verifier, run_gauntlet
from proxyserver.rollout_provider import SlimeRolloutProvider  # noqa: E402
from proxyserver.server import LLMProxyServer, make_completion_handler, stream_id  # noqa: E402

logger = logging.getLogger("slime-test")

MAX_TOKENS = 8192


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


async def main() -> None:
    model = MODEL_NAME
    verifier = make_verifier(model)

    # Wire the real provider stack, exactly as cli.py builds it
    # for `transport_mode: slime`.  ENDPOINT.root_url plays slime's sgl-router.
    provider = SlimeRolloutProvider(
        router_url=ENDPOINT.root_url,
        api_key=ENDPOINT.api_key,
    )
    error_logs = TemporaryDirectory(prefix="proxy-slime-errors-")
    proxy = LLMProxyServer(
        host="127.0.0.1",
        port=0,
        completion_handler=make_completion_handler(provider),
        on_session_deleted=provider.release_session,
        api_key=PROXY_API_KEY,
        key_delimiter=KEY_DELIMITER,
        error_log_dir=error_logs.name,
    )
    proxy_url = await proxy.start()
    logger.info("Proxy started at %s (router=%s)", proxy_url, provider.router_url)

    async with httpx.AsyncClient(timeout=30) as http:
        try:
            health = (await http.get(f"{proxy_url}/health")).json()
            assert health == {"status": "ok"}, f"unexpected health: {health}"

            # Track the actual agent streams before testing deletion cleanup.
            streams_by_session: dict[str, set[str]] = {}

            def sessions_have_streams(records: list) -> None:
                for record in records:
                    sid = record["session_id"]
                    streams = {stream_id(sid, turn["agent_id"]) for turn in record["turns"]}
                    assert streams, f"session {sid} recorded no agent streams"
                    for stream in streams:
                        assert provider.models.has_session(stream), f"provider has no token stream {stream} before deletion"
                    streams_by_session[sid] = streams

            def provider_dropped_session(sid: str) -> None:
                for stream in streams_by_session[sid]:
                    assert not provider.models.has_session(stream), f"provider did not drop the token stream of deleted session {stream}"

            await run_gauntlet(
                proxy_url, http,
                verifier=verifier, max_tokens=MAX_TOKENS,
                dump_dir=f"slime-{ENGINE}", proxy_api_key=PROXY_API_KEY,
                error_log_dir=error_logs.name,
                after_rollouts=sessions_have_streams,
                drop_check=provider_dropped_session,
            )

        finally:
            await proxy.stop()
            await provider.aclose()
            error_logs.cleanup()

    print("\n" + "=" * 70)
    print(f"PASS: Slime transport strict TITO OK — {NUM_AGENTS} agents x 2 turns via")
    print("      proxy -> handler -> SlimeRolloutProvider -> HTTP /generate")
    print(f"      -> {ENGINE} ({model}), the same wire protocol as slime's")
    print("      sgl-router: token-ID prompts in, sampled token IDs/logprobs out,")
    print("      every turn's prompt strictly extends the session token stream,")
    print("      tampered history rejected with 400, keyed routing rejects")
    print("      malformed/unauthorized keys with 401, truncation is reported")
    print("      as finish_reason='length', and stream=true is honored as an")
    print("      SSE replay.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
