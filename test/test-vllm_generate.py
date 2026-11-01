"""End-to-end vllm_generate transport test.

Requires a vLLM /inference/v1/generate endpoint. No training framework imports are needed locally.

Run:
    python test/test-vllm_generate.py
"""

from __future__ import annotations
import asyncio
import logging
import os
from tempfile import TemporaryDirectory

# vllm_generate is vLLM-only: select its fixtures before importing common.
os.environ["INFERENCE_ENGINE"] = "vllm"
os.environ["TRANSPORT_MODE"] = "vllm_generate"

import httpx
from common import ENDPOINT, ENGINE, KEY_DELIMITER, MODEL_NAME, NUM_AGENTS, PROXY_API_KEY, make_verifier, run_gauntlet
from proxyserver.rollout_provider import VLLMRolloutProvider
from proxyserver.service import LLMProxyServer, make_completion_handler, stream_id

logger = logging.getLogger("vllm-generate-test")

MAX_TOKENS = 8192


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


async def main() -> None:
    model = MODEL_NAME
    verifier = make_verifier(model)

    # Wire the real provider stack, exactly as cli.py builds it
    # for `transport_mode: vllm_generate`.  ENDPOINT.root_url is the router or native server root.
    provider = VLLMRolloutProvider(
        router_url=ENDPOINT.root_url,
        api_key=ENDPOINT.api_key,
        context_length=ENDPOINT.context_length,
    )
    error_logs = TemporaryDirectory(prefix="proxy-vllm-generate-errors-")
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
            assert health["status"] == "ok" and health["instance_id"], f"unexpected health: {health}"

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
                dump_dir=f"vllm_generate-{ENGINE}", proxy_api_key=PROXY_API_KEY,
                error_log_dir=error_logs.name,
                after_rollouts=sessions_have_streams,
                drop_check=provider_dropped_session,
            )

        finally:
            await proxy.stop()
            await provider.aclose()
            error_logs.cleanup()

    print("\n" + "=" * 70)
    print(f"PASS: vllm_generate transport strict TITO OK — {NUM_AGENTS} agents x 2 turns via")
    print("      proxy -> handler -> VLLMRolloutProvider -> HTTP /inference/v1/generate")
    print(f"      -> {ENGINE} ({model}): token-ID prompts in, sampled token IDs/logprobs out,")
    print("      every turn's prompt strictly extends the session token stream,")
    print("      tampered history rejected with 400, keyed routing rejects")
    print("      malformed/unauthorized keys with 401, truncation is reported as")
    print("      finish_reason='length', and stream=true is honored as an SSE replay.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
