"""End-to-end test of **relay mode with the real framework-side worker**
(``InferenceWorkerClient``, verl transport), strict TITO.

Topology (all in one process + local Ray actors, over real
HTTP/WebSocket/Ray RPC):

    50 mock agents              LLMProxyServer (relay mode)
    (OpenAI SDK)   --HTTP-->    InferenceRelay
                                     |  WebSocket:
                                     |  completion_request / completion_response
                                     v
                                InferenceWorkerClient (real, dials out)
                                RayRolloutProvider (TokenStreamManager, strict TITO)
                                     |  Ray RPC: lb.acquire_server / server.generate
                                     v
                        RolloutServerActor Ray actors --> real engine
                        (prompts tokenized locally from proxyserver/tokenization via
                        mapping.json)

This is the production deployment for verl: the proxy runs standalone (no
Ray anywhere near it) and the training cluster dials out with a real
:class:`~proxyserver.worker_client.InferenceWorkerClient` whose
:class:`~proxyserver.rollout_provider.RayRolloutProvider` reaches the
rollout servers through the framework's load balancer.  Where
``test-relay.py`` fakes the worker to pin the relay *protocol*, this test
runs the real worker end-to-end; only the two boundaries the training
cluster would provide are faked with local Ray actors — the load balancer
(``acquire_server`` / ``release_server``) and the rollout server
(``generate``), which fulfills each request against the real engine using a
**token-ID prompt**.

``INFERENCE_ENGINE`` (``vllm`` by default) selects both the endpoints from
``test/test_engines.yaml`` (the fake rollout servers are spread over every
``OPENAI_BASE_URL`` entry) and the adapter inside the worker's provider, so
the run exercises the real backend: under ``sglang`` the actor calls
SGLang's native ``/generate`` and reports its raw
``meta_info.finish_reason.type``, exactly as the reference (verl)
``AsyncSGLangServer`` does.

Per-session verification criteria are shared with the relay test — see
``common.run_agent``.  Tampered history must be rejected with a 400 (the
worker reports ``invalid_request``), never silently re-tokenized.

Run:
    python test/test-worker.py
    INFERENCE_ENGINE=sglang python test/test-worker.py
"""

from __future__ import annotations
import asyncio
import logging
import zlib
from typing import Any
import httpx
import ray
from common import (ENDPOINTS, ENGINE, KEY_DELIMITER, NUM_AGENTS, PROXY_API_KEY, MODEL_NAME,
                    EngineEndpoint, TokenOutput,
                    make_rollout_server, make_verifier, run_gauntlet)
from proxyserver.server import LLMProxyServer, stream_id
from proxyserver.worker_client import InferenceWorkerClient

logger = logging.getLogger("worker-test")

MAX_TOKENS = 8192
NUM_FAKE_SERVERS = 2


# ---------------------------------------------------------------------------
# Fake Ray actors: the framework's load balancer + rollout server boundaries
# ---------------------------------------------------------------------------


@ray.remote(max_concurrency=16)
class RolloutServerActor:
    """Stands in for the framework's rollout server actor (vLLM or SGLang).

    A thin Ray wrapper around ``common.make_rollout_server`` — the same
    stand-in the relay test drives — so both tests speak to the real engine
    through one implementation of the ``generate`` contract.  ``generate``
    is ``async`` because Ray runs async actor methods natively and the
    underlying clients are httpx-async.
    """

    def __init__(self, endpoint: EngineEndpoint):
        self._server = make_rollout_server(endpoint)

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
    ) -> TokenOutput:
        return await self._server.generate(prompt_ids, sampling_params)


@ray.remote
class FakeLoadBalancer:
    """Stands in for the framework's load-balancer actor (e.g. verl's
    ``GlobalRequestLoadBalancer``).

    ``acquire_server`` is sticky on the request key (the provider passes the
    session_id), mirroring the framework's prefix-cache-friendly routing.
    """

    def __init__(self, servers: list[Any]):
        self._servers = list(servers)

    def acquire_server(self, request_id: str) -> tuple[int, Any]:
        idx = zlib.crc32(request_id.encode()) % len(self._servers)
        return idx, self._servers[idx]

    def release_server(self, server_id: int) -> None:
        pass


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


async def main() -> None:
    model = MODEL_NAME
    verifier = make_verifier(model)

    # Fake framework boundaries: rollout servers + load balancer, spread
    # round-robin over every endpoint in test/test_engines.yaml.
    ray.init(num_cpus=2, include_dashboard=False, logging_level=logging.WARNING)
    servers = [
        RolloutServerActor.remote(ENDPOINTS[i % len(ENDPOINTS)])
        for i in range(NUM_FAKE_SERVERS)
    ]
    load_balancer = FakeLoadBalancer.remote(servers)

    # relay_request_timeout matches the 900s the agents and the engine HTTP
    # clients already use, as in test-relay.py.
    proxy = LLMProxyServer(host="127.0.0.1", port=0, api_key=PROXY_API_KEY,
                           key_delimiter=KEY_DELIMITER, relay_request_timeout=900)
    assert proxy.mode == "relay"
    proxy_url = await proxy.start()
    logger.info("Proxy started at %s (mode=%s)", proxy_url, proxy.mode)

    ws_url = proxy_url.replace("http://", "ws://") + "/ws/worker"
    worker = InferenceWorkerClient(
        proxy_ws_url=ws_url,
        load_balancer=load_balancer,
        inference_engine=ENGINE,
        worker_id="e2e-verl-worker",
    )
    await worker.start()

    async with httpx.AsyncClient(timeout=30) as http:
        try:
            # Wait until the worker's handshake lands.
            for _ in range(100):
                health = (await http.get(f"{proxy_url}/health")).json()
                if health.get("connected_workers") == 1:
                    break
                await asyncio.sleep(0.1)
            assert health["mode"] == "relay", f"unexpected health: {health}"
            assert health["connected_workers"] == 1, f"unexpected health: {health}"
            assert worker.is_connected

            # The shared gauntlet (common.run_gauntlet); drop_wait covers the
            # session_closed WebSocket hop to the real worker.
            agent_id_by_session: dict[str, str] = {}

            def record_agent_ids(records: list) -> None:
                agent_id_by_session.update({r["session_id"]: r["turns"][0]["agent_id"] for r in records})

            def worker_dropped_stream(sid: str) -> None:
                stream = stream_id(sid, agent_id_by_session[sid])
                assert not worker._provider.models.has_session(stream), f"worker did not drop the token stream of deleted session {stream}"

            await run_gauntlet(
                proxy_url, http,
                verifier=verifier, max_tokens=MAX_TOKENS,
                dump_dir=f"worker-{ENGINE}", proxy_api_key=PROXY_API_KEY,
                after_rollouts=record_agent_ids,
                drop_check=worker_dropped_stream,
                drop_wait=0.2,
            )

        finally:
            await worker.stop()
            await proxy.stop()
            ray.shutdown()

    print("\n" + "=" * 70)
    print(f"PASS: relay mode + real verl worker strict TITO OK — {NUM_AGENTS} agents")
    print(f"      x 2 turns via proxy -> WebSocket -> InferenceWorkerClient ->")
    print(f"      RayRolloutProvider[{ENGINE}] -> Ray RPC -> {ENGINE} ({model}):")
    print("      token-ID prompts in, sampled token IDs/logprobs out, every")
    print("      turn's prompt strictly extends the session token stream,")
    print("      tampered history rejected with 400, keyed routing")
    print("      (api_key-embedded session, no registration) rejects")
    print("      malformed/unauthorized keys with 401, truncation is reported")
    print("      as finish_reason='length', and stream=true is honored as an")
    print("      SSE replay.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
