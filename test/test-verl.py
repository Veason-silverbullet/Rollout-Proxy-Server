"""VeRL actor contracts without Ray/GPU dependencies; --ray also uses real Ray.

Run: python test/test-verl.py [--ray]
"""

from __future__ import annotations
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxyserver.engines import EngineAbort, EngineError
from proxyserver.server import LLMProxyServer, make_completion_handler
from proxyserver.rollout_provider import RayRolloutProvider
from offline_common import FakeTokenizer, check

MODEL = "Qwen3.5-9B"
MESSAGES = [{"role": "user", "content": "hello"}]


class ObjectRef:
    """Local awaiter cancellation must not cancel the remote actor's work."""

    def __init__(self, coro):
        self.remote_task = asyncio.create_task(coro)

    def __await__(self):
        return asyncio.shield(self.remote_task).__await__()

    def cancel(self):
        self.remote_task.cancel()


class Remote:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, **kwargs):
        return ObjectRef(self.fn(**kwargs))


class Server:
    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.gate = None
        self.error = None
        self.output = SimpleNamespace(token_ids=np.array([900, 901]), log_probs=np.array([-0.1, -0.2]),
                                      stop_reason="completed", extra_fields={"global_steps": 0})
        self.generate = Remote(self._generate)

    async def _generate(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return self.output


class Balancer:
    def __init__(self, server, modern=False):
        self.server = server
        self.acquired = []
        self.released = []
        self.started = asyncio.Event()
        self.gate = None
        self.release_started = asyncio.Event()
        self.release_gate = None
        self.acquire_server = Remote(self._acquire)
        self.release_server = Remote(self._release)
        if modern:
            async def acquire_fields():
                return ["prompt_ids", "sampling_params"]

            async def release_fields():
                return ["request_id"]

            self.require_acquire_fields = Remote(acquire_fields)
            self.require_release_fields = Remote(release_fields)

    async def _acquire(self, **kwargs):
        self.acquired.append(kwargs)
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        return "server", self.server

    async def _release(self, **kwargs):
        self.release_started.set()
        if self.release_gate is not None:
            await self.release_gate.wait()
        self.released.append(kwargs)


class FakeRay:
    def __init__(self, lb=None, initialized=True):
        self.lb = lb
        self.initialized = initialized
        self.inits = []
        self.lookups = []
        self.shutdowns = 0
        self.cancellations = 0

    def is_initialized(self):
        return self.initialized

    def init(self, **kwargs):
        self.inits.append(kwargs)
        self.initialized = True

    def get_actor(self, name, **kwargs):
        self.lookups.append((name, kwargs))
        if self.lb is None:
            raise ValueError("actor missing")
        return self.lb

    def shutdown(self):
        self.shutdowns += 1
        self.initialized = False

    def cancel(self, ref, recursive):
        assert recursive is False
        self.cancellations += 1
        ref.cancel()


def provider(lb, engine="vllm"):
    return RayRolloutProvider(lb, inference_engine=engine, tokenizer_loader=lambda path: FakeTokenizer())


async def test_outputs():
    for engine in ("vllm", "sglang"):
        server = Server()
        lb = Balancer(server, modern=True)
        with patch.dict(sys.modules, ray=FakeRay()):
            p = provider(lb, engine)
            for index, (reason, budget, expected) in enumerate((
                ("completed", 2, "length"), ("completed", 8, "stop"),
                ("length", 8, "length"), ("stop", 2, "stop"), (None, 2, "length"),
            )):
                server.output.stop_reason = reason
                sid = f"stream{index}"
                result = await p.generate(MESSAGES, sid, model=MODEL, max_tokens=budget)
                assert result[1:4] == ([900, 901], [-0.1, -0.2], expected)
                assert result[5] == {"weight_version": "0"}
                assert all(type(x) is int for x in result[1])
                assert lb.acquired[-1]["request_id"] == sid
                assert lb.acquired[-1]["prompt_ids"] == result[0]
                assert lb.acquired[-1]["sampling_params"]["logprobs"] is True
                assert lb.released[-1] == {"server_id": "server", "request_id": sid}
                retry = await p.generate(MESSAGES, sid, model=MODEL, max_tokens=budget)
                assert retry == result and len(server.calls) == index + 1
            check(f"{engine}: tokens, reasons, metadata, router fields, exact retries", True)
            assert len({c["request_id"] for c in server.calls}) == len(server.calls)
            for reason, tokens, probs, error in (
                ("aborted", [], [], EngineAbort), ("abort", [900], [-0.1], EngineAbort),
                ("stop", [], [], EngineError), ("stop", [900], None, EngineError),
                ("stop", [900, 901], [-0.1], EngineError),
            ):
                server.output = SimpleNamespace(token_ids=tokens, log_probs=probs, stop_reason=reason)
                try:
                    await p.generate(MESSAGES, "bad", model=MODEL)
                except error:
                    pass
                else:
                    raise AssertionError("invalid output committed")
                assert len(lb.acquired) == len(lb.released)
            server.output = SimpleNamespace(token_ids=[902], log_probs=[-0.3], stop_reason="stop", weight_version="v7")
            recovered = await p.generate(MESSAGES, "bad", model=MODEL)
            assert recovered[1] == [902] and recovered[5] == {"weight_version": "v7"}
            check(f"{engine}: invalid turns release reservations and remain retryable", True)
            await p.aclose()


async def test_cancellation():
    for stage in ("acquire", "generate", "error"):
        server = Server()
        lb = Balancer(server)
        ray = FakeRay()
        with patch.dict(sys.modules, ray=ray):
            p = provider(lb)
            if stage == "acquire":
                lb.gate = asyncio.Event()
            elif stage == "generate":
                server.gate = asyncio.Event()
            else:
                server.error = RuntimeError("engine failed")
            task = asyncio.create_task(p._call_engine([1], {"max_tokens": 3}, "sticky"))
            if stage != "error":
                await (lb.started if stage == "acquire" else server.started).wait()
                task.cancel()
                await asyncio.sleep(0)
                if lb.gate is not None:
                    lb.gate.set()
            try:
                await task
            except (asyncio.CancelledError, RuntimeError):
                pass
            else:
                raise AssertionError("failure or cancellation was swallowed")
            assert lb.released == [{"server_id": "server"}]
            assert ray.cancellations == (1 if stage == "generate" else 0)
            await p.aclose()
            assert ray.shutdowns == 0
            check(f"{stage}: release exactly once, preserve trainer connection", True)


async def test_connection():
    for initialized in (False, True):
        ray = FakeRay(Balancer(Server()), initialized=initialized)
        with patch.dict(sys.modules, ray=ray):
            p = RayRolloutProvider(verl_load_balancer="trainer", ray_address="cluster:6379", ray_namespace="training")
            p.start()
            p.start()
            assert ray.inits == ([] if initialized else [{"address": "cluster:6379", "namespace": "training"}])
            assert ray.lookups == [("trainer", {"namespace": "training"})]
            await p.aclose()
            await p.aclose()
            assert ray.shutdowns == (0 if initialized else 1)
    ray = FakeRay(initialized=False)
    with patch.dict(sys.modules, ray=ray):
        p = RayRolloutProvider(verl_load_balancer="missing")
        try:
            p.start()
        except ValueError:
            pass
        else:
            raise AssertionError("missing actor accepted")
        assert ray.shutdowns == 1
    with patch.dict(sys.modules, ray=None):
        try:
            RayRolloutProvider(verl_load_balancer="trainer").start()
        except RuntimeError as e:
            assert "same Ray version" in str(e)
        else:
            raise AssertionError("missing Ray accepted")
    check("named actors, optional dependency, connection ownership and failed lookup cleanup", True)


async def test_repeated_cancellation():
    for stage in ("acquire", "release"):
        server = Server()
        lb = Balancer(server)
        gate = asyncio.Event()
        if stage == "acquire":
            lb.gate = gate
        else:
            lb.release_gate = gate
        with patch.dict(sys.modules, ray=FakeRay()):
            p = provider(lb)
            task = asyncio.create_task(p._call_engine([1], {"max_tokens": 3}, "sticky"))
            await (lb.started if stage == "acquire" else lb.release_started).wait()
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
            try:
                assert not task.done(), "cleanup abandoned an outstanding Ray RPC"
            finally:
                gate.set()
                result = await asyncio.gather(task, return_exceptions=True)
            assert isinstance(result[0], asyncio.CancelledError)
            assert lb.released == [{"server_id": "server"}]
            assert len(server.calls) == (0 if stage == "acquire" else 1)
            await p.aclose()
            check(f"{stage}: repeated cancellation drains RPC and releases exactly once", True)


async def exercise_http(p, server=None):
    proxy = LLMProxyServer(api_key="test", key_delimiter="-", save_rollout_sessions=False,
                          completion_handler=make_completion_handler(p), on_session_deleted=p.release_session)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as client:
        async def chat(sid, messages):
            return await client.post("/v1/chat/completions", headers={"Authorization": f"Bearer test-{sid}-agent"},
                                     json={"model": MODEL, "messages": messages, "max_tokens": 8})

        first = await chat("trial", MESSAGES)
        assert first.status_code == 200, first.text
        retry = await chat("trial", MESSAGES)
        assert retry.status_code == 200, retry.text
        reply = first.json()["choices"][0]["message"]["content"]
        second = await chat("trial", MESSAGES + [{"role": "assistant", "content": reply},
                                                {"role": "user", "content": "continue"}])
        assert second.status_code == 200, second.text
        record = (await client.get("/sessions/trial")).json()
        assert len(record["turns"]) == 2
        assert record["turns"][0]["completion_token_ids"] == [900, 901]
        assert record["turns"][0]["weight_version"] == "0"
        deleted = await client.delete("/sessions/trial")
        assert deleted.status_code == 200, deleted.text
        assert not p._session_models
        if server is not None:
            assert len(server.calls) == 2
            first_prompt = server.calls[0]["prompt_ids"]
            assert server.calls[1]["prompt_ids"][:len(first_prompt) + 2] == first_prompt + [900, 901]
            server.gate = asyncio.Event()
            server.started.clear()
            pending = asyncio.create_task(chat("cancel", MESSAGES))
            await server.started.wait()
            response = await client.delete("/sessions/cancel")
            assert response.status_code == 200, response.text
            # Cancelling an ASGI task may propagate CancelledError to the
            # test client; no cancelled completion may enter the record.
            try:
                response = await pending
            except asyncio.CancelledError:
                pass
            else:
                assert response.status_code == 410, response.text
            assert not p._session_models
    await proxy.stop()
    check("local HTTP, recording, exact retry, strict continuation and deletion", True)


async def main():
    await test_connection()
    await test_outputs()
    await test_cancellation()
    await test_repeated_cancellation()
    server = Server()
    lb = Balancer(server)
    ray = FakeRay()
    with patch.dict(sys.modules, ray=ray):
        p = provider(lb)
        await exercise_http(p, server)
        assert len(lb.acquired) == len(lb.released) == 3
        assert ray.cancellations == 1
        await p.aclose()


def test_real_ray():
    import ray

    @ray.remote(num_cpus=0)
    class RolloutActor:
        def __init__(self):
            self.started = False
            self.cancelled = False

        async def generate(self, prompt_ids, sampling_params, request_id):
            if prompt_ids == [999]:
                self.started = True
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
            return SimpleNamespace(token_ids=[900, 901], log_probs=[-0.1, -0.2],
                                   stop_reason="completed", extra_fields={"global_steps": 0})

        async def status(self):
            return self.started, self.cancelled

    @ray.remote(num_cpus=0)
    class LoadBalancerActor:
        def __init__(self, server):
            self.server = server
            self.inflight = 0
            self.acquire_gate = asyncio.Event()
            self.acquire_started = False

        async def acquire_server(self, request_id):
            self.inflight += 1
            if request_id == "cancel-acquire":
                self.acquire_started = True
                await self.acquire_gate.wait()
            return "server", self.server

        def waiting(self):
            return self.acquire_started

        def unblock(self):
            self.acquire_gate.set()

        def release_server(self, server_id):
            self.inflight -= 1

        def count(self):
            return self.inflight

        def require_acquire_fields(self):
            return []

        def require_release_fields(self):
            return []

    ray.init(num_cpus=1, include_dashboard=False, namespace="proxy-test")
    try:
        server = RolloutActor.remote()
        lb = LoadBalancerActor.options(name="trainer").remote(server)
        p = RayRolloutProvider(verl_load_balancer="trainer", ray_namespace="proxy-test",
                              tokenizer_loader=lambda path: FakeTokenizer())
        p.start()
        asyncio.run(exercise_http(p))

        async def cancel_generation():
            task = asyncio.create_task(p._call_engine([999], {"max_tokens": 3}, "cancel"))
            for _ in range(100):
                started, _ = await server.status.remote()
                if started:
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError("Ray generation did not start")
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=10)
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("Ray cancellation did not propagate")
            # Ray may resolve the cancelled ObjectRef before the async
            # actor has processed its Task.cancel(). Observe that as well.
            for _ in range(100):
                _, cancelled = await server.status.remote()
                if cancelled:
                    break
                await asyncio.sleep(0.05)
            assert cancelled

        asyncio.run(cancel_generation())

        async def cancel_acquisition():
            task = asyncio.create_task(p._call_engine([1], {"max_tokens": 3}, "cancel-acquire"))
            try:
                for _ in range(100):
                    if await lb.waiting.remote():
                        break
                    await asyncio.sleep(0.05)
                else:
                    raise AssertionError("Ray acquisition did not start")
                for _ in range(3):
                    task.cancel()
                    await asyncio.sleep(0)
                assert not task.done(), "repeated cancellation abandoned the remote reservation"
            finally:
                await lb.unblock.remote()
                result = await asyncio.gather(task, return_exceptions=True)
            assert isinstance(result[0], asyncio.CancelledError)
            assert await lb.count.remote() == 0

        asyncio.run(cancel_acquisition())
        asyncio.run(p.aclose())
        assert ray.is_initialized() and ray.get(lb.count.remote()) == 0
        check("real Ray: named actor lookup, ObjectRef await, serialization, cancellation and reservation release", True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(main(), timeout=30))
    if "--ray" in sys.argv:
        test_real_ray()
    print("PASS: VeRL transport")
