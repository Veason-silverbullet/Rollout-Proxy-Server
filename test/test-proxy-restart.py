"""Instance-bound rollouts over real proxy apps with scripted inference."""

import asyncio
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import httpx
from fastapi import HTTPException
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proxyserver import client
from proxyserver.protocol import PROXY_INSTANCE_HEADER
from proxyserver.server import LLMProxyServer, make_completion_handler
from proxyserver.tokenization import fingerprint as fp
from offline_common import ScriptedProvider

URL = "http://proxy"
MODEL = "Qwen3.5-9B"
KEY = "base-key"
MESSAGES = [{"role": "user", "content": "hello"}]
FINGERPRINT = {
    "model": MODEL, "algorithm": fp.FINGERPRINT_ALGORITHM,
    "corpus_sha256": fp.probe_corpus_digest(), "fingerprint": "same", "template_pin": "pin",
}


def make_server(directory, *, delimiter="|"):
    provider = ScriptedProvider()
    provider.weight_version = "1"
    provider.set_runtime_sampling_overrides({"temperature": 1.0})
    capture = {"effective": False}

    def set_capture(enabled):
        capture["effective"] = enabled
        return dict(capture)

    server = LLMProxyServer(
        api_key=KEY, key_delimiter=delimiter, save_rollout_sessions=False,
        error_log_dir=directory,
        completion_handler=make_completion_handler(provider),
        on_session_deleted=provider.release_session,
        get_sampling_overrides=provider.get_sampling_overrides,
        set_sampling_overrides=provider.set_runtime_sampling_overrides,
        get_routed_experts_config=lambda: dict(capture),
        set_routed_experts_config=set_capture,
        get_tokenizer_fingerprint=lambda model: FINGERPRINT,
    )
    return server, provider, capture


class SwitchingTransport(httpx.AsyncBaseTransport):
    def __init__(self, server):
        self.server = server
        self.after_response = None

    async def handle_async_request(self, request):
        response = await httpx.ASGITransport(app=self.server.app).handle_async_request(request)
        if self.after_response is not None:
            self.after_response(request)
        return response


class RestartTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        root = Path(self.directory.name)
        self.old, self.old_provider, _ = make_server(root / "old")
        self.new, self.new_provider, self.new_capture = make_server(root / "new")
        self.transport = SwitchingTransport(self.old)
        self.http = httpx.AsyncClient(transport=self.transport, base_url=URL)
        self.proxy = client.AsyncProxyClient(URL, KEY, http_client=self.http)

    async def asyncTearDown(self):
        await self.proxy.aclose()
        await self.http.aclose()
        self.directory.cleanup()

    async def chat(self, key, **kwargs):
        return await self.http.post("/v1/chat/completions", headers={"Authorization": "Bearer " + key},
                                    json={"model": MODEL, "messages": MESSAGES, **kwargs})

    async def test_restart_cannot_create_or_collect_a_partial_rollout(self):
        self.assertIsNone(self.proxy.instance_id)
        self.assertEqual(self.proxy.agent_key("trial"), KEY + "|trial|agent")
        instance = await self.proxy.bind_instance()
        self.assertEqual(instance, self.old.instance_id)
        await self.proxy.synchronize(sampling={"temperature": 1.0}, routed_experts=False)
        key = self.proxy.agent_key("trial", "parent")
        for _ in range(2):
            self.assertEqual((await self.chat(key)).status_code, 200)
        self.assertEqual(self.old_provider.engine_calls, 1)
        await self.proxy.finalize("trial")
        record = await self.proxy.collect("trial", expected_weight_version="1",
                                          expected_sampling={"temperature": 1.0}, strict=True)
        self.assertEqual(len(record["turns"]), 1)

        self.transport.server = self.new
        stale = await self.chat(self.proxy.agent_key("trial", "child"))
        self.assertEqual(stale.status_code, 412)
        self.assertEqual(stale.headers[PROXY_INSTANCE_HEADER], self.new.instance_id)
        self.assertEqual(self.new_provider.engine_calls, 0)
        self.assertIsNone(self.new.recorder.dump_session("trial"))

        # Legacy callers can still open a plausible partial record on the new
        # instance. Bound collectors must not consume, seal, or delete it.
        self.assertEqual((await self.chat(client.agent_key(KEY, "trial", "child"))).status_code, 200)
        operations = [
            partial(self.proxy.collect, "trial"), partial(self.proxy.finalize, "trial"),
            self.proxy.bind_instance,
            partial(self.proxy.synchronize, sampling={"temperature": 2.0}, routed_experts=True),
            partial(self.proxy.request, "GET", "/tokenizer_fingerprint", params={"model": MODEL}),
        ]
        for operation in operations:
            with self.assertRaises(httpx.HTTPStatusError) as caught:
                await operation()
            self.assertEqual(caught.exception.response.status_code, 412)
        self.assertNotIn("trial", self.new._finalization_tasks)
        self.assertEqual(self.new_provider.get_sampling_overrides()["runtime"], {"temperature": 1.0})
        self.assertFalse(self.new_capture["effective"])
        await self.proxy.close("trial")
        self.assertIsNotNone(self.new.recorder.dump_session("trial"))
        self.assertEqual(self.proxy.instance_id, instance)
        with self.assertRaises(ValueError):
            await self.proxy.bind_instance(self.new.instance_id)

        async with client.AsyncProxyClient(URL, KEY, http_client=self.http) as recovered:
            self.assertEqual(await recovered.bind_instance(), self.new.instance_id)
            await recovered.synchronize(sampling={"temperature": 1.0})
            self.assertEqual((await self.chat(recovered.agent_key("fresh"))).status_code, 200)
            await recovered.finalize("fresh")
            self.assertEqual((await recovered.collect("fresh"))["session_id"], "fresh")
            await recovered.close("fresh")

    async def test_header_check_precedes_parsing_and_session_admission(self):
        response = await self.http.post("/v1/chat/completions", content="{", headers={
            "Authorization": "Bearer " + client.agent_key(KEY, "trial"),
            PROXY_INSTANCE_HEADER: self.new.instance_id,
        })
        self.assertEqual(response.status_code, 412)
        self.assertEqual(response.headers[PROXY_INSTANCE_HEADER], self.old.instance_id)
        self.assertIsNone(self.old.recorder.dump_session("trial"))
        self.assertEqual(self.old_provider.engine_calls, 0)
        response = await self.http.post("/v1/chat/completions", json={"model": MODEL, "messages": MESSAGES}, headers={
            "Authorization": "Bearer " + client.agent_key(KEY, "trial"),
            PROXY_INSTANCE_HEADER: self.old.instance_id,
        })
        self.assertEqual(response.status_code, 200)
        # The binding middleware also preserves streaming response formatting.
        response = await self.chat(client.agent_key(KEY, "stream", instance_id=self.old.instance_id), stream=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("data: [DONE]", response.text)
        self.assertEqual(response.headers[PROXY_INSTANCE_HEADER], self.old.instance_id)

    async def test_restart_between_control_updates_is_rejected(self):
        await self.proxy.bind_instance()

        def restart(request):
            if request.url.path == "/sampling_overrides":
                self.transport.server = self.new

        self.transport.after_response = restart
        with self.assertRaises(httpx.HTTPStatusError) as caught:
            await self.proxy.synchronize(sampling={"temperature": 2.0}, routed_experts=True)
        self.assertEqual(caught.exception.response.status_code, 412)
        self.assertEqual(self.old_provider.get_sampling_overrides()["runtime"], {"temperature": 2.0})
        self.assertEqual(self.new_provider.get_sampling_overrides()["runtime"], {"temperature": 1.0})
        self.assertFalse(self.new_capture["effective"])
        self.assertEqual(self.proxy.instance_id, self.old.instance_id)

    async def test_concurrent_discovery_does_not_rebind(self):
        self.transport.after_response = lambda request: setattr(self.transport, "server", self.new)
        results = await asyncio.gather(self.proxy.bind_instance(), self.proxy.bind_instance(), return_exceptions=True)
        self.assertEqual(results[0], self.old.instance_id)
        self.assertIsInstance(results[1], httpx.HTTPStatusError)
        self.assertEqual(results[1].response.status_code, 412)
        self.assertEqual(self.proxy.instance_id, self.old.instance_id)

    async def test_failed_explicit_binding_stays_pinned(self):
        self.transport.server = self.new
        for expected in (self.old.instance_id, None):
            with self.assertRaises(httpx.HTTPStatusError) as caught:
                await self.proxy.bind_instance(expected)
            self.assertEqual(caught.exception.response.status_code, 412)
            self.assertEqual(self.proxy.instance_id, self.old.instance_id)

    async def test_bound_requests_reject_missing_or_wrong_response_echo(self):
        for headers in ({}, {PROXY_INSTANCE_HEADER: self.new.instance_id}):
            def handle(request):
                self.assertEqual(request.headers[PROXY_INSTANCE_HEADER], self.old.instance_id)
                return httpx.Response(200, headers=headers, json={"session_id": "trial"})

            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
                async with client.AsyncProxyClient(URL, http_client=http, instance_id=self.old.instance_id) as proxy:
                    with self.assertRaisesRegex(ValueError, "instance changed or was not reported"):
                        await proxy.collect("trial")
                    self.assertEqual(proxy.instance_id, self.old.instance_id)

    async def test_discovery_requires_consistent_health_and_header(self):
        for body, headers in (
            ({"status": "ok"}, {}),
            ({"instance_id": "invalid"}, {}),
            ({"instance_id": self.old.instance_id}, {}),
            ({"instance_id": self.old.instance_id}, {PROXY_INSTANCE_HEADER: self.new.instance_id}),
        ):
            async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=body, headers=headers)
            )) as http:
                async with client.AsyncProxyClient(URL, http_client=http) as proxy:
                    with self.assertRaises(ValueError):
                        await proxy.bind_instance()
                    self.assertIsNone(proxy.instance_id)

    async def test_cleanup_does_not_hide_unconfirmed_precondition_failures(self):
        for headers in ({}, {PROXY_INSTANCE_HEADER: "invalid"}, {PROXY_INSTANCE_HEADER: self.old.instance_id}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(412, headers=headers)
            )) as http:
                async with client.AsyncProxyClient(URL, http_client=http, instance_id=self.old.instance_id) as proxy:
                    with self.assertRaises(httpx.HTTPStatusError) as caught:
                        await proxy.close("trial")
                    self.assertEqual(caught.exception.response.status_code, 412)


@contextmanager
def sync_http(request):
    with ExitStack() as stack:
        for method in ("get", "post", "put", "delete"):
            stack.enter_context(patch.object(client.httpx, method, partial(request, method.upper())))
        yield


class SynchronousTests(unittest.TestCase):
    def test_helpers_preserve_binding_through_setup_and_collection(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            old, _, _ = make_server(Path(directory) / "old")
            new, _, _ = make_server(Path(directory) / "new")
            old_http = stack.enter_context(TestClient(old.app, base_url=URL))
            new_http = stack.enter_context(TestClient(new.app, base_url=URL))
            active = old_http

            def request(method, url, **kwargs):
                kwargs.pop("timeout", None)  # ASGI TestClient has no network timeout.
                return active.request(method, url, **kwargs)

            stack.enter_context(sync_http(request))
            stack.enter_context(patch.object(fp, "checkpoint_fingerprint", return_value={"fingerprint": "same"}))
            instance = client.get_proxy_instance(URL)
            self.assertEqual(instance, old.instance_id)
            bound = {"expected_instance_id": instance}
            client.synchronize_proxy(URL, api_key=KEY, sampling={"temperature": 1.0}, **bound)
            self.assertEqual(client.check_tokenizer(URL, "checkpoint", MODEL, **bound), FINGERPRINT)
            key = client.agent_key(KEY, "trial", instance_id=instance)
            self.assertEqual(old_http.post("/v1/chat/completions", headers={"Authorization": "Bearer " + key},
                                          json={"model": MODEL, "messages": MESSAGES}).status_code, 200)
            client.finalize_session(URL, "trial", **bound)
            self.assertEqual(len(client.get_session(URL, "trial", strict=True, **bound)["turns"]), 1)
            active = new_http
            new.recorder.ensure_session("trial", model_name=MODEL)
            for operation in (
                partial(client.get_proxy_instance, URL, **bound),
                partial(client.get_session, URL, "trial", **bound),
                partial(client.finalize_session, URL, "trial", **bound),
                partial(client.synchronize_proxy, URL, api_key=KEY, sampling={}, **bound),
                partial(client.check_tokenizer, URL, "checkpoint", MODEL, **bound),
            ):
                with self.assertRaises(httpx.HTTPStatusError) as caught:
                    operation()
                self.assertEqual(caught.exception.response.status_code, 412)
            client.end_session(URL, "trial", **bound)
            self.assertIsNotNone(new.recorder.dump_session("trial"))

    def test_missing_echo_is_rejected_by_every_bound_helper(self):
        with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"instance_id": "a" * 32})
        )) as http, sync_http(http.request), patch.object(fp, "checkpoint_fingerprint", return_value={"fingerprint": "same"}):
            bound = {"expected_instance_id": "a" * 32}
            for operation in (
                partial(client.get_proxy_instance, URL, **bound),
                partial(client.get_session, URL, "trial", **bound),
                partial(client.finalize_session, URL, "trial", **bound),
                partial(client.end_session, URL, "trial", **bound),
                partial(client.synchronize_proxy, URL, api_key=KEY, sampling={}, **bound),
                partial(client.check_tokenizer, URL, "checkpoint", MODEL, **bound),
            ):
                with self.assertRaisesRegex(ValueError, "instance changed or was not reported"):
                    operation()

    def test_binding_keeps_delimiters_and_general_session_ids(self):
        with TemporaryDirectory() as directory:
            for delimiter in ("|", ":", "::"):
                server, _, _ = make_server(directory, delimiter=delimiter)
                for instance in (None, server.instance_id):
                    key = client.agent_key(KEY, "trial_42", "agent.1", key_delimiter=delimiter, instance_id=instance)
                    self.assertEqual(server.parse_session_key(key), ("trial_42", "agent.1"))
                stale = client.agent_key(KEY, "trial_42", key_delimiter=delimiter, instance_id="a" * 32)
                with self.assertRaises(HTTPException) as caught:
                    server.parse_session_key(stale)
                self.assertEqual(caught.exception.status_code, 412)
        for invalid in ("", "not-an-instance", "A" * 32, 123):
            with self.assertRaises(ValueError):
                client.AsyncProxyClient(URL, instance_id=invalid)
            with self.assertRaises(ValueError):
                client.agent_key(KEY, "trial", instance_id=invalid)

    def test_cleanup_does_not_hide_unconfirmed_precondition_failures(self):
        instance = "a" * 32
        for headers in ({}, {PROXY_INSTANCE_HEADER: "invalid"}, {PROXY_INSTANCE_HEADER: instance}):
            with httpx.Client(transport=httpx.MockTransport(
                lambda request: httpx.Response(412, headers=headers)
            )) as http, sync_http(http.request):
                with self.assertRaises(httpx.HTTPStatusError) as caught:
                    client.end_session(URL, "trial", expected_instance_id=instance)
                self.assertEqual(caught.exception.response.status_code, 412)


if __name__ == "__main__":
    unittest.main()
