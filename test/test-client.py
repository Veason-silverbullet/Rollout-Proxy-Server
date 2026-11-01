"""Offline HTTP contracts for optional collector setup and cleanup helpers."""

import json
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import httpx
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver import client
from proxyserver.model_registry import MIN_TEMPERATURE
from proxyserver.service import LLMProxyServer, make_completion_handler
from proxyserver.tokenization import fingerprint as fp
from sandbox import utils
from offline_common import ScriptedProvider, check

MODEL = "Qwen3.5-9B"
URL = "http://proxy"
PAYLOAD = {
    "model": MODEL, "algorithm": fp.FINGERPRINT_ALGORITHM,
    "corpus_sha256": fp.probe_corpus_digest(),
    "fingerprint": "token-identity", "template_pin": "serving-template",
}


@contextmanager
def mock_http(handler):
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with (patch.object(client.httpx, "get", http.get),
              patch.object(client.httpx, "put", http.put),
              patch.object(client.httpx, "post", http.post),
              patch.object(client.httpx, "delete", http.delete)):
            yield


def raises(kind, call, message=None):
    try:
        call()
    except kind as error:
        if message is not None:
            assert message in str(error), str(error)
        return error
    raise AssertionError(f"Expected {kind.__name__}")


def test_collection_identity():
    def valid(request):
        assert request.extensions["timeout"]["read"] == 60
        return httpx.Response(200, json={"session_id": "trial"})

    with mock_http(valid):
        assert client.get_session(URL, "trial") == {"session_id": "trial"}
    for body in ({"session_id": "other"}, {}, [], None):
        with mock_http(lambda request: httpx.Response(200, json=body)):
            raises(ValueError, lambda: client.get_session(URL, "trial"))
    with mock_http(lambda request: httpx.Response(404)):
        error = raises(httpx.HTTPStatusError, lambda: client.get_session(URL, "trial"))
        assert error.response.status_code == 404
    with mock_http(lambda request: httpx.Response(503)):
        raises(httpx.HTTPStatusError, lambda: client.get_session(URL, "trial"))
    assert utils.get_session is client.get_session
    assert utils.end_session is client.end_session
    check("collection rejects wrong/malformed records and missing sessions", True)


def test_cleanup():
    # Cleanup must only delete, including retries after a lost response.
    calls = []

    def handler(request):
        calls.append(request.method)
        assert request.method == "DELETE"
        assert request.url.path == "/sessions/trial"
        assert request.extensions["timeout"]["read"] == 10
        if len(calls) == 1:
            raise httpx.ReadTimeout("delete response lost", request=request)
        if len(calls) == 2:
            return httpx.Response(503)
        return httpx.Response(404)

    with mock_http(handler), patch.object(client.time, "sleep") as sleep:
        client.end_session(URL + "/", "trial")
    assert calls == ["DELETE", "DELETE", "DELETE"]
    assert [c.args[0] for c in sleep.call_args_list] == [0.5, 1.0]

    for deletion, attempts in ((400, 1), (401, 1), (501, 1), (429, 4), (503, 4), ("timeout", 4)):
        calls = []

        def handler(request):
            calls.append(request.method)
            assert request.method == "DELETE"
            if deletion == "timeout":
                raise httpx.ConnectTimeout("unavailable", request=request)
            return httpx.Response(deletion)

        with mock_http(handler), patch.object(client.time, "sleep") as sleep:
            raises(httpx.HTTPError, lambda: client.end_session(URL, "trial"))
        assert calls.count("DELETE") == attempts and sleep.call_count == attempts - 1

    check("cleanup only deletes, retries transient failures, and propagates terminal failures", True)


def test_cleanup_preserves_completion_state():
    with TemporaryDirectory() as directory:
        proxy = LLMProxyServer(
            completion_handler=make_completion_handler(ScriptedProvider()),
            session_dir=directory, error_log_dir=directory, save_rollout_sessions=True,
        )
        with TestClient(proxy.app) as http:
            with (patch.object(client.httpx, "get", http.get),
                  patch.object(client.httpx, "post", http.post),
                  patch.object(client.httpx, "delete", http.delete)):
                for sid, finalize in (("abandoned", False), ("finished", True)):
                    proxy.recorder.record_completion(
                        sid, [{"role": "user", "content": "hi"}], "answer",
                        [2], [-0.1], prompt_token_ids=[1],
                    )
                    if finalize:
                        client.finalize_session(URL, sid)
                        assert client.get_session(URL, sid)["completed"] is True
                    client.end_session(URL, sid)
                    snapshot = json.loads((proxy.session_store.run_dir / f"{sid}.json").read_text())
                    assert snapshot["completed"] is finalize
                    assert proxy.recorder.is_deleted(sid)
    check("synchronous cleanup preserves finalized and abandoned recording states", True)


def test_setup_failures():
    for bad in ({"top_p": float("nan")}, {"top_p": float("inf")},
                {"top_p": True}, {"top_p": 10 ** 999}, {"max_tokens": 8}, []):
        with patch.object(client.httpx, "put") as put:
            raises(ValueError, lambda: client.synchronize_proxy(URL, api_key="key", sampling=bad))
            put.assert_not_called()
    with patch.object(client.httpx, "put") as put:
        raises(ValueError, lambda: client.synchronize_proxy(URL, api_key="key"))
        raises(ValueError, lambda: client.synchronize_proxy(URL, api_key="key", sampling={}, routed_experts=1))
        put.assert_not_called()
    for body in (
        {"runtime": {"top_p": 1.0}, "effective": {"top_p": 0.9}},
        {"runtime": {"top_p": 1.0}, "effective": {"top_p": True}},
        {"runtime": {"top_p": 1.0, "temperature": 1.0}, "effective": {"top_p": 1.0}},
        {}, [],
    ):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(200, json=body)

        with mock_http(handler):
            raises(ValueError, lambda: client.synchronize_proxy(
                URL, api_key="key", sampling={"top_p": 1.0}, routed_experts=False,
            ))
        assert calls == ["/sampling_overrides"]
    for body in ({"effective": 1}, {"effective": False}, {}):
        with mock_http(lambda request: httpx.Response(200, json=body)):
            raises(ValueError, lambda: client.synchronize_proxy(URL, api_key="key", routed_experts=True))
    with mock_http(lambda request: httpx.Response(501)):
        raises(httpx.HTTPStatusError, lambda: client.synchronize_proxy(URL, api_key="key", routed_experts=True))
    # Clearing must verify the runtime layer, even though no effective keys
    # were requested. A later capture failure leaves the earlier write applied.
    with mock_http(lambda request: httpx.Response(200, json={"runtime": {"top_p": 1.0}, "effective": {}})):
        raises(ValueError, lambda: client.synchronize_proxy(URL, api_key="key", sampling={}))
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/sampling_overrides":
            return httpx.Response(200, json={"runtime": {}, "effective": {}})
        return httpx.Response(503)

    with mock_http(handler):
        raises(httpx.HTTPStatusError, lambda: client.synchronize_proxy(URL, api_key="key", sampling={}, routed_experts=True))
    assert calls == ["/sampling_overrides", "/routed_experts"]
    check("setup rejects unapplied/malformed settings and stops on HTTP errors", True)


def test_tokenizer_checks():
    with TemporaryDirectory() as directory, patch.object(fp, "checkpoint_fingerprint", return_value=PAYLOAD):
        path = Path(directory) / "nested" / "pins.json"
        for key, value in (("model", "other"), ("algorithm", "other"), ("corpus_sha256", "other"),
                           ("fingerprint", "other"), ("template_pin", None)):
            remote = {**PAYLOAD, key: value}
            with mock_http(lambda request: httpx.Response(200, json=remote)):
                raises(ValueError, lambda: client.check_tokenizer(URL, "/checkpoint", MODEL, pin_file=path))
            assert not path.exists()
        for status in (404, 501):
            with mock_http(lambda request: httpx.Response(status)):
                raises(httpx.HTTPStatusError, lambda: client.check_tokenizer(URL, "/checkpoint", MODEL, pin_file=path))
            assert not path.exists()
        with mock_http(lambda request: httpx.Response(200, json=PAYLOAD)):
            assert client.check_tokenizer(URL, "/checkpoint", MODEL) == PAYLOAD
            assert not path.exists()
            client.check_tokenizer(URL, "/checkpoint", MODEL, pin_file=path)
            assert json.loads(path.read_text()) == {MODEL: PAYLOAD["template_pin"]}
            initial = path.read_bytes()
            client.check_tokenizer(URL, "/checkpoint", MODEL, pin_file=path)
            assert path.read_bytes() == initial
        with mock_http(lambda request: httpx.Response(200, json={**PAYLOAD, "template_pin": "changed"})):
            raises(ValueError, lambda: client.check_tokenizer(URL, "/checkpoint", MODEL, pin_file=path), "Serving template changed")
            assert path.read_bytes() == initial
        with mock_http(lambda request: httpx.Response(200, json={**PAYLOAD, "model": "second"})):
            client.check_tokenizer(URL, "/checkpoint", "second", pin_file=path)
        assert json.loads(path.read_text()) == {MODEL: PAYLOAD["template_pin"], "second": PAYLOAD["template_pin"]}
        for invalid in ("not json", "[]", '{"existing": null}'):
            path.write_text(invalid)
            with mock_http(lambda request: httpx.Response(200, json=PAYLOAD)):
                raises(ValueError, lambda: client.check_tokenizer(URL, "/checkpoint", MODEL, pin_file=path))
            assert path.read_text() == invalid
        path.write_bytes(initial)
        with (mock_http(lambda request: httpx.Response(200, json={**PAYLOAD, "model": "third"})),
              patch.object(Path, "replace", side_effect=OSError("disk failure"))):
            raises(OSError, lambda: client.check_tokenizer(URL, "/checkpoint", "third", pin_file=path))
        assert path.read_bytes() == initial and list(path.parent.iterdir()) == [path]
    check("tokenizer checks fail before pinning; first-use pins persist and never silently change", True)


def test_real_http_contract():
    provider = ScriptedProvider()
    provider._config_sampling_overrides = {"temperature": 0.8}
    capture = {"effective": False}

    def set_capture(enabled):
        capture["effective"] = enabled
        return dict(capture)

    with TemporaryDirectory() as directory:
        proxy = LLMProxyServer(
            api_key="key", save_rollout_sessions=False, error_log_dir=directory,
            completion_handler=make_completion_handler(provider),
            get_sampling_overrides=provider.get_sampling_overrides,
            set_sampling_overrides=provider.set_runtime_sampling_overrides,
            get_routed_experts_config=lambda: dict(capture), set_routed_experts_config=set_capture,
            get_tokenizer_fingerprint=lambda model: {**PAYLOAD, "model": model},
            on_session_deleted=provider.release_session,
        )
        proxy.recorder.record_completion("trial", [{"role": "user", "content": "hi"}],
                                         "answer", [2], [-0.1], prompt_token_ids=[1])
        with TestClient(proxy.app) as http:
            with (patch.object(client.httpx, "get", http.get), patch.object(client.httpx, "put", http.put),
                  patch.object(client.httpx, "post", http.post), patch.object(client.httpx, "delete", http.delete),
                  patch.object(fp, "checkpoint_fingerprint", return_value=deepcopy(PAYLOAD))):
                result = client.synchronize_proxy(URL, api_key="key", sampling={"top_p": 1.0}, routed_experts=True)
                assert result["sampling_overrides"]["effective"] == {"temperature": 0.8, "top_p": 1.0}
                assert capture["effective"] is True
                client.synchronize_proxy(URL, api_key="key", routed_experts=False)
                assert capture["effective"] is False
                result = client.synchronize_proxy(URL, api_key="key", sampling={"temperature": 0})
                assert result["sampling_overrides"]["runtime"] == {"temperature": 0}
                assert result["sampling_overrides"]["effective"] == {"temperature": MIN_TEMPERATURE}
                result = client.synchronize_proxy(URL, api_key="key", sampling={})
                assert result["sampling_overrides"]["runtime"] == {}
                assert result["sampling_overrides"]["effective"] == {"temperature": 0.8}
                client.check_tokenizer(URL, "/checkpoint", MODEL, pin_file=Path(directory) / "pins.json")
                assert client.finalize_session(URL, "trial")["status"] == "finalized"
                assert client.get_session(URL, "trial", strict=True, require_logprobs=True)["session_id"] == "trial"
                assert client.get_session(URL, "trial")["completed"] is True
                client.end_session(URL, "trial")
                error = raises(httpx.HTTPStatusError, lambda: client.get_session(URL, "trial"))
                assert error.response.status_code == 404
                client.end_session(URL, "trial")
    check("helpers interoperate with real proxy collection/control/fingerprint endpoints", True)


if __name__ == "__main__":
    test_collection_identity()
    test_cleanup()
    test_cleanup_preserves_completion_state()
    test_setup_failures()
    test_tokenizer_checks()
    test_real_http_contract()
