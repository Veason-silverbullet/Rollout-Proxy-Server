"""Optional synchronous and asynchronous HTTP helpers for rollout collection and setup.

These calls belong to the collector, outside the proxy's generation path.
No inference engine or tokenizer is loaded when importing this module.
"""

from __future__ import annotations
import asyncio
import json
import re
import tempfile
import threading
import time
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any
import httpx
from .protocol import PROXY_INSTANCE_HEADER, PROXY_KEY_DELIMITER, PROXY_RESTART_MESSAGE, finite, validate_instance_id

DEFAULT_TIMEOUT = 60.0
_DELETE_DELAYS = (0.5, 1.0, 2.0)
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}


def _instance_headers(instance_id: str | None) -> dict[str, str]:
    return {} if instance_id is None else {PROXY_INSTANCE_HEADER: validate_instance_id(instance_id)}


def _check_instance_response(response: httpx.Response, expected: str | None) -> None:
    if expected is not None and response.is_success and response.headers.get(PROXY_INSTANCE_HEADER) != expected:
        raise ValueError("Proxy instance changed or was not reported; recover with fresh sessions")


def _replacement_response(response: httpx.Response, expected: str | None) -> bool:
    if expected is None or response.status_code != 412:
        return False
    try:
        actual = validate_instance_id(response.headers.get(PROXY_INSTANCE_HEADER))
    except ValueError:
        return False
    return actual != expected


def _object_response(response: httpx.Response, expected_instance_id: str | None = None) -> dict[str, Any]:
    _check_instance_response(response, expected_instance_id)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError(f"{response.request.url.path}: expected a JSON object")
    return value


def _health_instance(response: httpx.Response, expected: str | None) -> str:
    actual = validate_instance_id(_object_response(response, expected).get("instance_id"))
    if expected is not None and actual != expected:
        raise ValueError(PROXY_RESTART_MESSAGE)
    _check_instance_response(response, actual)
    return actual


def get_proxy_instance(
    proxy_url: str, *, expected_instance_id: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Discover once before a rollout, or recheck a previously pinned instance.

    Pass the returned ID to every synchronous helper and agent_key(). Rechecking
    with expected_instance_id fails on replacement instead of adopting it.
    """
    response = httpx.get(
        proxy_url.rstrip("/") + "/health", timeout=timeout,
        headers=_instance_headers(expected_instance_id),
    )
    return _health_instance(response, expected_instance_id)


def agent_key(
    api_key: str, session_id: str, agent_id: str = "agent", *,
    key_delimiter: str = PROXY_KEY_DELIMITER, instance_id: str | None = None,
) -> str:
    """Compose a routed agent key, optionally bound to a proxy instance."""
    if not key_delimiter:
        raise ValueError("key_delimiter must not be empty")
    for value in (session_id, agent_id):
        if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._]{1,256}", value)
                or key_delimiter in value):
            raise ValueError("Invalid session or agent ID for the configured delimiter")
    prefix = api_key
    if instance_id is not None:
        validate_instance_id(instance_id)
        if key_delimiter in instance_id:
            raise ValueError("Proxy instance ID contains the key delimiter; use the instance header instead")
        prefix += key_delimiter + instance_id
    return key_delimiter.join((prefix, session_id, agent_id))


def get_session(
    proxy_url: str,
    session_id: str,
    timeout: float = DEFAULT_TIMEOUT,
    *,
    expected_sampling: Mapping[str, float | int] | None = None,
    expected_weight_version: str | None = None,
    trained_agent_ids: Collection[str | None] | None = None,
    strict: bool = False,
    require_logprobs: bool = False,
    require_routed_experts: bool = False,
    expected_instance_id: str | None = None,
) -> dict[str, Any]:
    """Collect a matching session record; unknown IDs raise HTTPStatusError.

    Optional policy checks raise ValueError before returning mismatched samples.
    Leave the weight expectation unset for unversioned direct engines. Call
    finalize_session first for a stable record, then collect before end_session
    releases it. Without finalization this returns a live snapshot.
    """
    response = httpx.get(
        f"{proxy_url.rstrip('/')}/sessions/{session_id}", timeout=timeout,
        headers=_instance_headers(expected_instance_id),
    )
    record = _object_response(response, expected_instance_id)
    if record.get("session_id") != session_id:
        raise ValueError(f"Proxy returned a record for another session (expected {session_id!r})")
    if (expected_sampling is not None or expected_weight_version is not None
            or strict or require_logprobs or require_routed_experts):
        from .validation import validate_session_record

        validate_session_record(
            record, expected_sampling=expected_sampling,
            expected_weight_version=expected_weight_version,
            trained_agent_ids=trained_agent_ids,
            strict=strict, require_logprobs=require_logprobs,
            require_routed_experts=require_routed_experts,
        )
    return record


def finalize_session(
    proxy_url: str, session_id: str, timeout: float = DEFAULT_TIMEOUT, *,
    expected_instance_id: str | None = None,
) -> dict[str, Any]:
    """Seal and drain a session before collection.

    A timeout leaves finalization running. Retry this operation or delete the
    session to abandon it. A successful response does not delete the record.
    """
    result = _object_response(httpx.post(
        f"{proxy_url.rstrip('/')}/sessions/{session_id}/finalize", timeout=timeout,
        headers=_instance_headers(expected_instance_id),
    ), expected_instance_id)
    _check_finalization(result, session_id)
    return result


def _check_finalization(result, session_id):
    if result.get("session_id") != session_id or result.get("status") != "finalized":
        raise ValueError("Proxy did not finalize the requested session")


def end_session(
    proxy_url: str,
    session_id: str,
    *,
    expected_instance_id: str | None = None,
) -> None:
    """Delete or abandon a session without marking it completed.

    Finalize explicitly before collection, then delete after fetching the record.
    DELETE's 404 is success; transport errors and transient statuses
    (408/429/500/502/503/504) get up to four total attempts, with 0.5/1/2-second
    delays. A terminal deletion failure is raised. A fixed 10-second timeout
    applies to each request, not the entire operation. A confirmed replacement
    counts as cleanup: the old instance's sessions are gone, and the replacement's
    state is left untouched. It does not validate the rollout; recheck the pinned
    instance before accepting a batch.
    """
    url = f"{proxy_url.rstrip('/')}/sessions/{session_id}"
    headers = _instance_headers(expected_instance_id)
    for attempt in range(len(_DELETE_DELAYS) + 1):
        try:
            response = httpx.delete(url, timeout=10, headers=headers)
            _check_instance_response(response, expected_instance_id)
            if _replacement_response(response, expected_instance_id):
                return
            if response.status_code == 404:
                return
            response.raise_for_status()
            return
        except httpx.HTTPStatusError as error:
            if error.response.status_code not in _TRANSIENT_STATUS or attempt == len(_DELETE_DELAYS):
                raise
        except httpx.TransportError:
            if attempt == len(_DELETE_DELAYS):
                raise
        time.sleep(_DELETE_DELAYS[attempt])


def _matches_sampling(actual: Any, expected: dict[str, Any]) -> bool:
    return isinstance(actual, dict) and all(
        finite(actual.get(key)) and actual[key] == value
        for key, value in expected.items()
    )


def _validate_controls(sampling, routed_experts):
    """Normalize controls before either client sends an update."""
    if sampling is None and routed_experts is None:
        raise ValueError("Request sampling and/or routed_experts settings")
    if routed_experts is not None and type(routed_experts) is not bool:
        raise ValueError("routed_experts must be a bool or None")
    if sampling is not None:
        from .model_registry import validate_sampling_overrides

        if not isinstance(sampling, Mapping):
            raise ValueError("sampling must be a mapping")
        sampling = validate_sampling_overrides(dict(sampling), "sampling")
        if any(type(value) not in (int, float) for value in sampling.values()):
            raise ValueError("sampling values must be finite numbers")
    return sampling


def _check_sampling_response(payload, sampling):
    from .model_registry import apply_temperature_floor

    runtime = payload.get("runtime")
    if (not _matches_sampling(runtime, sampling) or set(runtime) != set(sampling)
            or not _matches_sampling(payload.get("effective"), apply_temperature_floor(sampling))):
        raise ValueError("Proxy did not apply the requested sampling distribution")


def synchronize_proxy(
    proxy_url: str,
    *,
    api_key: str,
    sampling: Mapping[str, float | int] | None = None,
    routed_experts: bool | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    expected_instance_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Apply requested controls and verify the returned effective settings.

    None skips a control, so direct engines need not support routed experts.
    Sampling replaces the runtime override layer; {} clears it. Only supplied
    sampling keys are required in the effective policy, which may also contain
    config defaults. Supply all distribution keys to pin a complete policy.

    Updates are sequential, not atomic: a later failure does not roll back an
    earlier successful update. Do not start collection after a failed call;
    correct the problem and synchronize again. Returned values are the full
    endpoint responses, keyed by "sampling_overrides" and/or "routed_experts".
    """
    sampling = _validate_controls(sampling, routed_experts)
    base = proxy_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", **_instance_headers(expected_instance_id)}
    result = {}
    if sampling is not None:
        payload = _object_response(httpx.put(
            base + "/sampling_overrides", json=sampling, headers=headers, timeout=timeout,
        ), expected_instance_id)
        _check_sampling_response(payload, sampling)
        result["sampling_overrides"] = payload
    if routed_experts is not None:
        payload = _object_response(httpx.put(
            base + "/routed_experts", json={"enabled": routed_experts}, headers=headers, timeout=timeout,
        ), expected_instance_id)
        if payload.get("effective") is not routed_experts:
            raise ValueError("Proxy did not apply the requested routed-experts setting")
        result["routed_experts"] = payload
    return result


def _check_template_pin(path: Path, model: str, pin: str) -> None:
    """Atomically save first-use pins. The caller owns this file's writes."""
    pins = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(pins, dict) or any(not isinstance(v, str) or not v for v in pins.values()):
        raise ValueError("Template pin file must map model names to nonempty pins")
    if model in pins:
        if pins[model] != pin:
            raise ValueError(f"Serving template changed for {model!r}; review and update its pin explicitly")
        return
    pins[model] = pin
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(pins, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def check_tokenizer(
    proxy_url: str,
    checkpoint: str | Path,
    model: str,
    *,
    pin_file: str | Path | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    expected_instance_id: str | None = None,
) -> dict[str, Any]:
    """Compare checkpoint/proxy token identity and optionally pin the template.

    The checkpoint may be a local path or a Hugging Face identifier. Tokenizer
    loading/hashing is lazy and cached by checkpoint path for this process.
    The serving template intentionally differs from release templates; compare
    its pin against prior serving pins, not the checkpoint's template hash.

    Without pin_file, this function writes nothing. With it, first use records
    the model's serving pin; subsequent changes raise ValueError without
    overwriting it. The JSON file maps model names to pins. Use one collector
    as the file's writer (or serialize callers externally). Call before rollout
    collection; no checks are inserted into generation requests.
    """
    from .tokenization.fingerprint import checkpoint_fingerprint

    headers = _instance_headers(expected_instance_id)
    local = checkpoint_fingerprint(str(Path(checkpoint).expanduser()))
    remote = _object_response(httpx.get(
        proxy_url.rstrip("/") + "/tokenizer_fingerprint",
        params={"model": model}, timeout=timeout, headers=headers,
    ), expected_instance_id)
    pin = _check_tokenizer_response(local, remote, model)
    if pin_file is not None:
        _check_template_pin(Path(pin_file).expanduser(), model, pin)
    return remote


class AsyncProxyClient:
    """Collector HTTP client owned by one event loop.

    Use ``async with`` or ``aclose()`` for transport shutdown. ``close(sid)``
    deletes a session without marking it complete. Finalize before collection.
    An injected http_client remains owned by its caller. Synchronization must
    be serialized at rollout boundaries; its two updates are not atomic.
    bind_instance() opts into restart protection for requests and agent keys.
    A binding lasts for this client's lifetime; recovery uses a new client.
    """

    def __init__(self, url: str, api_key: str = "", *, timeout: float = DEFAULT_TIMEOUT,
                 key_delimiter: str = PROXY_KEY_DELIMITER, http_client: httpx.AsyncClient | None = None,
                 instance_id: str | None = None):
        if not key_delimiter:
            raise ValueError("key_delimiter must not be empty")
        self.url, self.api_key = url.rstrip("/"), api_key
        self.timeout, self.key_delimiter = timeout, key_delimiter
        self._http = http_client
        self._owns_http = http_client is None
        self._closed = False
        self._close_task: asyncio.Task | None = None
        self._pin_lock = threading.Lock()
        self._instance_id = validate_instance_id(instance_id) if instance_id is not None else None
        self._instance_lock = asyncio.Lock()

    @property
    def instance_id(self) -> str | None:
        return self._instance_id

    async def bind_instance(self, instance_id: str | None = None) -> str:
        """Pin discovery once, or verify an explicitly allocated instance.

        Bind before starting concurrent rollout work. Subsequent calls recheck
        the original instance, including at batch boundaries; they never adopt
        a replacement. An explicit allocation stays pinned even if verification
        fails, so retrying cannot silently switch instances.
        """
        async with self._instance_lock:
            if instance_id is not None:
                validate_instance_id(instance_id)
                if self._instance_id is not None and instance_id != self._instance_id:
                    raise ValueError("Client is already bound to another proxy instance; create a new client for recovery")
                self._instance_id = instance_id
            response = await self._response("GET", "/health")
            actual = _health_instance(response, self._instance_id)
            self._instance_id = actual
            return actual

    async def __aenter__(self):
        if self._closed:
            raise RuntimeError("Proxy client is closed")
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self):
        self._closed = True
        if self._owns_http and self._http is not None:
            if self._close_task is None:
                self._close_task = asyncio.create_task(self._http.aclose())
                self._close_task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            # Transport cleanup survives cancellation; later callers can join it.
            await asyncio.shield(self._close_task)

    def agent_key(self, session_id: str, agent_id: str = "agent") -> str:
        return agent_key(self.api_key, session_id, agent_id,
                         key_delimiter=self.key_delimiter, instance_id=self._instance_id)

    async def _response(self, method: str, path: str, **kwargs):
        if self._closed:
            raise RuntimeError("Proxy client is closed")
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout)
        kwargs.setdefault("timeout", self.timeout)
        expected = self._instance_id
        response = await self._http.request(
            method, self.url + path,
            headers={"Authorization": f"Bearer {self.api_key}", **_instance_headers(expected)}, **kwargs,
        )
        _check_instance_response(response, expected)
        return response

    async def request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        return _object_response(await self._response(method, path, **kwargs))

    async def collect(
        self, session_id: str, *,
        expected_sampling: Mapping[str, float | int] | None = None,
        expected_weight_version: str | None = None,
        trained_agent_ids: Collection[str | None] | None = None,
        strict: bool = False,
        require_logprobs: bool = False,
        require_routed_experts: bool = False,
    ) -> dict[str, Any]:
        """GET a snapshot; unknown IDs raise HTTPStatusError. Finalize first for stability."""
        response = await self._response("GET", f"/sessions/{session_id}")
        record = _object_response(response)
        if record.get("session_id") != session_id:
            raise ValueError("Proxy returned another session's recording")
        if (expected_sampling is not None or expected_weight_version is not None
                or strict or require_logprobs or require_routed_experts):
            from .validation import validate_session_record
            validate_session_record(
                record, expected_sampling=expected_sampling,
                expected_weight_version=expected_weight_version,
                trained_agent_ids=trained_agent_ids, strict=strict,
                require_logprobs=require_logprobs,
                require_routed_experts=require_routed_experts,
            )
        return record

    async def finalize(self, session_id: str) -> dict[str, Any]:
        result = await self.request("POST", f"/sessions/{session_id}/finalize")
        _check_finalization(result, session_id)
        return result

    async def close(self, session_id: str) -> None:
        """Delete with transient retries and fixed 10s request timeouts.

        Finalize explicitly before collection; cleanup does not mark success.
        A confirmed replacement leaves its state untouched and counts as cleanup
        of the old session. Recheck bind_instance() before accepting a batch.
        """
        base = f"/sessions/{session_id}"
        for attempt in range(len(_DELETE_DELAYS) + 1):
            try:
                response = await self._response("DELETE", base, timeout=10)
                if _replacement_response(response, self._instance_id):
                    return
                if response.status_code != 404:
                    response.raise_for_status()
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _TRANSIENT_STATUS or attempt == len(_DELETE_DELAYS):
                    raise
            except httpx.TransportError:
                if attempt == len(_DELETE_DELAYS):
                    raise
            await asyncio.sleep(_DELETE_DELAYS[attempt])

    async def synchronize(self, *, sampling: Mapping[str, float | int] | None = None,
                          routed_experts: bool | None = None) -> dict[str, dict[str, Any]]:
        sampling = _validate_controls(sampling, routed_experts)
        result = {}
        if sampling is not None:
            payload = await self.request("PUT", "/sampling_overrides", json=sampling)
            _check_sampling_response(payload, sampling)
            result["sampling_overrides"] = payload
        if routed_experts is not None:
            payload = await self.request("PUT", "/routed_experts", json={"enabled": routed_experts})
            if payload.get("effective") is not routed_experts:
                raise ValueError("Proxy did not apply the requested routed-experts setting")
            result["routed_experts"] = payload
        return result

    def _pin_template(self, path: Path, model: str, pin: str) -> None:
        # One client can run concurrent checks. Keep read/check/replace atomic
        # within this writer, including when a waiting caller is cancelled.
        with self._pin_lock:
            _check_template_pin(path, model, pin)

    async def check_tokenizer(self, checkpoint: str | Path, model: str, *,
                              pin_file: str | Path | None = None) -> dict[str, Any]:
        from .tokenization.fingerprint import checkpoint_fingerprint
        local = await asyncio.to_thread(checkpoint_fingerprint, str(Path(checkpoint).expanduser()))
        remote = await self.request("GET", "/tokenizer_fingerprint", params={"model": model})
        pin = _check_tokenizer_response(local, remote, model)
        if pin_file is not None:
            await asyncio.to_thread(self._pin_template, Path(pin_file).expanduser(), model, pin)
        return remote


def _check_tokenizer_response(local, remote, model):
    from .tokenization.fingerprint import FINGERPRINT_ALGORITHM, probe_corpus_digest
    if remote.get("model") != model:
        raise ValueError("Proxy returned a tokenizer fingerprint for another model")
    if (remote.get("algorithm") != FINGERPRINT_ALGORITHM
            or remote.get("corpus_sha256") != probe_corpus_digest()):
        raise ValueError("Incompatible tokenizer fingerprint protocol")
    if remote.get("fingerprint") != local["fingerprint"]:
        raise ValueError("Checkpoint and proxy tokenizers do not match")
    pin = remote.get("template_pin")
    if not isinstance(pin, str) or not pin:
        raise ValueError("Proxy did not report its serving-template pin")
    return pin
