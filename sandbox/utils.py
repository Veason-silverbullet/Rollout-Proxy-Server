"""Sandbox-side client utilities for the proxy's session-management API.

The proxy server already exposes the endpoints the sandbox needs (see
``proxyserver/server.py``, "session management"):

    GET    /sessions/{session_id}           -> full SessionRecord JSON, 404 if unknown
    POST   /sessions/{session_id}/complete  -> stamp ``completed: true`` on the record
    DELETE /sessions/{session_id}           -> drop the in-memory record and release
                                               the session's sticky worker binding

These helpers are plain HTTP calls to those endpoints — the sandbox needs no
proxy internals, only the proxy's base URL (e.g. ``http://host:port``).

Fetch the record *before* ending a session with ``delete=True``: the GET is
served from the proxy's in-memory record, which deletion frees (the JSON
persisted under ``proxyserver/sessions/`` survives deletion).
"""

from __future__ import annotations
from typing import Any
import httpx

DEFAULT_TIMEOUT = 30.0


def get_session(
    proxy_url: str,
    session_id: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any] | None:
    """Fetch a rollout session record from the proxy by session ID.

    Returns the ``SessionRecord`` as a dict — ``model_name``, ``completed``,
    and per-turn token IDs / logprobs / tool calls (turns are delta-encoded;
    see ``proxyserver/rollout_sessions.py`` for the reconstruction scheme) —
    or ``None`` if the proxy does not know the session.
    """
    resp = httpx.get(f"{proxy_url.rstrip('/')}/sessions/{session_id}", timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def end_session(
    proxy_url: str,
    session_id: str,
    delete: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    """End a rollout session in the proxy by session ID.

    Marks the session completed, which stamps ``completed: true`` into the
    persisted record, then — unless ``delete=False`` — deletes it so the
    proxy frees the in-memory record, releases the sticky worker binding,
    and fires its ``on_session_deleted`` cleanup hook.
    """
    base = proxy_url.rstrip("/")
    resp = httpx.post(f"{base}/sessions/{session_id}/complete", timeout=timeout)
    resp.raise_for_status()
    if delete:
        resp = httpx.delete(f"{base}/sessions/{session_id}", timeout=timeout)
        resp.raise_for_status()
