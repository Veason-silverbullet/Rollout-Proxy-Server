"""Compatibility imports for the optional synchronous proxy client helpers.

Install rollout-proxy-server to use these helpers. New callers can import
from proxyserver.client directly. Fetch records before deleting sessions:
the proxy's in-memory records are released by end_session().
"""

from proxyserver.client import (
    DEFAULT_TIMEOUT,
    agent_key,
    check_tokenizer,
    end_session,
    finalize_session,
    get_proxy_instance,
    get_session,
    synchronize_proxy,
)

__all__ = [
    "DEFAULT_TIMEOUT", "finalize_session", "get_session", "end_session", "synchronize_proxy", "check_tokenizer",
    "get_proxy_instance", "agent_key",
]
