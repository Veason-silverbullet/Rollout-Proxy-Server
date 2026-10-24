"""Lightweight wire constants shared by the proxy and its HTTP clients."""

import math
from typing import Any

PROXY_KEY_DELIMITER = "|"
PROXY_INSTANCE_HEADER = "x-rps-proxy-instance"
PROXY_RESTART_MESSAGE = "Proxy instance changed; recover with fresh sessions and synchronize again"


def validate_instance_id(value: str) -> str:
    if (not isinstance(value, str) or len(value) != 32
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError("Proxy instance ID must be a lowercase UUID hex string")
    return value


def finite(value: Any) -> bool:
    """Whether ``value`` is a finite int or float; bools and overflowing ints are not."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False
