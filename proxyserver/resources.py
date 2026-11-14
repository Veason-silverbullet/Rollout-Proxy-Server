"""Locate bundled assets and writable data independently of installation."""

from __future__ import annotations
import atexit
import os
from contextlib import ExitStack
from functools import cache
from importlib.resources import as_file, files
from pathlib import Path


_resource_contexts = ExitStack()
atexit.register(_resource_contexts.close)


@cache
def resource_path(name: str) -> Path:
    """Keep a bundled file or directory available for the process lifetime.

    Wheel and editable installs reuse existing paths. Archive imports extract
    files/directories for the process lifetime, since tokenizers and the model
    mapping require filesystem paths. Python 3.12 supports directory resources.
    """
    return _resource_contexts.enter_context(as_file(files(__package__).joinpath(name)))


def storage_path(value: str | Path) -> Path:
    """Resolve YAML storage paths under ``PROXY_DATA_DIR``.

    The default root is ``.rollout-proxy-server`` in the working directory.
    Absolute paths bypass the root. Resolve when configuration or server
    instances are created so later environment changes apply to new instances.
    Explicit server constructor paths instead resolve directly against cwd.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        root = Path(os.environ.get("PROXY_DATA_DIR") or ".rollout-proxy-server").expanduser()
        path = root / path
    return path.resolve()
