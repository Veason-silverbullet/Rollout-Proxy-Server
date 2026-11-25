"""Check installed entry points, packaged assets, archives, and writable paths."""

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver.config import load_config
from proxyserver.resources import resource_path
from proxyserver.service import LLMProxyServer, make_completion_handler
from offline_common import ScriptedProvider, check


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def test_installed_commands_and_assets():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["HF_HUB_OFFLINE"] = "1"
    with TemporaryDirectory() as directory:
        for command in (
            [sys.executable, "-m", "proxyserver"],
            [sys.executable, "-m", "proxyserver.cli"],
            [str(Path(sys.executable).parent / "rollout-proxy-server")],
        ):
            result = subprocess.run([*command, "--help"], cwd=directory, env=env,
                                    capture_output=True, text=True, check=True)
            assert "--transport-mode" in result.stdout and "--config" in result.stdout
        subprocess.run([sys.executable, "-c", '''
from pathlib import Path
from proxyserver.config import load_config
from proxyserver.model_registry import load_model_mapping, resolve_profile

for engine, mode in (("sglang", "sglang_generate"), ("vllm", "vllm_generate"),
                     ("sglang", "direct"), ("vllm", "direct")):
    cfg = load_config(inference_engine=engine, transport_mode=mode)
    expected = "direct" if mode == "direct" else f"{engine}_generate"
    assert (cfg.inference_engine, cfg.transport_mode) == (engine, expected)
    assert cfg.proxy_api_delimiter == "|"
for model in load_model_mapping():
    bundle = Path(resolve_profile(model).tokenizer_path)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        assert (bundle / name).is_file(), bundle / name
    if bundle.name.startswith("Qwen"):
        assert (bundle / "merges.txt").is_file()
'''], cwd=directory, env=env, check=True)
    check("all installed commands and bundled assets work outside the checkout", True)


def test_storage_paths():
    with TemporaryDirectory() as directory, working_directory(directory), \
            patch.dict(os.environ, {"PROXY_DATA_DIR": ""}):
        root = Path(directory).resolve()
        cfg = load_config(inference_engine="vllm", transport_mode="direct")
        assert cfg.rollout_session_dir == root / ".rollout-proxy-server/sessions"
        assert cfg.log_dir == root / ".rollout-proxy-server/logs"
        with patch.dict(os.environ, {"PROXY_DATA_DIR": "data-one"}):
            config_file = root / "custom.yaml"
            config_file.write_text(
                "inference_engine: vllm\ntransport_mode: direct\n"
                f"rollout_session_dir: traces\nlog_dir: {root / 'absolute-logs'}\n"
            )
            cfg = load_config("custom.yaml")
            assert cfg.path == config_file
            assert cfg.rollout_session_dir == root / "data-one/traces"
            assert cfg.log_dir == root / "absolute-logs"
            proxy = LLMProxyServer(completion_handler=make_completion_handler(ScriptedProvider()))
            assert proxy.session_store.run_dir.parent == root / "data-one/sessions"
            assert proxy.error_log._run_dir.parent == root / "data-one/logs"
            explicit = LLMProxyServer(
                completion_handler=make_completion_handler(ScriptedProvider()),
                session_dir="explicit-sessions", error_log_dir="explicit-logs",
            )
            assert explicit.session_store.run_dir.parent == root / "explicit-sessions"
            assert explicit.error_log._run_dir.parent == root / "explicit-logs"
            proxy.recorder.record_completion("storage", [{"role": "user", "content": "hi"}],
                                             "answer", [2], [-0.1], prompt_token_ids=[1])
            proxy.error_log.log("storage", "test")
            snapshot = proxy.session_store.run_dir / "storage.json"
            assert json.loads(snapshot.read_text())["session_id"] == "storage"
            assert (proxy.error_log._run_dir / "storage.log").is_file()
        with patch.dict(os.environ, {"PROXY_DATA_DIR": "data-two"}):
            later = load_config(inference_engine="vllm", transport_mode="direct")
            assert later.rollout_session_dir == root / "data-two/sessions"
            assert proxy.session_store.run_dir.parent == root / "data-one/sessions"
    # Exercise actual '~' expansion without leaving files in the user's home.
    with TemporaryDirectory(prefix="rps-paths-", dir=Path.home()) as directory:
        folder = Path(directory).resolve()
        abbreviated = "~/" + str(folder.relative_to(Path.home().resolve()))
        config_file = folder / "tilde.yaml"
        config_file.write_text(
            "inference_engine: vllm\ntransport_mode: direct\n"
            f"rollout_session_dir: {abbreviated}/sessions\nlog_dir: {abbreviated}/logs\n"
        )
        cfg = load_config(abbreviated + "/tilde.yaml")
        assert cfg.path == config_file and cfg.rollout_session_dir == folder / "sessions"
        assert cfg.log_dir == folder / "logs"
        proxy = LLMProxyServer(completion_handler=make_completion_handler(ScriptedProvider()),
                               session_dir=abbreviated + "/explicit", error_log_dir=abbreviated + "/errors")
        assert proxy.session_store.run_dir.parent == folder / "explicit"
        assert proxy.error_log._run_dir.parent == folder / "errors"
    check("default, environment, YAML, explicit, and tilde storage paths resolve at creation", True)


def test_archive_resources():
    with TemporaryDirectory() as directory:
        archive = Path(directory) / "resources.zip"
        with ZipFile(archive, "w") as bundle:
            bundle.writestr("proxyserver/__init__.py", "")
            bundle.writestr("proxyserver/resources.py", Path(resource_path("resources.py")).read_text())
            bundle.writestr("proxyserver/configs/probe.yaml", "transport_mode: direct\n")
            bundle.writestr("proxyserver/tokenization/probe.txt", "tokenizer asset")
        env = {**os.environ, "PYTHONPATH": str(archive)}
        subprocess.run([sys.executable, "-c", '''
from proxyserver.resources import resource_path
directory = resource_path("configs")
assert (directory / "probe.yaml").read_text() == "transport_mode: direct\\n"
assert resource_path("configs") == directory
asset = resource_path("tokenization/probe.txt")
assert asset.read_text() == "tokenizer asset" and directory.is_dir()
'''], cwd=directory, env=env, check=True)
    check("archive resources remain available for the process lifetime", True)


if __name__ == "__main__":
    test_installed_commands_and_assets()
    test_storage_paths()
    test_archive_resources()
