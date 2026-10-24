"""Real CLI signals must drain detached completions before closing the provider."""

from __future__ import annotations
import asyncio
import json
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proxyserver import cli, rollout_provider
from proxyserver.config import ProxyConfig
from offline_common import ScriptedProvider, check


def run_child(root: Path, detached: bool) -> None:
    instances = []
    original_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    class Provider(ScriptedProvider):
        def __init__(self, base_urls, **kwargs):
            super().__init__()
            self.base_urls = base_urls
            self.started = asyncio.Event()
            instances.append(self)

        async def _call_engine(self, prompt_ids, sampling_params, session_id):
            self.started.set()
            while not (root / "release").exists():
                await asyncio.sleep(0.01)
            return await super()._call_engine(prompt_ids, sampling_params, session_id)

        async def aclose(self):
            (root / "closed").touch()

    class Server(cli.LLMProxyServer):
        async def start(self):
            url = await super().start()
            if detached:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url=url) as client:
                    request = asyncio.create_task(client.post(
                        "/v1/chat/completions", headers={"Authorization": "Bearer test|trial|agent"},
                        json={"model": "Qwen3.5-9B", "messages": [{"role": "user", "content": "Hello"}]},
                    ))
                    await asyncio.wait_for(instances[0].started.wait(), 5)
                    request.cancel()
                    await asyncio.gather(request, return_exceptions=True)
                assert len(self._orphan_tasks) == 1
                assert self.recorder.dump_session("trial")["turns"] == []
            (root / "ready").touch()
            return url

        async def stop(self):
            (root / "stopping").touch()
            await super().stop()

    rollout_provider.DirectRolloutProvider = Provider
    cli.LLMProxyServer = Server
    config = ProxyConfig(
        inference_engine="vllm", proxy_api_delimiter="|", host="127.0.0.1", port=0,
        rollout_session_dir=root / "sessions", log_dir=root / "logs", save_rollout_sessions=True,
        inference_engine_base_url=["http://unused.invalid"],
    )
    cli.run_standalone_proxy(config=config, api_key="test", log_level="ERROR")
    assert {sig: signal.getsignal(sig) for sig in original_handlers} == original_handlers
    (root / "restored").touch()


def test_shutdown_signals() -> None:
    for sig, detached in ((signal.SIGTERM, True), (signal.SIGINT, True), (signal.SIGTERM, False)):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "child.log").open("w+") as log:
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--child", str(root), str(int(detached))],
                    stdout=log, stderr=subprocess.STDOUT,
                )

                def wait_for(name):
                    deadline = time.monotonic() + 10
                    while not (root / name).exists():
                        assert process.poll() is None, f"{sig.name}: exited before {name}: {process.returncode}"
                        assert time.monotonic() < deadline, f"{sig.name}: timed out waiting for {name}"
                        time.sleep(0.01)

                try:
                    wait_for("ready")
                    process.send_signal(sig)
                    wait_for("stopping")
                    if detached:
                        # A second signal during the drain must not cancel the accepted turn.
                        process.send_signal(sig)
                        assert process.poll() is None and not (root / "closed").exists()
                        assert not list((root / "sessions").rglob("trial.json"))
                        (root / "release").touch()
                    assert process.wait(timeout=10) == 0
                    assert (root / "closed").exists() and (root / "restored").exists()
                    if detached:
                        records = list((root / "sessions").rglob("trial.json"))
                        assert len(records) == 1
                        turns = json.loads(records[0].read_text())["turns"]
                        assert len(turns) == 1 and turns[0]["completion_token_ids"] == [900, 901, 902]
                    check(f"{sig.name}, detached={detached}: drains, persists, closes, and restores signal handlers", True)
                except BaseException:
                    log.seek(0)
                    print(log.read())
                    raise
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        run_child(Path(sys.argv[2]), bool(int(sys.argv[3])))
    else:
        test_shutdown_signals()

