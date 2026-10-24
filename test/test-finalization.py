"""Deterministic recording-barrier races; no inference endpoint required."""
import asyncio
from contextlib import suppress
from pathlib import Path
import sys
import threading
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver.server import LLMProxyServer, make_completion_handler
from offline_common import ScriptedProvider


class ParsingGate(ScriptedProvider):
    def __init__(self):
        super().__init__()
        self.parsing = asyncio.Event()
        self.release = asyncio.Event()
        self.fail = False

    async def parse_completion(self, *args, **kwargs):
        self.parsing.set()
        await self.release.wait()
        if self.fail:
            raise ValueError("injected parse failure after commit")
        return await super().parse_completion(*args, **kwargs)


def stack(provider):
    return LLMProxyServer(api_key="key", key_delimiter="|", save_rollout_sessions=False,
                          completion_handler=make_completion_handler(provider),
                          on_session_deleted=provider.release_session)


async def chat(http, sid="trial", agent="agent"):
    return await http.post('/v1/chat/completions',
        headers={'Authorization': f'Bearer key|{sid}|{agent}'},
        json={'model': 'Qwen3.5-9B', 'messages': [{'role': 'user', 'content': 'hello'}]})


async def eventually(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0)


async def test_drain_queue_and_detached_parse():
    provider = ParsingGate()
    proxy = stack(provider)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(proxy.app), base_url='http://proxy') as http:
        first = asyncio.create_task(chat(http))
        await provider.parsing.wait()
        queued = asyncio.create_task(chat(http))  # accepted before sealing; exact replay
        await eventually(lambda: len(proxy._session_tasks.get('trial', ())) == 3)
        finalize = asyncio.create_task(http.post('/sessions/trial/finalize'))
        await eventually(lambda: 'trial' in proxy._finalization_tasks)
        concurrent = asyncio.create_task(http.post('/sessions/trial/finalize'))
        assert (await chat(http, agent='late')).status_code == 409
        assert not finalize.done()
        # Losing both an agent request and a finalization caller cannot cancel
        # the committed pipeline or its barrier.
        first.cancel()
        finalize.cancel()
        for task in (first, finalize):
            with suppress(asyncio.CancelledError):
                await task
        assert not proxy._finalization_tasks['trial'].done()
        provider.release.set()
        assert (await queued).status_code == 200
        assert (await concurrent).status_code == 200
        result = (await http.get('/sessions/trial')).json()
        assert result['completed'] and len(result['turns']) == 1
        assert provider.engine_calls == 1
        assert (await http.post('/sessions/trial/finalize')).json()['status'] == 'finalized'
        assert (await chat(http)).status_code == 409
        assert (await http.delete('/sessions/trial')).status_code == 200
        assert 'trial' not in proxy._finalization_tasks
        assert (await http.post('/sessions/trial/finalize')).status_code == 410
        assert (await http.post('/sessions/unknown/finalize')).status_code == 404
        assert (await http.post('/sessions/unknown/complete')).status_code == 404


async def test_recording_failure_and_deletion():
    for deleting in (False, True):
        provider = ParsingGate()
        provider.fail = True
        proxy = stack(provider)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(proxy.app), base_url='http://proxy') as http:
            turn = asyncio.create_task(chat(http))
            await provider.parsing.wait()
            finalize = asyncio.create_task(http.post('/sessions/trial/finalize'))
            await eventually(lambda: 'trial' in proxy._finalization_tasks)
            if deleting:
                assert (await http.delete('/sessions/trial')).status_code == 200
                assert (await finalize).status_code == 410
                with suppress(asyncio.CancelledError):
                    await turn
                assert (await http.get('/sessions/trial')).status_code == 404
            else:
                provider.release.set()
                assert (await turn).status_code == 500
                assert (await finalize).status_code == 500
                assert not proxy.recorder.get_session('trial').completed
                assert (await http.post('/sessions/trial/finalize')).status_code == 500
                assert (await http.get('/sessions/trial')).status_code == 500
                assert (await chat(http)).status_code == 409
                await http.delete('/sessions/trial')


async def test_drain_threaded_recording():
    provider = ScriptedProvider()
    proxy = stack(provider)
    entered, release = threading.Event(), threading.Event()
    record = proxy.recorder.record_completion

    def delayed(**kwargs):
        entered.set()
        assert release.wait(5)
        record(**kwargs)

    proxy.recorder.record_completion = delayed
    async with httpx.AsyncClient(transport=httpx.ASGITransport(proxy.app), base_url='http://proxy') as http:
        turn = asyncio.create_task(chat(http))
        await eventually(entered.is_set)
        turn.cancel()
        with suppress(asyncio.CancelledError):
            await turn
        finalize = asyncio.create_task(http.post('/sessions/trial/finalize'))
        try:
            await eventually(lambda: 'trial' in proxy._finalization_tasks)
            assert not finalize.done()
        finally:
            release.set()
        assert (await finalize).status_code == 200
        assert len((await http.get('/sessions/trial')).json()['turns']) == 1
        await http.delete('/sessions/trial')


async def test_all_agents_drain_independently():
    provider = ScriptedProvider()
    entered = set()
    gates = {"trial-one": asyncio.Event(), "trial-two": asyncio.Event()}
    parse = provider.parse_completion

    async def gated(stream, *args, **kwargs):
        entered.add(stream)
        await gates[stream].wait()
        return await parse(stream, *args, **kwargs)

    provider.parse_completion = gated
    proxy = stack(provider)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(proxy.app), base_url='http://proxy') as http:
        first = asyncio.create_task(chat(http, agent='one'))
        second = asyncio.create_task(chat(http, agent='two'))
        await eventually(lambda: len(entered) == 2)
        finalize = asyncio.create_task(http.post('/sessions/trial/finalize'))
        await eventually(lambda: 'trial' in proxy._finalization_tasks)
        gates['trial-one'].set()
        assert (await first).status_code == 200
        assert not finalize.done()
        gates['trial-two'].set()
        assert (await second).status_code == 200
        assert (await finalize).status_code == 200
        assert len((await http.get('/sessions/trial')).json()['turns']) == 2
        await http.delete('/sessions/trial')


async def test_finalization_does_not_wait_for_response_delivery():
    provider = ParsingGate()
    proxy = stack(provider)
    sending, release_send = asyncio.Event(), asyncio.Event()

    async def app(scope, receive, send):
        async def gated_send(message):
            if scope['path'] == '/v1/chat/completions' and message['type'] == 'http.response.body':
                sending.set()
                await release_send.wait()
            await send(message)
        await proxy.app(scope, receive, gated_send)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://proxy') as http:
        turn = asyncio.create_task(chat(http))
        await provider.parsing.wait()
        finalized = asyncio.create_task(http.post('/sessions/trial/finalize'))
        await eventually(lambda: 'trial' in proxy._session_work_changed)
        provider.release.set()
        try:
            await sending.wait()
            assert (await asyncio.wait_for(asyncio.shield(finalized), 2)).status_code == 200
            assert not turn.done()
            record = (await http.get('/sessions/trial')).json()
            assert record['completed'] and len(record['turns']) == 1
        finally:
            release_send.set()
            await turn
            await finalized
            await http.delete('/sessions/trial')
        assert not proxy._session_work_changed


async def main():
    for test in (test_drain_queue_and_detached_parse, test_recording_failure_and_deletion, test_drain_threaded_recording, test_all_agents_drain_independently, test_finalization_does_not_wait_for_response_delivery):
        await asyncio.wait_for(test(), 10)
        print('PASS', test.__name__)


if __name__ == '__main__':
    asyncio.run(main())
