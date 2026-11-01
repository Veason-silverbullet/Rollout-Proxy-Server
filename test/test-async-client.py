"""Async collector ownership, concurrency, failures, and nonblocking setup."""
import asyncio
import sys
import threading
from pathlib import Path
from unittest.mock import patch
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proxyserver import client
from proxyserver.tokenization import fingerprint as fp


async def test_lifecycle_and_concurrency():
    real = httpx.AsyncClient
    clients, entered = [], []
    release = asyncio.Event()

    async def handler(request):
        assert request.extensions['timeout']['read'] == 60
        sid = request.url.path.split('/')[2]
        entered.append(sid)
        await release.wait()
        return httpx.Response(200, json={'session_id': sid, 'turns': []})

    def factory(**kwargs):
        http = real(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(http)
        return http

    with patch.object(client.httpx, 'AsyncClient', factory):
        async with client.AsyncProxyClient('http://proxy', 'key') as proxy:
            assert proxy.agent_key('trial') == 'key|trial|agent'
            tasks = [asyncio.create_task(proxy.collect(sid, strict=True)) for sid in ('a', 'b')]
            async with asyncio.timeout(3):
                while len(entered) < 2:
                    await asyncio.sleep(0)
            assert len(clients) == 1
            release.set()
            records = await asyncio.gather(*tasks)
            assert [r['session_id'] for r in records] == ['a', 'b']
        assert clients[0].is_closed
        await proxy.aclose()
        try:
            await proxy.collect('a')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Closed client was reused')
    async with real(transport=httpx.MockTransport(handler)) as borrowed:
        async with client.AsyncProxyClient('http://proxy', http_client=borrowed) as proxy:
            await proxy.collect('a')
        assert not borrowed.is_closed
    async with client.AsyncProxyClient('http://proxy', 'key', key_delimiter='::') as proxy:
        assert proxy.agent_key('trial') == 'key::trial::agent'


async def test_failures_and_cleanup():
    calls = []
    delete_attempts = 0

    async def handler(request):
        nonlocal delete_attempts
        calls.append((request.method, request.url.path))
        cleanup = request.method == 'DELETE'
        assert request.extensions['timeout']['read'] == (10 if cleanup else 7)
        if request.method == 'DELETE':
            delete_attempts += 1
            if delete_attempts == 1:
                raise httpx.ReadTimeout('lost response', request=request)
            return httpx.Response(404)
        if request.url.path.endswith('/finalize'):
            return httpx.Response(404)
        if request.url.path.endswith('/timeout'):
            raise httpx.ReadTimeout('timeout', request=request)
        if request.url.path.endswith('/missing'):
            return httpx.Response(404)
        return httpx.Response(200, json=[])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with client.AsyncProxyClient('http://proxy', timeout=7, http_client=http) as proxy:
            for sid, kind in [('missing', httpx.HTTPStatusError), ('bad', ValueError), ('timeout', httpx.ReadTimeout)]:
                try:
                    await proxy.collect(sid)
                except kind as error:
                    if sid == 'missing':
                        assert error.response.status_code == 404
                else:
                    raise AssertionError('Accepted invalid response')
            try:
                await proxy.finalize('old_server')
            except httpx.HTTPStatusError:
                pass
            else:
                raise AssertionError('Silently accepted unsupported finalization')
            assert not any(path.endswith('/complete') for _, path in calls)
            with patch.object(client.asyncio, 'sleep', return_value=None):
                await proxy.close('trial')
            assert delete_attempts == 2
            assert not any(path.endswith('/complete') for _, path in calls)


async def test_cleanup_preserves_completion_state():
    import json
    import tempfile
    from proxyserver.service import LLMProxyServer, make_completion_handler
    from offline_common import ScriptedProvider

    with tempfile.TemporaryDirectory() as tmp:
        server = LLMProxyServer(
            completion_handler=make_completion_handler(ScriptedProvider()),
            session_dir=Path(tmp), save_rollout_sessions=True,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app)) as http:
            async with client.AsyncProxyClient('http://proxy', http_client=http) as proxy:
                for sid, finalize in (('abandoned', False), ('finished', True)):
                    server.recorder.record_completion(
                        sid, [{"role": "user", "content": "hi"}], "answer",
                        [2], [-0.1], prompt_token_ids=[1],
                    )
                    if finalize:
                        await proxy.finalize(sid)
                        assert (await proxy.collect(sid))['completed'] is True
                    await proxy.close(sid)
                    snapshot = json.loads((server.session_store.run_dir / f'{sid}.json').read_text())
                    assert snapshot['completed'] is finalize
                    assert server.recorder.is_deleted(sid)


async def test_setup_off_event_loop():
    started, release = threading.Event(), threading.Event()
    pin_threads = []
    loop_thread = threading.get_ident()

    def fingerprint(checkpoint):
        started.set()
        assert release.wait(5)
        return {'fingerprint': 'same'}

    def pin(*args):
        pin_threads.append(threading.get_ident())

    async def handler(request):
        if request.url.path == '/sampling_overrides':
            return httpx.Response(200, json={'runtime': {}, 'effective': {'top_p': 1.0}})
        return httpx.Response(200, json={'model': 'model', 'algorithm': fp.FINGERPRINT_ALGORITHM,
            'corpus_sha256': fp.probe_corpus_digest(), 'fingerprint': 'same', 'template_pin': 'pin'})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with client.AsyncProxyClient('http://proxy', http_client=http) as proxy:
            with patch.object(fp, 'checkpoint_fingerprint', fingerprint), patch.object(client, '_check_template_pin', pin):
                task = asyncio.create_task(proxy.check_tokenizer('checkpoint', 'model', pin_file='unused'))
                try:
                    async with asyncio.timeout(3):
                        while not started.is_set():
                            await asyncio.sleep(0)
                    assert not task.done()
                finally:
                    release.set()
                assert (await task)['model'] == 'model'
                assert pin_threads and pin_threads[0] != loop_thread
            result = await proxy.synchronize(sampling={})
            assert result['sampling_overrides']['runtime'] == {}


async def test_cancelled_transport_shutdown_can_be_joined():
    entered, release = asyncio.Event(), asyncio.Event()
    real = httpx.AsyncClient

    class SlowClose(real):
        async def aclose(self):
            entered.set()
            await release.wait()
            await super().aclose()

    owned = SlowClose(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={'session_id': 'trial'})
    ))
    with patch.object(client.httpx, 'AsyncClient', return_value=owned):
        proxy = client.AsyncProxyClient('http://proxy')
        await proxy.collect('trial')
        closing = asyncio.create_task(proxy.aclose())
        await entered.wait()
        closing.cancel()
        try:
            await closing
        except asyncio.CancelledError:
            pass
        release.set()
        await proxy.aclose()
        assert owned.is_closed


async def test_finalization_response_contract():
    for payload in ([], {}, {'session_id': 'other', 'status': 'finalized'},
                    {'session_id': 'trial', 'status': 'completed'},
                    {'session_id': 'trial', 'status': 'finalized'}):
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload)
        )) as http:
            async with client.AsyncProxyClient('http://proxy', http_client=http) as proxy:
                if payload == {'session_id': 'trial', 'status': 'finalized'}:
                    assert await proxy.finalize('trial') == payload
                else:
                    try:
                        await proxy.finalize('trial')
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('Invalid finalization response accepted')


async def check_concurrent_pin_updates(cancel_first=False):
    from tempfile import TemporaryDirectory
    entered, release, second_writer = threading.Event(), threading.Event(), threading.Event()
    original_file = client.tempfile.NamedTemporaryFile
    original_pin = client.AsyncProxyClient._pin_template
    writers = 0

    def file(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original_file(*args, **kwargs)

    def pin(self, *args):
        nonlocal writers
        writers += 1
        if writers == 2:
            second_writer.set()
        return original_pin(self, *args)

    with TemporaryDirectory() as directory:
        path = Path(directory) / 'pins.json'
        async with client.AsyncProxyClient('http://proxy') as proxy:
            count = 0
            async def request(*args, **kwargs):
                nonlocal count
                count += 1
                return {'model': 'model', 'algorithm': fp.FINGERPRINT_ALGORITHM,
                        'corpus_sha256': fp.probe_corpus_digest(),
                        'fingerprint': 'same', 'template_pin': f'pin{count}'}
            proxy.request = request
            with patch.object(fp, 'checkpoint_fingerprint', return_value={'fingerprint': 'same'}), \
                    patch.object(client.tempfile, 'NamedTemporaryFile', file), \
                    patch.object(client.AsyncProxyClient, '_pin_template', pin):
                first = asyncio.create_task(proxy.check_tokenizer('checkpoint', 'model', pin_file=path))
                try:
                    async with asyncio.timeout(3):
                        while not entered.is_set():
                            await asyncio.sleep(0)
                    if cancel_first:
                        first.cancel()
                        try:
                            await first
                        except asyncio.CancelledError:
                            pass
                    second = asyncio.create_task(proxy.check_tokenizer('checkpoint', 'model', pin_file=path))
                    async with asyncio.timeout(3):
                        while not second_writer.is_set():
                            await asyncio.sleep(0)
                finally:
                    release.set()
                if not cancel_first:
                    await first
                try:
                    await second
                except ValueError as error:
                    assert 'template changed' in str(error)
                else:
                    raise AssertionError('Concurrent checks accepted conflicting pins')
        import json
        assert json.loads(path.read_text()) == {'model': 'pin1'}


async def test_concurrent_pin_updates():
    await check_concurrent_pin_updates()
    await check_concurrent_pin_updates(cancel_first=True)


async def test_disabled_validation_matches_sync_client():
    record = {'session_id': 'legacy'}
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=record)
    )) as http:
        async with client.AsyncProxyClient('http://proxy', http_client=http) as proxy:
            assert await proxy.collect('legacy') == record
            assert await proxy.collect('legacy', strict=False, expected_sampling=None) == record
            try:
                await proxy.collect('legacy', strict=True)
            except ValueError as error:
                assert 'turns' in str(error)
            else:
                raise AssertionError('Strict collection accepted missing turns')


async def main():
    for test in (test_lifecycle_and_concurrency, test_failures_and_cleanup, test_cleanup_preserves_completion_state, test_setup_off_event_loop, test_cancelled_transport_shutdown_can_be_joined, test_finalization_response_contract, test_concurrent_pin_updates, test_disabled_validation_matches_sync_client):
        await asyncio.wait_for(test(), 10)
        print('PASS', test.__name__)


if __name__ == '__main__':
    asyncio.run(main())
