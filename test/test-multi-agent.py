"""Offline multi-agent tests using a scripted in-process provider.

Checks composed API keys, independent token streams and model pins,
concurrent agents, duplicate serialization, per-agent delta recording,
and deletion of every stream in a rollout.
Run: python test/test-multi-agent.py
"""

from __future__ import annotations
import asyncio
import sys
from pathlib import Path
from typing import Any
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver.server import LLMProxyServer, make_completion_handler, stream_id  # noqa: E402
from offline_common import GatedProvider, ScriptedProvider, check  # noqa: E402

# Any model in tokenization/mapping.json; the tokenizer itself is faked.
MODEL = "Qwen3.5-9B"
# Deliberately contains the delimiter, as real keys do ("sk-...-...").
PROXY_API_KEY = "multi-agent-test-key"
DELIMITER = "-"


def agent_key(session_id: str, agent_id: str) -> str:
    """Compose the keyed api_key an agent would be provisioned with."""
    return f"{PROXY_API_KEY}{DELIMITER}{session_id}{DELIMITER}{agent_id}"


class _Harness:
    """Shared agent-side helpers over a running proxy (``self.url``)."""

    url: str = ""

    async def chat(self, http: httpx.AsyncClient, session_id: str,
                   messages: list[dict[str, Any]], agent_id: str,
                   timeout: float = 30.0) -> httpx.Response:
        return await http.post(
            f"{self.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {agent_key(session_id, agent_id)}"},
            json={"model": MODEL, "messages": messages, "max_tokens": 32},
            timeout=timeout,
        )

    async def record(self, http: httpx.AsyncClient, session_id: str) -> dict[str, Any] | None:
        resp = await http.get(f"{self.url}/sessions/{session_id}")
        return resp.json() if resp.status_code == 200 else None


class ProxyStack(_Harness):
    """One proxy (injected completion handler) on a scripted provider."""

    def __init__(self, provider=None):
        self.provider = provider if provider is not None else ScriptedProvider()
        self.proxy = LLMProxyServer(
            host="127.0.0.1", port=0, api_key=PROXY_API_KEY,
            key_delimiter=DELIMITER,
            completion_handler=make_completion_handler(self.provider),
            on_session_deleted=self.provider.release_session,
            save_rollout_sessions=False,
        )

    async def __aenter__(self) -> "ProxyStack":
        self.url = await self.proxy.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.proxy.stop()


PLANNER1 = [{"role": "user", "content": "plan the work"}]
CODER1 = [{"role": "user", "content": "write the code"}]
DEFAULT1 = [{"role": "user", "content": "solo task"}]


def planner_turn2(reply: str) -> list[dict[str, Any]]:
    return PLANNER1 + [{"role": "assistant", "content": reply},
                       {"role": "user", "content": "refine the plan"}]


def test_parse_session_key() -> None:
    print("Keyed api_keys parse session and agent")
    proxy = LLMProxyServer(api_key=PROXY_API_KEY, save_rollout_sessions=False, completion_handler=make_completion_handler(ScriptedProvider()))
    parse = proxy.parse_session_key
    check("parses session and agent",
          parse(agent_key("trial_1", "planner")) == ("trial_1", "planner"))
    check("the real key may itself contain the delimiter",
          parse(f"{PROXY_API_KEY}-s.1-a_2") == ("s.1", "a_2"))
    check("wrong real key is refused",
          parse(f"wrong-key-trial_1-planner") is None)

    simple_key = "simplekey"
    simple_parse = LLMProxyServer(api_key=simple_key, save_rollout_sessions=False, completion_handler=make_completion_handler(ScriptedProvider())).parse_session_key
    check("bare real key is refused", simple_parse(simple_key) is None)
    check("a key naming no agent_id is refused",
          simple_parse(f"{simple_key}-trial_1") is None)
    check("empty session id is refused", simple_parse(f"{simple_key}-") is None)
    check("empty agent id is refused", simple_parse(f"{simple_key}-trial_1-") is None)
    check("empty session id with an agent id is refused",
          simple_parse(f"{simple_key}--planner") is None)
    check("unsafe agent id characters are refused",
          simple_parse(f"{simple_key}-trial_1-a/b") is None)


async def test_independent_streams() -> None:
    print("\nEach agent owns an independent TITO stream")
    async with ProxyStack() as stack, httpx.AsyncClient() as http:
        sid = "rollout_multi"
        p1 = await stack.chat(http, sid, PLANNER1, agent_id="planner")
        assert p1.status_code == 200, p1.text
        reply = p1.json()["choices"][0]["message"]["content"]

        # A second agent opens its own conversation on the same session...
        c1 = await stack.chat(http, sid, CODER1, agent_id="coder")
        check("second agent's opener is served", c1.status_code == 200)

        # ...and must NOT have reset the first agent's stream: its
        # continuation still lands as a strict extension.  (On a single
        # shared stream, the coder's assistant-free opener would have
        # replaced the planner's stream and this request would 400.)
        p2 = await stack.chat(http, sid, planner_turn2(reply), agent_id="planner")
        check("first agent continues after the second agent's opener",
              p2.status_code == 200)
        check("three generations, none re-sampled", stack.provider.engine_calls == 3)

        # A third agent is just another independent stream.
        d1 = await stack.chat(http, sid, DEFAULT1, agent_id="solo")
        check("a third agent's stream is independent too", d1.status_code == 200)

        # Cross-agent history is a TITO violation, not a silent re-render:
        # the coder's stream never contained the planner's turns.
        wrong = await stack.chat(http, sid, planner_turn2(reply), agent_id="coder")
        check("one agent's history under another agent's key is rejected (400)",
              wrong.status_code == 400)

        turns = (await stack.record(http, sid))["turns"]
        check("all agents' turns land in one record, in recorder append order",
              [t["agent_id"] for t in turns] == ["planner", "coder", "planner", "solo"])
        check("continuations stay delta-encoded per agent",
              turns[0]["new_conversation"] and turns[1]["new_conversation"]
              and not turns[2]["new_conversation"]
              and [m["role"] for m in turns[2]["request_messages"]] == ["user"])

        # Deleting the session releases every agent's stream.
        models = stack.provider.models
        check("provider holds one stream per agent before deletion",
              models.has_session(stream_id(sid, "planner"))
              and models.has_session(stream_id(sid, "coder"))
              and models.has_session(stream_id(sid, "solo")))
        resp = await http.delete(f"{stack.url}/sessions/{sid}")
        check("delete answers 200", resp.status_code == 200)
        check("every agent's stream is released",
              not models.has_session(stream_id(sid, "planner"))
              and not models.has_session(stream_id(sid, "coder"))
              and not models.has_session(stream_id(sid, "solo")))
        check("model pins are gone too", stack.provider._session_models == {})
        check("the record is freed", await stack.record(http, sid) is None)


async def test_agents_generate_concurrently() -> None:
    print("\nAgents of one session generate concurrently; one agent's turns still serialize")
    provider = GatedProvider()
    provider.gate = asyncio.Event()
    async with ProxyStack(provider) as stack, httpx.AsyncClient() as http:
        sid = "rollout_parallel"
        pending = asyncio.gather(
            stack.chat(http, sid, PLANNER1, agent_id="planner"),
            stack.chat(http, sid, CODER1, agent_id="coder"),
        )
        try:
            started = [await asyncio.wait_for(provider.started.get(), 5) for _ in range(2)]
            check("both agents enter generation before either is released",
                  set(started) == {stream_id(sid, "planner"), stream_id(sid, "coder")}
                  and not pending.done())
        finally:
            provider.gate.set()
            a, b = await asyncio.wait_for(pending, 5)
        check("both agents answer 200", a.status_code == 200 and b.status_code == 200)
        check("two generations ran", stack.provider.engine_calls == 2)

        # Racing duplicates of ONE agent still serialize on its stream lock
        # and are re-served from one generation.
        reply = a.json()["choices"][0]["message"]["content"]
        provider.gate.clear()
        pending = asyncio.gather(
            stack.chat(http, sid, planner_turn2(reply), agent_id="planner"),
            stack.chat(http, sid, planner_turn2(reply), agent_id="planner"),
        )
        try:
            await asyncio.wait_for(provider.started.get(), 5)
        finally:
            provider.gate.set()
            c, d = await asyncio.wait_for(pending, 5)
        check("racing duplicates of one agent both answer 200",
              c.status_code == 200 and d.status_code == 200)
        check("...with the same completion, from a single generation",
              stack.provider.engine_calls == 3
              and c.json()["choices"][0]["message"]["content"]
              == d.json()["choices"][0]["message"]["content"])


async def main() -> None:
    test_parse_session_key()
    await test_independent_streams()
    await test_agents_generate_concurrently()
    print("\n" + "=" * 70)
    print("PASS: multi-agent rollouts — the keyed api_key's agent_id gives each\n"
          "      agent an independent strict-TITO stream (concurrent across\n"
          "      agents, serialized within one), all turns land in the one\n"
          "      session record tagged per agent, and deleting the session\n"
          "      releases every agent's stream for the whole rollout.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
