## Aligning with sandboxes

The sandbox (a Docker / Daytona container) mints a session ID, hands the agent its task, and — once the agent signals the task is over — collects the recorded token IDs and logprobs from the proxy and ends the session.

```mermaid
flowchart LR
    subgraph AG["Agent"]
        A["Agent Harness<br/>(OpenAI SDK)"]
    end
    subgraph PR["Proxy"]
        P["TITO Proxy Server"]
    end
    subgraph SB["Sandbox"]
        S["Rollout Server<br/>(Docker / Daytona)"]
    end
    S -- "(5) Fetch record by session ID" --> P
    A -- "(2) Requests to LLM" --> P
    P -- "(3) Responses from LLM" --> A
    A -- "(4) Task ending signal" --> S
    S -- "(1) Task with session ID" --> A
    P -- "(5) Session record with token IDs and logprobs" --> S
    S -- "(6) Complete and delete session" --> P
```

(1) The sandbox starts a rollout: it generates a fresh `session_id` and gives each agent the task plus the proxy's base URL and its own composed key `{PROXY_API_KEY}{DELIMITER}{session_id}{DELIMITER}{agent_id}` — every agent owns an independent TITO token stream, and the rollout's one record tags each turn with its `agent_id`.
(2) The agent talks ordinary OpenAI `chat/completions` to the proxy; the session opens lazily on the first request.
(3) The proxy records every turn's token IDs and logprobs, then returns an OpenAI chat completion with content, tool calls when present, and token counts to the agent.
(4) The agent reports the task finished (framework-specific — e.g. a submit call into the sandbox).
(5) The sandbox fetches the full `SessionRecord`, including the recorded token IDs and logprobs, from the proxy.
(6) The sandbox ends the session: marks it completed, then deletes it so the proxy frees the in-memory record and the inference side drops the session's token stream.

## Session API used by the sandbox

| Endpoint                       | Purpose                                                                                       |
| ------------------------------ | --------------------------------------------------------------------------------------------- |
| `GET /sessions/{id}`           | Fetch the `SessionRecord` — model, per-turn token IDs / logprobs / tool calls; 404 if unknown |
| `POST /sessions/{id}/complete` | Stamp `completed: true` into the (in-memory and persisted) record                             |
| `DELETE /sessions/{id}`        | Free the in-memory record and release per-session inference state                             |

## `utils.py`

[`utils.py`](utils.py) wraps the three endpoints in two helpers:

```python
from utils import get_session, end_session

PROXY_URL = "http://proxy-host:9400"

record = get_session(PROXY_URL, session_id)   # Step (5); None if unknown
train_on(record["turns"])                     # token IDs + logprobs per turn
end_session(PROXY_URL, session_id)            # Step (6): complete, then delete
```

Things to know:

- **Fetch before ending.** `GET /sessions/{id}` serves from the proxy's in-memory record, which `end_session` deletes. Successfully written JSON snapshots under `proxyserver/sessions/` survive deletion when persistence is enabled, but contain only the fields configured for persistence; routed-expert captures are omitted by default. Failed writes are logged, so disk snapshots are not a lossless recovery guarantee.
- **Always end sessions.** Deletion releases every agent's provider state, including its strict-TITO token stream, which grows with every turn — a sandbox that never ends sessions leaks inference-side memory across rollouts.
- **Turns are delta-encoded, per agent.** Each turn stores only the messages / prompt tokens its request appended beyond the same agent's previous turn; reconstruct by concatenating turns with the same `agent_id` (see the scheme in the [main README](../README.md)).
- The complete and delete endpoints return 200 for unknown or already-ended sessions. Completing an unknown session is a no-op; deletion creates or refreshes a tombstone even for an unknown ID, rejecting chat-completion requests with 410 while the tombstone is retained. Fetching an unknown session returns 404, which `get_session()` translates to `None`. `end_session(..., delete=False)` marks an existing rollout completed but keeps it fetchable in memory.
