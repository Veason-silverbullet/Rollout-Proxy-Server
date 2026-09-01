# Testing

The tests in this directory are **end-to-end**, not mocks of the proxy: they run the real proxy, real HTTP/WebSocket/Ray transports, and a real inference engine. Only the boundaries a training framework would provide are faked.

## Engine endpoints

`test/test_engines.yaml` (gitignored — it holds API keys) lists the engines to test against, one section per engine (these are **test fixtures**, not proxy configuration — in production only the training framework's load balancer knows the engine endpoints):

```yaml
vllm:
  OPENAI_BASE_URL: [http://<vllm-host>:<port>/v1]
  OPENAI_API_KEY: [<key>]              # a single key is broadcast to all
  MODEL_NAME: Qwen3.5-35B-A3B
sglang:
  OPENAI_BASE_URL: [http://<sglang-host>:<port>/v1]
  OPENAI_API_KEY: [<key>]
  MODEL_NAME: Qwen3.5-35B-A3B
```

`OPENAI_BASE_URL` / `OPENAI_API_KEY` are lists, one entry per running engine instance. `$INFERENCE_ENGINE` (default `vllm`) picks the section **and** the proxy's adapter, so a run always tests one engine against its own adapter; the keyed-routing delimiter comes from the proxy's own `configs/{engine}-{transport}.yaml` (transport from `$TRANSPORT_MODE`, default `verl`). `$OPENAI_BASE_URL` / `$OPENAI_API_KEY` / `$MODEL_NAME` override the fixtures when set (collapsing the endpoint list to that single endpoint).

Both engines should serve the same model. The engines only ever receive token-ID prompts and return sampled token IDs (vLLM via `return_tokens_as_token_ids`, SGLang via its native `/generate`). The tests tokenize locally with the bundled tokenizer that `proxyserver/tokenization/mapping.json` maps the fixture's `MODEL_NAME` to — the same resolution the proxy performs in production — so `transformers` must be installed and the model name must have a mapping entry.

## Running

From the repository root, the whole ladder in one command (offline suites, a
live-endpoint preflight, the contract test, then all seven e2e runs — live
tests get one automatic retry against engine transients; per-test logs land
in `test/proxy/test_all-{timestamp}/`):

```bash
bash test/test_all.sh             # everything
bash test/test_all.sh --offline   # no live endpoints needed
```

Or individually:

```bash
python test/test-engines.py               # offline: engine routing + retry re-serve, no endpoint needed
python test/test-sglang-transport.py      # offline: the slime transport wire — output_ids, fallbacks, clamp
python test/test-token-stream.py          # offline: strict-TITO manager vs every tokenizer return shape
python test/test-profiles.py              # offline: per-model profiles — markers, parser, reasoning format
python test/test-recorder.py              # offline: recorder deltas, dedup, routed-experts blob, tombstones
python test/test-routed-experts.py        # offline: R3 capture — int32→uint8 repack, refusals, layers
python test/test-relay-recovery.py        # offline e2e: lost-response recovery in relay and local mode
python test/test-multi-agent.py           # offline e2e: multi-agent rollouts, one TITO stream per agent
python test/test-tool-parser.py           # offline: built-in tool-call parsers + mapping.json entries
python test/test-sampling-overrides.py    # offline e2e: override layers, precedence, GET/PUT auth
python test/test-tokenizer-fingerprint.py # offline: tokenizer-identity fingerprint + bundle pins
python test/test-contract.py              # do the live engines match what engines.py assumes?

export PROXY_API_KEY="test-proxy-real-key"          # optional, has a default
python test/test-relay.py                           # relay mode, mock worker, vLLM
python test/test-worker.py                          # relay mode, real verl worker, vLLM
python test/test-direct.py                          # local mode (direct), vLLM
INFERENCE_ENGINE=sglang python test/test-relay.py   # relay mode, mock worker, SGLang
INFERENCE_ENGINE=sglang python test/test-worker.py  # relay mode, real verl worker, SGLang
INFERENCE_ENGINE=sglang python test/test-direct.py  # local mode (direct), SGLang
python test/test-slime.py                           # local mode (slime), always SGLang
```

## What each test covers

`test-engines.py` is the exception to the end-to-end rule: it stubs the load balancer's rollout server (its `.remote()` calls answer with awaitables, like real Ray ObjectRefs) so it runs anywhere, and pins the behavior that differs between backends — `stop_reason` normalization, tool calls surviving `finish_reason`, aborted/empty completions being refused before they reach the token stream — plus the lost-response recovery: an exact retry of the previous request is re-served the cached turn without a second engine call, while edited repeats stay rejected.

`test-sglang-transport.py` (offline) pins the slime transport's wire behavior against a scripted engine: sampled token ids read from `output_ids` with the `meta_info.output_token_logprobs`-triple fallback for tokenizer-initialized engines, a missing `output_ids` key tolerated but a genuinely misaligned logprob payload refused rather than reconciled, truncation surfacing as `finish_reason: "length"`, the per-stream `X-SMG-Routing-Key` header, `weight_version` carried through `engine_meta`, and the context-window clamp — including against the sgl-router's own `RouterManager` stub payload and a window pinned from the YAML config.

`test-token-stream.py` (offline) pins `TokenStreamManager` against every tokenizer return shape — `transformers` >= 5 returns a `BatchEncoding` from `apply_chat_template`, and treating it as a sequence silently replaces the token stream with its dict keys — plus new-conversation replacement, cached-turn re-serve, the missing-echo refusal, the replay cap, and LRU-eviction messaging; two tests compare built streams against the real bundled Qwen3.5 chat template (skipped without `transformers`).

`test-profiles.py` (offline) pins the per-model profiles (`tokenization/{qwen3_5,llama3_1}.py`): the stated strict-TITO chat markers, tool-call parser, and reasoning format per family, and — the load-bearing check — that a stream built from the profile's markers matches the template's own render of the same conversation.

`test-recorder.py` (offline) pins the recorder's delta storage and its duplicate-delivery dedup: a re-served retry and a late orphan delivery can both hand `record_completion` the same turn, and it must land in the `SessionRecord` exactly once. Both are per agent — a multi-agent session's interleaved turns keep independent delta baselines, and the dedup compares against the same agent's last turn, not the session's overall last. It also pins the routed-experts blob's latest-turn-wins superseding, deletion tombstones (a deleted session drops late turns instead of being silently re-created), and per-session snapshot gating.

`test-routed-experts.py` (offline) pins the R3 capture path engine-wire to record blob: `_pack_routed_experts` repacking SGLang's base64 int32 meta to the uint8 record blob with every refusal (missing payload, misaligned element count, ids over uint8), the config-gated `return_routed_experts` flag on `/generate` requests, and the provider's config/runtime capture layers behind `PUT /routed_experts`.

`test-multi-agent.py` (offline e2e) drives the real proxy stack in both operating modes with keyed api_keys of the form `{KEY}-{session_id}-{agent_id}` and pins multi-agent rollouts: the keyed api_key requires an agent_id (the real key may itself contain the delimiter), each agent owns an independent strict-TITO stream (one agent's assistant-free opener must not reset another's stream, and cross-agent history is a 400), agents of one session generate concurrently while duplicates of one agent still serialize and generate once, all turns land in the one session record tagged with their `agent_id`, and deleting the session releases every agent's stream — provider state in local mode, sticky bindings plus the `session_closed` fan-out to the worker in relay mode.

`test-relay-recovery.py` (offline) drives the real proxy stack — `LLMProxyServer` over real HTTP, with a real `InferenceWorkerClient` over WebSocket in relay mode and the injected local completion handler in local mode — with only the engine transport scripted, and pins lost-response recovery end to end: a relay timeout answers 503 but the late completion is still recorded and an exact retry is re-served without a second generation; a retry landing while its timed-out original is *still generating* parks on the provider's per-session lock and is re-served that generation instead of starting a second one; a worker disconnect mid-generation buffers the response and re-delivers it to the still-waiting caller after reconnect; a duplicate racing its original serializes on the per-session lock so the engine runs exactly once; and a local-mode client disconnect mid-turn leaves the shielded pipeline to finish detached — the turn is recorded and the exact retry re-served.

`test-tool-parser.py` (offline) pins the built-in tool-call parsers (`qwen3_coder` / `hermes` / `llama3_json`) and their `mapping.json` wiring — the incident it guards: with parsing disabled, a model's `<tool_call>` XML went back to the agent as plain text with `tool_calls: null` and the agent looped. Runs against the bundled tokenizer when `transformers` is installed.

`test-sampling-overrides.py` (offline e2e) pins the training-side sampling-override layers over the real proxy HTTP surface: config parsing (the removed `sampling_defaults` field refused loudly), the full precedence `runtime push > config overrides > request > generation_config`, `GET /sampling_overrides` open while `PUT` requires the bare real key (keyed per-session forms refused), 501s in relay mode, and a pushed policy shaping the very next completion.

`test-tokenizer-fingerprint.py` (offline) pins the tokenizer-identity fingerprint (training–rollout consistency): the cross-repo probe-corpus digest (the same constant is pinned in `harbor.slime_bridge`'s copy of the algorithm, so the two implementations cannot drift silently), algorithm determinism and sensitivity — with the chat template deliberately *outside* the strict fingerprint — the registry's per-profile caching, the `GET /tokenizer_fingerprint` surface (open, 404 naming the mapping for an unmapped model, 501 in relay mode), and the recorded pins of the bundled Qwen3.5 tokenizer. Those pins are the deployment pin: changing the bundle or the profile markers is reward-adjacent, and updating the constants here is where it gets acknowledged.

`test-contract.py` runs **no proxy at all**. It asks each live engine directly whether it still behaves the way `engines.py` assumes: token-ID prompts accepted, logprobs aligned one-to-one with sampled tokens, EOS included on a natural stop, and `stop_reason` reported in that engine's reference (verl) vocabulary (`"completed"` for vLLM, `"stop"`/`"length"` for SGLang). An engine upgrade that changes this would keep the fake-based tests green — this is what catches it.

`test-relay.py`, `test-worker.py`, `test-direct.py`, and `test-slime.py` share their agent fixtures and verification logic (`test/common.py`) and run the same driver gauntlet (`common.run_gauntlet`) against whichever engine is selected:

- **50 mock agents concurrently, 2 turns each**, via the standard OpenAI SDK with keyed routing and **no registration call**. Prompts deliberately stress token boundaries: unicode/emoji, CJK, RTL, LaTeX, JSON escapes, code, exact repetition.
- **Strict TITO verification** of every recorded session: each turn's `prompt_token_ids` must strictly extend the session's token stream, and the inter-turn delta must equal the trainer-side tokenization of just the new user message — i.e. training could reconstruct the stream byte-identically.
- **Truncation reporting**: a response cut off at `max_tokens` must reach the trainer as `finish_reason: "length"` on *both* engines — the assertion the engine adapter exists for, since vLLM cannot distinguish truncation in `stop_reason` at all.
- **Tamper rejection**: a request whose history was edited must fail with an error (400 in both modes), never be silently re-tokenized.
- **Keyed-routing auth**: wrong real key, a bare real key with no session id appended, or empty session id → 401 before any inference.
- **Cleanup propagation**: deleting a session drops the worker/provider token stream.

What differs is the topology under test:

|                | `test-relay.py`                                      | `test-worker.py`                                                                                       | `test-direct.py`                                                   | `test-slime.py`                                               |
| -------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------- |
| Proxy mode     | relay (WebSocket)                                    | relay (WebSocket)                                                                                      | local (injected completion handler)                                | local (injected completion handler)                           |
| Inference path | mock worker speaking the relay protocol → the engine | real `InferenceWorkerClient` → real `RayRolloutProvider` → fake Ray load balancer/servers → the engine | real `DirectRolloutProvider` → the engine's own HTTP API           | real `SlimeRolloutProvider` → HTTP `/generate` → the engine   |
| Faked boundary | the framework's worker process                       | the framework's load balancer + rollout server actors                                                  | nothing — direct mode's production path *is* the engine's HTTP API | slime's sgl-router (transparent — the SGLang endpoint itself) |

The relay and worker tests drive the same rollout-server stand-ins from `test/common.py`, which are faithful replicas of the reference (verl) servers: `VLLMRolloutServer` posts token-ID prompts to `/v1/completions` and collapses `finish_reason` into `"completed"`/`"aborted"` the way verl's `vllm_async_server.py` does; `SGLangRolloutServer` posts `input_ids` to `/generate` with `return_logprob`, translates `max_tokens`/`logprobs` into `max_new_tokens`/`return_logprob`, and passes `meta_info.finish_reason.type` through verbatim — including the reference behavior of emptying the output when the logprob payload does not line up with the output ids. `test-direct.py` and `test-slime.py` need no stand-in at all: direct mode's production transport is the engine's own HTTP API, and slime's sgl-router is a transparent HTTP proxy, so the endpoint itself speaks the production wire protocol (the router would only add load balancing). `test-direct.py` passes **every** endpoint listed for the engine under test, exercising the provider's round-robin placement and sticky per-session bindings when more than one instance is running.

Each test dumps every agent's `SessionRecord` to `test/proxy/{relay,worker,direct,slime}-{engine}/` for inspection, and prints a `PASS` summary on success.
