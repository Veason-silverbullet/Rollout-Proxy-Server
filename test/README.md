# Testing

The suite combines offline unit/HTTP tests with live inference integration tests. Offline tests inject scripted engine results or fake HTTP clients while exercising the production provider, token stream, recording, and server logic. Live tests call vLLM and SGLang.

```bash
bash test/test_all.sh --offline
bash test/test_all.sh
```

The runner uses `../venv/bin/python` when present, otherwise `python3`; set `PYTHON` to override. Logs go to `test/proxy/test_all-{timestamp}/`. Individual scripts can also be run directly. `HF_HUB_OFFLINE=1` prevents optional tokenizer downloads. Bundled-tokenizer checks require `transformers` and fail if a bundle cannot load. Optional external tokenizers selected through `TOKENIZER_MODEL` print a skip reason if unavailable.

## Offline coverage

| Script                          | Checks                                                                                                                                                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test-configuration.py`         | Required handler and endpoint targets, direct default, CLI provider/control wiring, shutdown, HTTP-only routes                                                                                                                  |
| `test-verl.py`                  | VeRL actor contracts, token/logprob validation, finish reasons, weight versions, sticky stream IDs, retries, cancellation, reservation release, named actor lookup, Ray connection ownership, and local HTTP recording/deletion |
| `test-engines.py`               | Native finish reasons, shared generation validation, model pinning, replay/rejection caching, direct and Slime wire formats, context clamping                                                                                   |
| `test-sglang-transport.py`      | Sampled-ID fallback, logprob alignment, routing headers, weight versions, context-window discovery/configuration                                                                                                                |
| `test-token-stream.py`          | Strict token continuation, tokenizer return shapes, gap tokens, replay cap, new conversations, eviction                                                                                                                         |
| `test-profiles.py`              | Per-model markers, reasoning/tool parsers, integration-helper expectations, prompt agreement with bundled templates                                                                                                             |
| `test-recorder.py`              | Delta storage, duplicate delivery, per-agent recording, tombstones, routing blobs, disk snapshots                                                                                                                               |
| `test-routed-experts.py`        | Routing capture packing and validation, request flag, runtime capture settings                                                                                                                                                  |
| `test-recovery.py`              | Client timeouts, detached completions, retry races, duplicate serialization, deletion cleanup                                                                                                                                   |
| `test-multi-agent.py`           | Composed API keys, independent streams, concurrent agents, deletion of all streams                                                                                                                                              |
| `test-tool-parser.py`           | Qwen/Hermes/Llama tool formats, schema typing, reasoning cleanup, injected parsers                                                                                                                                              |
| `test-sampling-overrides.py`    | Config validation, override precedence, authenticated policy updates, effective generation params                                                                                                                               |
| `test-tokenizer-fingerprint.py` | Token identity algorithm, registry caching, HTTP endpoint, bundle/template pins                                                                                                                                                 |

The runner also byte-compiles source, tokenization modules, and tests.

`test-live-helpers.py` checks fixture/environment endpoint selection, full/smoke workload selection, native finish reasons, EOS validation, and context-clamped completions with fake engine responses. Token-stream tests also force eviction and deletion between lookup and prompt/replay processing to verify concurrent bookkeeping. Full-template comparisons for all model families live in `test-profiles.py`; shared SGLang discovery/clamp scenarios run against both providers in `test-sglang-transport.py`.

VeRL tests use awaitable actor fakes by default, so the offline suite does not require Ray or VeRL. To also check actual Ray actors, ObjectRef awaiting, named actor lookup, and serialization on a temporary CPU-only cluster:

```bash
python test/test-verl.py --ray
```

This requires Ray installed and local process/socket access. It does not launch VeRL or a GPU engine; deployed VeRL compatibility still requires a trainer-cluster integration run.

## Live engine fixtures

Create `test/test_engines.yaml` (gitignored because it contains API keys):

```yaml
vllm:
  OPENAI_BASE_URL: [http://<vllm-host>:8000/v1]
  OPENAI_API_KEY: [<key>]
  MODEL_NAME: Qwen3.5-35B-A3B
sglang:
  OPENAI_BASE_URL: [http://<sglang-host>:30000/v1]
  OPENAI_API_KEY: [<key>]
  MODEL_NAME: Qwen3.5-35B-A3B
```

Each endpoint list can contain multiple instances. Supply one key per endpoint or one shared key. `$INFERENCE_ENGINE` selects the engine (default `vllm`); `$OPENAI_BASE_URL`, `$OPENAI_API_KEY`, and `$MODEL_NAME` override the selected fixtures. Config selection uses `$TRANSPORT_MODE`, defaulting to `direct`. `$PROXY_API_KEY` overrides the test proxy key.

The full runner, its preflight, and `test-contract.py` test both engines when none of these engine/endpoint environment variables is set. Setting any of them selects only `$INFERENCE_ENGINE` (default `vllm`), so a single overridden URL is never tested as both engine protocols. Endpoint overrides select one instance, with unspecified fields taken from the first fixture entry; all three endpoint fields can be supplied without a fixture file. Missing endpoints fail explicitly. The runner includes the Slime test only when SGLang is selected; standalone `test-slime.py` always uses SGLang.

```bash
python test/test-contract.py
python test/test-direct.py
INFERENCE_ENGINE=sglang python test/test-direct.py
python test/test-slime.py
```

`test-contract.py` uses independent native HTTP clients with no proxy. It verifies token-ID input/output, aligned logprobs, sampled EOS, and explicit `stop`/`length` reasons on the selected engines. Natural stops must end with an EOS ID from the bundled generation config (or the tokenizer's EOS ID when no config override exists); length-truncated responses must not end with any of those IDs.

`test-direct.py` runs the real proxy with `DirectRolloutProvider`, including endpoint placement and sticky bindings across multiple configured instances. `test-slime.py` runs `SlimeRolloutProvider` against a SGLang endpoint speaking the router's transparent HTTP protocol; it does not test the router's load-balancing policy.

Both integration tests run the same gauntlet: 50 concurrent agents with two turns each, exact token reconstruction, Unicode/tool-boundary cases, model validation, tamper rejection, authentication, truncation, SSE response formatting, and session deletion. Records are dumped under `test/proxy/{direct,slime}-{engine}/`.

The full 50-agent workload remains the default (`LIVE_TEST_MODE=full`). For a smaller smoke run, use `LIVE_TEST_MODE=smoke bash test/test_all.sh`, or set the same variable for an individual live script. Smoke mode runs eight agents covering Unicode, code, JSON, CJK, symbols, escaping, RTL, and sequence continuation; the remaining gauntlet checks still run. Error-log assertions use a temporary directory unique to each proxy run. The direct test observes outgoing generation requests to verify that both turns actually use the bound endpoint.

The full runner performs an endpoint preflight and retries each live test once to account for transient engine failures. Offline tests are never automatically retried. Passing offline tests does not establish compatibility with a deployed engine version; run the live contract and integration tests before deploying engine changes.
