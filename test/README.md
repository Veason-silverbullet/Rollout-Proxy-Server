# Testing

The suite combines offline unit/HTTP tests with live inference integration tests. Offline tests inject scripted engine results or fake HTTP clients while exercising the production provider, token stream, recording, and server logic. Live tests call vLLM and SGLang.

```bash
python -m pip install -e '.[test]'
bash test/test_all.sh --offline
bash test/test_all.sh
```

The runner prefers `.venv/bin/python`, then `python3`; set `PYTHON` to override. Install the package and test extra into that interpreter first. Each run gets a unique `.artifacts/proxy_server/test_all-*/` directory for logs, runtime storage, and live record dumps. The runner sets `PROXY_DATA_DIR` and `PROXY_TEST_ARTIFACT_DIR` to isolate its files. Individual live scripts create a unique `.artifacts/proxy_server/live-*/` record directory when no artifact root is supplied.

`--offline` automatically sets `HF_HUB_OFFLINE=1`, preventing optional tokenizer downloads. Bundled-tokenizer checks require `transformers` and fail if a bundle cannot load. Optional external tokenizers selected through `TOKENIZER_MODEL` print a skip reason if unavailable.

## Offline coverage

| Script                          | Checks                                                                                                                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test-packaging.py`             | Installed console/module commands, all bundled assets, writable storage, path normalization, and archive resources                                                        |
| `test-configuration.py`         | Required targets, direct default, generation provider selection for both engines, CLI controls, shutdown, HTTP-only routes                                                |
| `test-shutdown.py`              | Real CLI signals, draining detached completions, and provider shutdown ordering                                                                                           |
| `test-request-validation.py`    | Rejected malformed/multimodal requests and unsupported constraints, atomic sampling updates, pinned tool schemas, parse recovery, and concurrent model pinning            |
| `test-proxy-restart.py`         | Instance-bound agent keys, partial-record rejection, sync/async clients, response echoes, control-update races, concurrent discovery, and replacement-safe cleanup        |
| `test-native-http.py`           | SGLang/vLLM HTTP contracts, token recording, retries, and cancellation                                                                                                    |
| `test-engines.py`               | Native finish reasons, shared generation validation, model pinning, replay/rejection caching, direct and SGLang generation wire formats, context clamping                 |
| `test-sglang-transport.py`      | Sampled-ID fallback, logprob alignment, routing headers, weight versions, context-window discovery/configuration                                                          |
| `test-vllm-transport.py`        | Native token endpoint, affinity/authentication, truncated sampling, NumPy routing capture, validation, clamping, HTTP recording, retries, cancellation, and deletion      |
| `test-token-stream.py`          | Strict token continuation, tokenizer return shapes, gap tokens, GLM stop boundaries, Harmony action endings/tool-result speakers, replay cap, new conversations, eviction |
| `test-profiles.py`              | Per-model markers, reasoning/tool parsers, integration-helper expectations, prompt agreement with bundled templates, gpt-oss special-token IDs                            |
| `test-recorder.py`              | Delta storage, duplicate delivery, per-agent recording, tombstones, routing blobs, disk snapshots                                                                         |
| `test-finalization.py`          | Draining queued work, detached parsing and threaded recording, recording failures, deletion, and concurrent agents                                                        |
| `test-record-validation.py`     | Raw token/logprob integrity, routing captures, optional training fields, and conversation resets                                                                          |
| `test-routed-experts.py`        | Routing capture packing and validation, request flag, runtime capture settings                                                                                            |
| `test-recovery.py`              | Client timeouts, detached completions, retry races, duplicate serialization, GLM/Harmony tool retry/recording, Harmony action stops, deletion cleanup                     |
| `test-multi-agent.py`           | Composed API keys, independent streams, concurrent agents, deletion of all streams                                                                                        |
| `test-tool-parser.py`           | Qwen/Hermes/Llama/GLM/Harmony tool formats, schema typing, reasoning cleanup, built-in parser selection, and fixed process-lifetime model mappings                        |
| `test-sampling-overrides.py`    | Override precedence, runtime policy updates, sampling provenance through API/disk, restart drift, in-flight changes, exact retries, and cache isolation                   |
| `test-collection-validation.py` | Sampling and weight checks, agent selection, legacy/direct records, malformed records, and optional validation in the sandbox collector                                   |
| `test-training.py`              | Token reconstruction, conversation segments, capture preservation, trial allocation, role/policy validation, and dependency isolation                                     |
| `test-client.py`                | Session identity, cleanup retries/failures, control synchronization, tokenizer comparison, persistent template pins, and real proxy HTTP contracts                        |
| `test-async-client.py`          | Async client ownership, concurrency, cleanup and cancellation, nonblocking setup, finalization responses, and concurrent template pins                                    |
| `test-tokenizer-fingerprint.py` | Token identity algorithm, registry caching, HTTP endpoint, bundle/template pins including Harmony action endings                                                          |
| `test-live-helpers.py`          | Endpoint and transport selection, full/smoke workloads, native finish reasons, EOS validation, and context-clamped completions with fake engines                          |

The runner also byte-compiles source, tokenization modules, and tests.

Token-stream tests also force eviction and deletion between lookup and prompt/replay processing to verify concurrent bookkeeping. Full-template comparisons for all model families live in `test-profiles.py`; shared SGLang discovery/clamp scenarios run against both providers in `test-sglang-transport.py`.

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

Each endpoint list can contain multiple instances. Supply one key per endpoint or one shared key (`EMPTY` for an unauthenticated endpoint). `$OPENAI_BASE_URL`, `$OPENAI_API_KEY`, and `$MODEL_NAME` override the selected fixtures. `$CONTEXT_LENGTH` or a fixture's `CONTEXT_LENGTH` field supplies the engine window to either router provider. `$PROXY_API_KEY` overrides the test proxy key.

Fixtures may use `{engine}-{transport}` section names, such as `sglang-sglang_generate` and `vllm-vllm_generate`, to keep router URLs separate from direct-engine URLs. The selected transport section wins; shared `sglang` / `vllm` sections are the fallback. All sections accept the same fields. The default suite and standalone generation scripts use the canonical transport names.

Set `$TRANSPORT_MODE` to run one HTTP transport (`direct`, `sglang_generate`, or `vllm_generate`). The runner and its preflight select the same endpoints; router-only runs need no direct-engine fixtures and do not run the native direct-engine contract. Generation transports infer the matching engine. `$INFERENCE_ENGINE`, when provided, must match the protocol. VeRL deployments use the corresponding generation transport and native HTTP server URL.

Without `$TRANSPORT_MODE`, the runner preserves its default direct-engine coverage plus `sglang_generate` when SGLang is selected. The runner and `test-contract.py` use both engines when no engine/endpoint environment overrides are set; otherwise they select only `$INFERENCE_ENGINE` (default `vllm`). Standalone `test-direct.py` always selects one engine (`$INFERENCE_ENGINE`, default `vllm`). Endpoint overrides select one instance, with unspecified fields taken from the first matching fixture entry; all three endpoint fields can be supplied without a fixture file. Missing endpoints fail explicitly. Standalone scripts select their named transport before loading fixtures.

```bash
python test/test-contract.py
python test/test-direct.py
INFERENCE_ENGINE=sglang python test/test-direct.py
python test/test-sglang_generate.py
python test/test-vllm_generate.py
TRANSPORT_MODE=sglang_generate bash test/test_all.sh
TRANSPORT_MODE=vllm_generate bash test/test_all.sh
```

`test-contract.py` uses independent native HTTP clients with no proxy. It verifies token-ID input/output, aligned logprobs, sampled EOS, and explicit `stop`/`length` reasons on the selected engines. Natural stops must end with an EOS ID from the bundled generation config (or the tokenizer's EOS ID when no config override exists); length-truncated responses must not end with any of those IDs.

`test-direct.py` runs the real proxy with `DirectRolloutProvider`, including endpoint placement and sticky bindings across multiple configured instances. `test-sglang_generate.py` runs `SGLangRolloutProvider` against a SGLang endpoint speaking the router's transparent HTTP protocol; it does not test the router's load-balancing policy. `test-vllm_generate.py` runs `VLLMRolloutProvider` against a vLLM `/inference/v1/generate` endpoint.

All three integration tests run the same gauntlet: 50 concurrent agents with two turns each, exact token reconstruction, Unicode/tool-boundary cases, model validation, tamper rejection, authentication, truncation, SSE response formatting, and session deletion. Records are dumped under the run's artifact directory in `{direct,sglang_generate,vllm_generate}-{engine}/` subdirectories.

Router fixtures must point to endpoints implementing the selected transport's protocol. Truncated sampling, including the bundled Qwen defaults (`top_p: 0.95`, `top_k: 20`), works without sampling masks. The proxy floors temperature to `0.01`, including zero.

The full 50-agent workload remains the default (`LIVE_TEST_MODE=full`). For a smaller smoke run, use `LIVE_TEST_MODE=smoke bash test/test_all.sh`, or set the same variable for an individual live script. Smoke mode runs eight agents covering Unicode, code, JSON, CJK, symbols, escaping, RTL, and sequence continuation; the remaining gauntlet checks still run. Error-log assertions use a temporary directory unique to each proxy run. The direct test observes outgoing generation requests to verify that both turns actually use the bound endpoint.

The full runner performs an endpoint preflight and retries each live test once to account for transient engine failures. Offline tests are never automatically retried. Passing offline tests does not establish compatibility with a deployed engine version.
