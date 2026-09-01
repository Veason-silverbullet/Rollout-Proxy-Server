#!/usr/bin/env bash
# Run the full test ladder, cheapest to most expensive:
#
#   1. byte-compile every source and test file
#   2. test-engines.py        (offline: adapters, providers, slime + direct
#                              transports, lost-response retry re-serve)
#   3. test-sglang-transport.py (offline: the slime transport — sampled
#                              ids from output_ids with the logprob-triple
#                              fallback, misaligned payloads refused, the
#                              X-SMG-Routing-Key header, weight_version,
#                              and the context-window clamp incl. the
#                              sgl-router RouterManager stub payload)
#   4. test-token-stream.py   (offline: strict-TITO manager, every tokenizer
#                              return shape; bundled Qwen3.5 templates + the
#                              real $TOKENIZER_MODEL tokenizer, each skipped
#                              if transformers or the tokenizer is missing)
#   5. test-profiles.py       (offline: per-model profiles — stated chat
#                              markers, tool-call parser, reasoning format;
#                              built streams match the template's own render)
#   6. test-recorder.py       (offline: delta storage + duplicate-delivery
#                              dedup + routed-experts superseding and
#                              snapshot gating)
#   7. test-routed-experts.py (offline: R3 capture — int32 -> uint8 repack
#                              + refusals, the /generate request flag, the
#                              provider's config/runtime capture layers)
#   8. test-relay-recovery.py (offline e2e: real proxy + real worker client
#                              + local mode, scripted engine — relay-timeout
#                              orphan recording, retry re-serve (also racing
#                              a still-generating original), worker-disconnect
#                              buffering, local-mode disconnect shielding)
#   9. test-multi-agent.py    (offline e2e: multi-agent rollouts — keyed
#                              api_keys with agent_id, one TITO stream per
#                              agent, concurrent agents, per-agent record
#                              tagging, delete releasing every stream)
#   10. test-tool-parser.py    (offline: built-in qwen3_coder / hermes /
#                              llama3_json tool-call parsers +
#                              tokenization/mapping.json entries; bundled
#                              tokenizer when transformers is installed)
#   11. test-sampling-overrides.py (offline e2e: the sampling-override layers
#                              — config parsing, the five-layer precedence,
#                              GET/PUT /sampling_overrides auth + 501s, a
#                              pushed policy shaping the next completion)
#   12. test-tokenizer-fingerprint.py (offline: the tokenizer-identity
#                              fingerprint — cross-repo corpus pin, algorithm
#                              sensitivity, registry caching, GET
#                              /tokenizer_fingerprint + 404/501s, and the
#                              recorded pins of the bundled Qwen3.5 tokenizer;
#                              the bundle section skips without transformers)
#   13. test-contract.py      (live: both engines' wire contract, no proxy)
#   14. test-relay.py / test-worker.py / test-direct.py on vllm and sglang,
#      then test-slime.py     (live e2e: 50 agents x 2 turns each)
#
# Live tests get one automatic retry: the shared engine endpoints flake
# under load (keep-alive drops, >900s generation stalls, transient 400
# bursts — see test/README.md).  A failure that survives the retry is real.
#
# Usage:
#   bash test/test_all.sh              # everything (needs live endpoints)
#   bash test/test_all.sh --offline    # stages 1-12 only, no endpoints needed
#
# Environment:
#   PYTHON           interpreter to use (default: ../venv/bin/python next to
#                    the repo if present, else python3)
#   PROXY_API_KEY    shared secret for keyed routing (default: test key)
#   TOKENIZER_MODEL  tokenizer handed to test-token-stream (default:
#                    Qwen/Qwen3-0.6B).  Set it in the environment and
#                    test-tool-parser inherits it too; left unset, that test
#                    uses the bundled proxyserver/tokenization/Qwen3.5.
#
# Logs: one file per test under test/proxy/test_all-{timestamp}/ (gitignored).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ]; then
    if [ -x "$ROOT/../venv/bin/python" ]; then PYTHON="$ROOT/../venv/bin/python"; else PYTHON=python3; fi
fi
export PROXY_API_KEY="${PROXY_API_KEY:-test-proxy-real-key}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-0.6B}"
OFFLINE_ONLY=0
[ "${1:-}" = "--offline" ] && OFFLINE_ONLY=1

LOG_DIR="test/proxy/test_all-$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "$LOG_DIR"

RESULTS=()
FAILED=0

run() {
    # run <label> <retries> <command...>
    local label="$1" retries="$2"; shift 2
    local log="$LOG_DIR/${label}.log" attempt=0 t0 dt
    while :; do
        attempt=$((attempt + 1))
        t0=$SECONDS
        if "$@" >"$log" 2>&1; then
            dt=$((SECONDS - t0))
            local mark="PASS"; [ $attempt -gt 1 ] && mark="PASS (retry)"
            printf '%-14s %-22s %4ss\n' "$mark" "$label" "$dt"
            RESULTS+=("$(printf '%-14s %-22s %4ss' "$mark" "$label" "$dt")")
            return 0
        fi
        dt=$((SECONDS - t0))
        if [ $attempt -le $retries ]; then
            printf '%-14s %-22s %4ss  retrying (engine transients are common)\n' "FLAKE?" "$label" "$dt"
            continue
        fi
        printf '%-14s %-22s %4ss  log: %s\n' "FAIL" "$label" "$dt" "$log"
        echo "---- last lines of $log ----"
        tail -n 12 "$log"
        echo "-----------------------------"
        RESULTS+=("$(printf '%-14s %-22s %4ss  log: %s' "FAIL" "$label" "$dt" "$log")")
        FAILED=1
        return 1
    done
}

preflight() {
    # Fail in seconds, not after 50 agents hang for minutes each.
    "$PYTHON" - <<'EOF'
import sys
sys.path.insert(0, "test")
import httpx
from common import load_engine_endpoints  # noqa: E402  (reads test/test_engines.yaml)

ok = True
for engine in ("vllm", "sglang"):
    for ep in load_engine_endpoints(engine):
        try:
            r = httpx.get(f"{ep.root_url}/health",
                          headers={"Authorization": f"Bearer {ep.api_key}"}, timeout=10)
            if r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}")
            print(f"  {engine}: {ep.root_url} reachable")
        except Exception as e:
            print(f"  {engine}: {ep.root_url} UNREACHABLE ({type(e).__name__}: {e})")
            ok = False
sys.exit(0 if ok else 1)
EOF
}

echo "== offline =="
run "py-compile"            0 "$PYTHON" -m py_compile proxyserver/*.py test/*.py
run "engines"               0 "$PYTHON" test/test-engines.py
run "sglang-transport"      0 "$PYTHON" test/test-sglang-transport.py
run "token-stream"          0 env TOKENIZER_MODEL="$TOKENIZER_MODEL" "$PYTHON" test/test-token-stream.py
run "profiles"              0 "$PYTHON" test/test-profiles.py
run "recorder"              0 "$PYTHON" test/test-recorder.py
run "routed-experts"        0 "$PYTHON" test/test-routed-experts.py
run "relay-recovery"        0 "$PYTHON" test/test-relay-recovery.py
run "multi-agent"           0 "$PYTHON" test/test-multi-agent.py
run "tool-parser"           0 "$PYTHON" test/test-tool-parser.py
run "sampling-overrides"    0 "$PYTHON" test/test-sampling-overrides.py
run "tokenizer-fingerprint" 0 "$PYTHON" test/test-tokenizer-fingerprint.py

if [ "$OFFLINE_ONLY" -eq 1 ]; then
    echo; echo "== summary (offline only) =="
    printf '%s\n' "${RESULTS[@]}"
    exit $FAILED
fi

echo; echo "== live endpoint preflight =="
if ! preflight; then
    echo "Engine endpoints unreachable — fix test/test_engines.yaml or run with --offline."
    exit 1
fi

echo; echo "== live =="
run "contract"      1 "$PYTHON" test/test-contract.py
run "relay-vllm"    1 env INFERENCE_ENGINE=vllm   "$PYTHON" test/test-relay.py
run "worker-vllm"   1 env INFERENCE_ENGINE=vllm   "$PYTHON" test/test-worker.py
run "direct-vllm"   1 env INFERENCE_ENGINE=vllm   "$PYTHON" test/test-direct.py
run "relay-sglang"  1 env INFERENCE_ENGINE=sglang "$PYTHON" test/test-relay.py
run "worker-sglang" 1 env INFERENCE_ENGINE=sglang "$PYTHON" test/test-worker.py
run "direct-sglang" 1 env INFERENCE_ENGINE=sglang "$PYTHON" test/test-direct.py
run "slime"         1 "$PYTHON" test/test-slime.py

echo; echo "== summary =="
printf '%s\n' "${RESULTS[@]}"
if [ $FAILED -eq 0 ]; then
    echo "ALL TESTS PASS"
else
    echo "SOME TESTS FAILED (a failure that survived its retry is real — check the log)"
fi
exit $FAILED
