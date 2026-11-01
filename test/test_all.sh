#!/usr/bin/env bash
# Run syntax checks and offline tests, followed by live HTTP integration tests.
# bash test/test_all.sh --offline needs no inference endpoints.
# PYTHON selects the interpreter (default .venv/bin/python or python3).
# PROXY_API_KEY selects the test key; TOKENIZER_MODEL selects the token-stream
# test's real tokenizer (default: bundled Qwen3.5).
# Logs and runtime files are isolated under .artifacts/proxy_server/.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ]; then
    if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON=python3; fi
fi
export PROXY_API_KEY="${PROXY_API_KEY:-test-proxy-real-key}"
OFFLINE_ONLY=0
[ "${1:-}" = "--offline" ] && OFFLINE_ONLY=1

mkdir -p "$ROOT/.artifacts/proxy_server" || exit 1
LOG_DIR=$(mktemp -d "$ROOT/.artifacts/proxy_server/test_all-XXXXXXXX") || exit 1
export PROXY_TEST_ARTIFACT_DIR="$LOG_DIR"
export PROXY_DATA_DIR="$LOG_DIR/runtime"
if [ "$OFFLINE_ONLY" -eq 1 ]; then export HF_HUB_OFFLINE=1; fi
if ! "$PYTHON" -c 'import importlib.metadata; import openai; importlib.metadata.distribution("rollout-proxy-server")'; then
    echo "Install runtime and test dependencies: $PYTHON -m pip install -e '.[test]'"
    exit 1
fi

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
from common import live_test_endpoints  # noqa: E402

ok = True
for transport, ep in live_test_endpoints():
    try:
        r = httpx.get(f"{ep.root_url}/health",
                     headers={"Authorization": f"Bearer {ep.api_key}"}, timeout=10)
        if r.status_code >= 500:
            raise RuntimeError(f"HTTP {r.status_code}")
        print(f"  {transport}/{ep.engine}: {ep.root_url} reachable")
    except Exception as e:
        print(f"  {transport}/{ep.engine}: {ep.root_url} UNREACHABLE ({type(e).__name__}: {e})")
        ok = False
sys.exit(0 if ok else 1)
EOF
}

echo "== offline =="
run "async-client"          0 "$PYTHON" test/test-async-client.py
run "proxy-restart"         0 "$PYTHON" test/test-proxy-restart.py
run "finalization"          0 "$PYTHON" test/test-finalization.py
run "record-validation"     0 "$PYTHON" test/test-record-validation.py
run "training"              0 "$PYTHON" test/test-training.py
run "py-compile"            0 "$PYTHON" -m py_compile proxyserver/*.py proxyserver/tokenization/*.py sandbox/*.py test/*.py
run "packaging"             0 "$PYTHON" test/test-packaging.py
run "configuration"         0 "$PYTHON" test/test-configuration.py
run "shutdown"              0 "$PYTHON" test/test-shutdown.py
run "request-validation"    0 "$PYTHON" test/test-request-validation.py
run "native-http"           0 "$PYTHON" test/test-native-http.py
run "engines"               0 "$PYTHON" test/test-engines.py
run "live-helpers"          0 "$PYTHON" test/test-live-helpers.py
run "sglang-transport"      0 "$PYTHON" test/test-sglang-transport.py
run "vllm-transport"        0 "$PYTHON" test/test-vllm-transport.py
run "token-stream"          0 "$PYTHON" test/test-token-stream.py
run "profiles"              0 "$PYTHON" test/test-profiles.py
run "recorder"              0 "$PYTHON" test/test-recorder.py
run "routed-experts"        0 "$PYTHON" test/test-routed-experts.py
run "recovery"              0 "$PYTHON" test/test-recovery.py
run "multi-agent"           0 "$PYTHON" test/test-multi-agent.py
run "tool-parser"           0 "$PYTHON" test/test-tool-parser.py
run "sampling-overrides"    0 "$PYTHON" test/test-sampling-overrides.py
run "collection-validation" 0 "$PYTHON" test/test-collection-validation.py
run "client"                0 "$PYTHON" test/test-client.py
run "tokenizer-fingerprint" 0 "$PYTHON" test/test-tokenizer-fingerprint.py

if [ "$OFFLINE_ONLY" -eq 1 ]; then
    echo; echo "== summary (offline only) =="
    printf '%s\n' "${RESULTS[@]}"
    exit $FAILED
fi

echo; echo "== live endpoint preflight =="
if ! preflight; then
    echo "Engine endpoint preflight failed — check test/test_engines.yaml and environment overrides, or run with --offline."
    exit 1
fi

echo; echo "== live =="
LIVE_CASES=$("$PYTHON" - <<'EOF'
import sys
sys.path.insert(0, "test")
from common import live_test_cases
for transport, engine in live_test_cases():
    print(transport, engine)
EOF
) || exit 1
while read -r transport engine; do
    if [ "$transport" = "direct" ]; then
        run "contract-$engine" 1 env TRANSPORT_MODE="$transport" INFERENCE_ENGINE="$engine" "$PYTHON" test/test-contract.py
    fi
    run "$transport-$engine" 1 env TRANSPORT_MODE="$transport" INFERENCE_ENGINE="$engine" "$PYTHON" "test/test-$transport.py"
done <<< "$LIVE_CASES"

echo; echo "== summary =="
printf '%s\n' "${RESULTS[@]}"
if [ $FAILED -eq 0 ]; then
    echo "ALL TESTS PASS"
else
    echo "SOME TESTS FAILED (a failure that survived its retry is real — check the log)"
fi
exit $FAILED
