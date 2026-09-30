#!/usr/bin/env bash
# Run syntax checks and offline tests, followed by live HTTP integration tests.
# bash test/test_all.sh --offline needs no inference endpoints.
# PYTHON selects the interpreter (default ../venv/bin/python or python3).
# PROXY_API_KEY selects the test key; TOKENIZER_MODEL selects the token-stream
# test's real tokenizer (default: bundled Qwen3.5).
# Logs are written under test/proxy/test_all-{timestamp}.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -z "${PYTHON:-}" ]; then
    if [ -x "$ROOT/../venv/bin/python" ]; then PYTHON="$ROOT/../venv/bin/python"; else PYTHON=python3; fi
fi
export PROXY_API_KEY="${PROXY_API_KEY:-test-proxy-real-key}"
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
from common import live_test_engines, load_engine_endpoints  # noqa: E402

ok = True
for engine in live_test_engines():
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
run "py-compile"            0 "$PYTHON" -m py_compile proxyserver/*.py proxyserver/tokenization/*.py test/*.py
run "configuration"         0 "$PYTHON" test/test-configuration.py
run "verl"                  0 "$PYTHON" test/test-verl.py
run "engines"               0 "$PYTHON" test/test-engines.py
run "live-helpers"          0 "$PYTHON" test/test-live-helpers.py
run "sglang-transport"      0 "$PYTHON" test/test-sglang-transport.py
run "token-stream"          0 "$PYTHON" test/test-token-stream.py
run "profiles"              0 "$PYTHON" test/test-profiles.py
run "recorder"              0 "$PYTHON" test/test-recorder.py
run "routed-experts"        0 "$PYTHON" test/test-routed-experts.py
run "recovery"              0 "$PYTHON" test/test-recovery.py
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
    echo "Engine endpoint preflight failed — check test/test_engines.yaml and environment overrides, or run with --offline."
    exit 1
fi

echo; echo "== live =="
run "contract"      1 "$PYTHON" test/test-contract.py
LIVE_ENGINES=$("$PYTHON" - <<'EOF'
import sys
sys.path.insert(0, "test")
from common import live_test_engines
print(" ".join(live_test_engines()))
EOF
) || exit 1
for engine in $LIVE_ENGINES; do
    run "direct-$engine" 1 env INFERENCE_ENGINE="$engine" "$PYTHON" test/test-direct.py
    if [ "$engine" = "sglang" ]; then
        run "slime" 1 "$PYTHON" test/test-slime.py
    fi
done

echo; echo "== summary =="
printf '%s\n' "${RESULTS[@]}"
if [ $FAILED -eq 0 ]; then
    echo "ALL TESTS PASS"
else
    echo "SOME TESTS FAILED (a failure that survived its retry is real — check the log)"
fi
exit $FAILED
