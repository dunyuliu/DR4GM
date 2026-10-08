#!/bin/bash
#
# run_streamlit_boot_smoke.sh -- boot the DR4GM Interactive Explorer as a
# real `streamlit run` process (headless) and confirm /_stcore/health
# returns "ok". Catches bugs an AppTest headless run can't: port binding,
# server startup, missing runtime deps -- not just script-body exceptions.
#
# Offline: the app's default dataset resolves to the vendored
# data/eqdyna.0001.A.coarse.npz, so no network access occurs.
#
# PID discipline: the Streamlit process is tracked by the PID this script
# started ($!) via a pidfile; cleanup kills that PID only, never pkill -f.
#
# Usage:
#   bash test_system/run_streamlit_boot_smoke.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_PATH="$REPO_ROOT/src/web/dr4gm_interactive_explorer.py"

PORT="${DR4GM_SMOKE_PORT:-8599}"
PIDFILE="$(mktemp -t dr4gm_streamlit_smoke.XXXXXX.pid)"
LOGFILE="$(mktemp -t dr4gm_streamlit_smoke.XXXXXX.log)"

cleanup() {
  if [ -f "$PIDFILE" ]; then
    pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
      done
      if kill -0 "$pid" 2>/dev/null; then
        echo "BOOT SMOKE FAIL: PID $pid did not exit after SIGTERM" >&2
        kill -9 "$pid" 2>/dev/null || true
        exit 1
      fi
    fi
    rm -f "$PIDFILE"
  fi
  rm -f "$LOGFILE"
}
trap cleanup EXIT

cd "$REPO_ROOT"
streamlit run "$APP_PATH" \
  --server.headless true \
  --server.port "$PORT" \
  --browser.gatherUsageStats false \
  > "$LOGFILE" 2>&1 &
STREAMLIT_PID=$!
echo "$STREAMLIT_PID" > "$PIDFILE"

HEALTH_URL="http://localhost:${PORT}/_stcore/health"
READY=0
for _ in $(seq 1 60); do
  if ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
    echo "BOOT SMOKE FAIL: streamlit process exited early. Log:" >&2
    cat "$LOGFILE" >&2
    exit 1
  fi
  if curl -fsS "$HEALTH_URL" 2>/dev/null | grep -qx "ok"; then
    READY=1
    break
  fi
  sleep 0.5
done

if [ "$READY" -ne 1 ]; then
  echo "BOOT SMOKE FAIL: $HEALTH_URL did not return 'ok' within timeout. Log:" >&2
  cat "$LOGFILE" >&2
  exit 1
fi

RESPONSE="$(curl -fsS "$HEALTH_URL")"
if [ "$RESPONSE" != "ok" ]; then
  echo "BOOT SMOKE FAIL: expected 'ok', got '$RESPONSE'" >&2
  exit 1
fi

echo "BOOT SMOKE PASS: $HEALTH_URL -> ok (PID $STREAMLIT_PID)"

# Cleanup (kill by $STREAMLIT_PID, verify no leftover) runs via the EXIT trap.
# Explicitly re-verify here too, so a failure to terminate is caught before
# the trap's own exit-code is relied upon.
kill "$STREAMLIT_PID" 2>/dev/null || true
for _ in $(seq 1 20); do
  kill -0 "$STREAMLIT_PID" 2>/dev/null || break
  sleep 0.25
done
if kill -0 "$STREAMLIT_PID" 2>/dev/null; then
  echo "BOOT SMOKE FAIL: PID $STREAMLIT_PID still running after kill" >&2
  exit 1
fi
rm -f "$PIDFILE"
echo "BOOT SMOKE PASS: PID $STREAMLIT_PID terminated cleanly, no leftover process"
