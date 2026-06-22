#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
STATE_DIR="$(mktemp -d)"
OLD_NODE_STATE_DIR="$STATE_DIR/old-node"

source "$ROOT_DIR/scripts/system-common.sh"
BACKEND_DEFAULT="$(TSBENCHMARK_BACKEND_CMD= tsbenchmark_backend_cmd)"
FRONTEND_DEFAULT="$(TSBENCHMARK_FRONTEND_CMD= tsbenchmark_frontend_cmd)"
if [[ "$BACKEND_DEFAULT" != *".venv/bin/uvicorn"* ]]; then
  echo "backend default command should exec the uvicorn binary, got: $BACKEND_DEFAULT" >&2
  exit 1
fi
if [[ "$FRONTEND_DEFAULT" != *"node_modules/.bin/vite"* ]]; then
  echo "frontend default command should exec the vite binary, got: $FRONTEND_DEFAULT" >&2
  exit 1
fi

cleanup() {
  TSBENCHMARK_SYSTEM_DIR="$STATE_DIR" "$ROOT_DIR/scripts/stop-system.sh" >/dev/null 2>&1 || true
  TSBENCHMARK_SYSTEM_DIR="$OLD_NODE_STATE_DIR" "$ROOT_DIR/scripts/stop-system.sh" >/dev/null 2>&1 || true
  rm -rf "$STATE_DIR"
}
trap cleanup EXIT

FAKE_BIN="$STATE_DIR/fake-bin"
mkdir -p "$FAKE_BIN" "$OLD_NODE_STATE_DIR"
cat >"$FAKE_BIN/node" <<'NODE'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" || "${1:-}" == "-v" ]]; then
  echo "v15.14.0"
  exit 0
fi
echo "fake old node cannot execute $*" >&2
exit 1
NODE
chmod +x "$FAKE_BIN/node"
if PATH="$FAKE_BIN:$PATH" \
  TSBENCHMARK_SYSTEM_DIR="$OLD_NODE_STATE_DIR" \
  TSBENCHMARK_BACKEND_CMD='exec /bin/sleep 60' \
  TSBENCHMARK_START_GRACE_SECONDS=0.2 \
  "$ROOT_DIR/scripts/start-system.sh" >/tmp/tsbenchmark-old-node.out 2>&1; then
  echo "expected unsupported Node.js startup to fail" >&2
  exit 1
fi
grep -q "Node.js 20.19+ or 22.12+" /tmp/tsbenchmark-old-node.out
test ! -f "$OLD_NODE_STATE_DIR/backend.pid"

cat >"$FAKE_BIN/node" <<'NODE'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" || "${1:-}" == "-v" ]]; then
  echo "v22.22.3"
  exit 0
fi
if [[ "${1:-}" == "--tsbenchmark-sleep" ]]; then
  /bin/sleep 60
  exit 0
fi
echo "unexpected fake node invocation: $*" >&2
exit 1
NODE
chmod +x "$FAKE_BIN/node"

export TSBENCHMARK_SYSTEM_DIR="$STATE_DIR"
export TSBENCHMARK_BACKEND_CMD='exec /bin/sleep 60'
export TSBENCHMARK_FRONTEND_CMD='exec node --tsbenchmark-sleep'
export TSBENCHMARK_START_GRACE_SECONDS=0.2

PATH="$FAKE_BIN:/bin:/usr/bin" "$ROOT_DIR/scripts/start-system.sh" >/tmp/tsbenchmark-start.out

test -s "$STATE_DIR/backend.pid"
test -s "$STATE_DIR/frontend.pid"
kill -0 "$(cat "$STATE_DIR/backend.pid")"
kill -0 "$(cat "$STATE_DIR/frontend.pid")"
test -f "$STATE_DIR/backend.log"
test -f "$STATE_DIR/frontend.log"

if "$ROOT_DIR/scripts/start-system.sh" >/tmp/tsbenchmark-start-again.out 2>&1; then
  echo "expected duplicate start to fail" >&2
  exit 1
fi
grep -q "already running" /tmp/tsbenchmark-start-again.out

"$ROOT_DIR/scripts/status-system.sh" >/tmp/tsbenchmark-status.out
grep -q "backend: running" /tmp/tsbenchmark-status.out
grep -q "frontend: running" /tmp/tsbenchmark-status.out

"$ROOT_DIR/scripts/stop-system.sh" >/tmp/tsbenchmark-stop.out
grep -q "stopped backend" /tmp/tsbenchmark-stop.out
grep -q "stopped frontend" /tmp/tsbenchmark-stop.out
test ! -f "$STATE_DIR/backend.pid"
test ! -f "$STATE_DIR/frontend.pid"

"$ROOT_DIR/scripts/status-system.sh" >/tmp/tsbenchmark-status-stopped.out
grep -q "backend: stopped" /tmp/tsbenchmark-status-stopped.out
grep -q "frontend: stopped" /tmp/tsbenchmark-status-stopped.out
