#!/usr/bin/env bash
# Track A 基线 runner（plan Task A0）：起一个隔离的后端（进程内确定性桩适配器），
# 用真实 CSV 走完整 API 链（upload→manifest→load-job→wizard→run→ranking→sample forecast），
# 把结果落到 docs/superpowers/baselines/<date>-baseline-record.md，跑完自动停后端。
#
# 用法：scripts/baseline-run.sh [csv_path]   默认 test/flow_template.csv
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/system-common.sh"

ROOT="$TSBENCHMARK_ROOT_DIR"
PORT="${TSBENCHMARK_BASELINE_PORT:-18900}"
HOST="127.0.0.1"
CSV="${1:-$ROOT/test/flow_template.csv}"
RUNTIME="$ROOT/runtime/baseline"
RECORD="$ROOT/docs/superpowers/baselines/$(date +%F)-baseline-record.md"
PY="$ROOT/backend/.venv/bin/python"

if [[ ! -f "$CSV" ]]; then
  echo "csv not found: $CSV" >&2
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  (cd "$ROOT/backend" && uv sync --quiet)
fi

rm -rf "$RUNTIME"
mkdir -p "$RUNTIME"
LOG="$RUNTIME/backend.log"

(
  cd "$ROOT/backend" \
    && TSBENCHMARK_RUNTIME_DIR="$RUNTIME" \
       TSBENCHMARK_DATABASE_URL="sqlite:///$RUNTIME/baseline.db" \
       TSBENCHMARK_MODEL_ADAPTER=stub \
       exec .venv/bin/uvicorn app.main:create_app --factory --host "$HOST" --port "$PORT"
) >"$LOG" 2>&1 &
BACKEND_PID=$!
trap 'kill "$BACKEND_PID" 2>/dev/null || true' EXIT

ready=""
for _ in $(seq 1 60); do
  if curl -fsS "http://$HOST:$PORT/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    break
  fi
  sleep 0.5
done
if [[ "$ready" != 1 ]]; then
  echo "backend failed to become ready on http://$HOST:$PORT; see $LOG" >&2
  tail -n 30 "$LOG" >&2 || true
  exit 1
fi

echo "backend ready on http://$HOST:$PORT (pid $BACKEND_PID); walking API chain with $CSV"
"$PY" "$ROOT/scripts/baseline_run.py" "http://$HOST:$PORT" "$CSV" "$RECORD"
echo "baseline record written: $RECORD"
