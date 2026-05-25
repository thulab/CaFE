#!/usr/bin/env bash
# 管理本地 timer-rest-service 桩服务（backend/stub_service）。
# 用法：scripts/stub-service.sh {start|stop|status}
# 默认监听 127.0.0.1:10810，与 TSBENCHMARK_TIMER_SERVICE_BASE_URL 的默认值一致。
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/system-common.sh"

SERVICE="stub-service"
HOST="${TSBENCHMARK_STUB_HOST:-127.0.0.1}"
PORT="${TSBENCHMARK_STUB_PORT:-10810}"

stub_cmd() {
  printf 'cd "%s/backend" && { [[ -x .venv/bin/uvicorn ]] || uv sync --quiet; } && exec .venv/bin/uvicorn stub_service.main:app --host "%s" --port "%s"\n' \
    "$TSBENCHMARK_ROOT_DIR" "$HOST" "$PORT"
}

start() {
  mkdir -p "$TSBENCHMARK_SYSTEM_DIR"
  tsbenchmark_remove_stale_pid "$SERVICE"
  if tsbenchmark_is_running "$SERVICE"; then
    echo "$SERVICE already running (pid $(tsbenchmark_read_pid "$SERVICE"))"
    return 0
  fi

  local pid_file log_file pid
  pid_file="$(tsbenchmark_pid_file "$SERVICE")"
  log_file="$(tsbenchmark_log_file "$SERVICE")"
  : >"$log_file"

  nohup bash -c "$(stub_cmd)" >>"$log_file" 2>&1 &
  pid="$!"
  echo "$pid" >"$pid_file"

  sleep "${TSBENCHMARK_START_GRACE_SECONDS:-1}"
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pid_file"
    echo "failed to start $SERVICE; see $log_file" >&2
    tail -n 20 "$log_file" >&2 || true
    return 1
  fi
  echo "started $SERVICE on http://$HOST:$PORT (pid $pid, log $log_file)"
}

stop() {
  if ! tsbenchmark_is_running "$SERVICE"; then
    echo "$SERVICE not running"
    rm -f "$(tsbenchmark_pid_file "$SERVICE")"
    return 0
  fi
  local pid
  pid="$(tsbenchmark_read_pid "$SERVICE")"
  kill "$pid" 2>/dev/null || true
  rm -f "$(tsbenchmark_pid_file "$SERVICE")"
  echo "stopped $SERVICE (pid $pid)"
}

status() {
  if tsbenchmark_is_running "$SERVICE"; then
    echo "$SERVICE running (pid $(tsbenchmark_read_pid "$SERVICE")) on http://$HOST:$PORT"
  else
    echo "$SERVICE stopped"
  fi
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
