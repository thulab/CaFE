#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/system-common.sh"

SERVICE="paper-sample-explorer"
HOST="${TSBENCHMARK_PAPER_EXPLORER_HOST:-0.0.0.0}"
PORT="${TSBENCHMARK_PAPER_EXPLORER_PORT:-8766}"

probe_host() {
  if [[ "$HOST" == "0.0.0.0" || "$HOST" == "::" ]]; then
    printf '127.0.0.1\n'
  else
    printf '%s\n' "$HOST"
  fi
}

start() {
  mkdir -p "$TSBENCHMARK_SYSTEM_DIR"
  tsbenchmark_remove_stale_pid "$SERVICE"
  if tsbenchmark_is_running "$SERVICE"; then
    echo "$SERVICE already running (pid $(tsbenchmark_read_pid "$SERVICE"))"
    return 0
  fi

  local pid_file log_file pid health_url
  pid_file="$(tsbenchmark_pid_file "$SERVICE")"
  log_file="$(tsbenchmark_log_file "$SERVICE")"
  health_url="http://$(probe_host):$PORT/api/health"
  : >"$log_file"

  if command -v setsid >/dev/null 2>&1; then
    nohup setsid python3 "$SCRIPT_DIR/paper_sample_explorer.py" \
      --host "$HOST" \
      --port "$PORT" \
      "$@" >>"$log_file" 2>&1 </dev/null &
  else
    nohup python3 "$SCRIPT_DIR/paper_sample_explorer.py" \
      --host "$HOST" \
      --port "$PORT" \
      "$@" >>"$log_file" 2>&1 </dev/null &
  fi
  pid="$!"
  echo "$pid" >"$pid_file"

  for _ in {1..600}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pid_file"
      echo "failed to start $SERVICE; see $log_file" >&2
      tail -n 30 "$log_file" >&2 || true
      return 1
    fi
    if curl -fsS --connect-timeout 1 --max-time 2 "$health_url" >/dev/null 2>&1; then
      echo "started $SERVICE on http://$HOST:$PORT (pid $pid, log $log_file)"
      return 0
    fi
    sleep 0.1
  done

  echo "$SERVICE is still building its index (pid $pid, log $log_file)"
  echo "run '$0 status' to check readiness"
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
  for _ in {1..50}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$(tsbenchmark_pid_file "$SERVICE")"
  echo "stopped $SERVICE (pid $pid)"
}

status() {
  tsbenchmark_remove_stale_pid "$SERVICE"
  if ! tsbenchmark_is_running "$SERVICE"; then
    echo "$SERVICE stopped"
    return 0
  fi

  local health_url
  health_url="http://$(probe_host):$PORT/api/health"
  if curl -fsS --connect-timeout 1 --max-time 2 "$health_url" >/dev/null 2>&1; then
    echo "$SERVICE running and ready (pid $(tsbenchmark_read_pid "$SERVICE")) on http://$HOST:$PORT"
  else
    echo "$SERVICE running but not ready (pid $(tsbenchmark_read_pid "$SERVICE")); index may still be building"
  fi
  echo "log: $(tsbenchmark_log_file "$SERVICE")"
}

logs() {
  local log_file
  log_file="$(tsbenchmark_log_file "$SERVICE")"
  if [[ ! -f "$log_file" ]]; then
    echo "no log file: $log_file" >&2
    return 1
  fi
  tail -n 80 "$log_file"
}

foreground() {
  exec python3 "$SCRIPT_DIR/paper_sample_explorer.py" \
    --host "$HOST" \
    --port "$PORT" \
    "$@"
}

command="${1:-start}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$command" in
  start) start "$@" ;;
  stop) stop ;;
  status) status ;;
  restart)
    stop
    start "$@"
    ;;
  logs) logs ;;
  foreground) foreground "$@" ;;
  *)
    echo "usage: $0 {start|stop|restart|status|logs|foreground} [explorer options]" >&2
    exit 2
    ;;
esac
