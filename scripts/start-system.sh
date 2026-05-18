#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/system-common.sh"

mkdir -p "$TSBENCHMARK_SYSTEM_DIR"

start_service() {
  local service="$1"
  local command="$2"
  local pid_file log_file pid

  tsbenchmark_remove_stale_pid "$service"
  if tsbenchmark_is_running "$service"; then
    echo "$service already running (pid $(tsbenchmark_read_pid "$service"))"
    return 1
  fi

  pid_file="$(tsbenchmark_pid_file "$service")"
  log_file="$(tsbenchmark_log_file "$service")"
  : >"$log_file"

  nohup bash -lc "$command" >>"$log_file" 2>&1 &
  pid="$!"
  echo "$pid" >"$pid_file"

  sleep "${TSBENCHMARK_START_GRACE_SECONDS:-1}"
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pid_file"
    echo "failed to start $service; see $log_file" >&2
    tail -n 20 "$log_file" >&2 || true
    return 1
  fi

  echo "started $service (pid $pid, log $log_file)"
}

start_service backend "$(tsbenchmark_backend_cmd)"
start_service frontend "$(tsbenchmark_frontend_cmd)"
echo "system state: $TSBENCHMARK_SYSTEM_DIR"
