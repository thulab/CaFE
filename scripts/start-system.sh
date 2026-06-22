#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/system-common.sh"

mkdir -p "$TSBENCHMARK_SYSTEM_DIR"

tsbenchmark_require_frontend_runtime

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

  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash -c "$command" >>"$log_file" 2>&1 </dev/null &
  else
    nohup bash -c "$command" >>"$log_file" 2>&1 </dev/null &
  fi
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

# stub-service 默认随系统一起拉起，使 model_adapter=rest（默认值）能就地跑通。
# 要接真实推理服务时，设 TSBENCHMARK_START_STUB=0 跳过桩，并按需 export
# TSBENCHMARK_TIMER_SERVICE_BASE_URL=... 指向真实服务。
if [[ "${TSBENCHMARK_START_STUB:-1}" != "0" ]]; then
  stub_host="${TSBENCHMARK_STUB_HOST:-127.0.0.1}"
  stub_port="${TSBENCHMARK_STUB_PORT:-10810}"
  if curl -fsS \
    --connect-timeout "${TSBENCHMARK_TIMER_PROBE_CONNECT_TIMEOUT_SECONDS:-1}" \
    --max-time "${TSBENCHMARK_TIMER_PROBE_TIMEOUT_SECONDS:-3}" \
    "http://$stub_host:$stub_port/health/liveness" >/dev/null 2>&1; then
    echo "stub-service skipped; timer service already responding on http://$stub_host:$stub_port"
  else
    start_service stub-service "$(tsbenchmark_stub_cmd)"
  fi
fi
start_service backend "$(tsbenchmark_backend_cmd)"
start_service frontend "$(tsbenchmark_frontend_cmd)"
echo "system state: $TSBENCHMARK_SYSTEM_DIR"
