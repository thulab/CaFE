#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/system-common.sh"

stop_service() {
  local service="$1"
  local pid_file pid
  pid_file="$(tsbenchmark_pid_file "$service")"
  pid="$(tsbenchmark_read_pid "$service" || true)"

  if [[ -z "$pid" ]]; then
    rm -f "$pid_file"
    echo "$service already stopped"
    return 0
  fi

  if kill -0 "$pid" 2>/dev/null; then
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
    echo "stopped $service"
  else
    echo "$service was not running"
  fi

  rm -f "$pid_file"
}

stop_service backend
stop_service frontend
# stub-service 放最后停：让 backend 先释放任何 in-flight forecast 调用。
# 未被 start-system 拉起时 pid 文件不存在，stop_service 会安静 no-op。
stop_service stub-service
