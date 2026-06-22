#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/system-common.sh"

status_service() {
  local service="$1"
  if tsbenchmark_is_running "$service"; then
    echo "$service: running (pid $(tsbenchmark_read_pid "$service"), log $(tsbenchmark_log_file "$service"))"
  else
    tsbenchmark_remove_stale_pid "$service"
    echo "$service: stopped"
  fi
}

status_service stub-service
status_service backend
status_service frontend
