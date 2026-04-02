#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${REPO_ROOT}/runtime/system"
BACKEND_PID_FILE="${RUNTIME_DIR}/backend.pid"
FRONTEND_PID_FILE="${RUNTIME_DIR}/frontend.pid"
BACKEND_PORT=8000
FRONTEND_PORT=8501

is_pid_running() {
  local pid="$1"
  kill -0 "${pid}" 2>/dev/null
}

stop_pid() {
  local name="$1"
  local pid="$2"

  if ! is_pid_running "${pid}"; then
    echo "${name} PID ${pid} is not running."
    return 0
  fi

  kill "${pid}" 2>/dev/null || true
  for _ in {1..10}; do
    if ! is_pid_running "${pid}"; then
      echo "${name} stopped (PID ${pid})."
      return 0
    fi
    sleep 1
  done

  kill -9 "${pid}" 2>/dev/null || true
  if ! is_pid_running "${pid}"; then
    echo "${name} force-stopped (PID ${pid})."
    return 0
  fi

  echo "Failed to stop ${name} PID ${pid}."
  return 1
}

stop_from_pid_file() {
  local name="$1"
  local pid_file="$2"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if [[ -n "${pid}" ]]; then
      stop_pid "${name}" "${pid}"
    fi
    rm -f "${pid_file}"
  else
    echo "${name} PID file not found."
  fi
}

stop_from_port() {
  local name="$1"
  local port="$2"

  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi

  local pid
  pid="$(lsof -ti tcp:"${port}" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
  if [[ -n "${pid}" ]]; then
    stop_pid "${name}" "${pid}"
  fi
}

stop_from_pid_file "Frontend" "${FRONTEND_PID_FILE}"
stop_from_pid_file "Backend" "${BACKEND_PID_FILE}"

stop_from_port "Frontend" "${FRONTEND_PORT}"
stop_from_port "Backend" "${BACKEND_PORT}"

echo "System shutdown complete."
