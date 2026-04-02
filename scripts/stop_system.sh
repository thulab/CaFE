#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONF_PATH="${TSBENCHMARK_CONF:-${REPO_ROOT}/conf/system.toml}"

read_conf() {
  PYTHONPATH="${REPO_ROOT}" python -m backend.app.config get "$1" "${CONF_PATH}"
}

resolve_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${REPO_ROOT}/${path}"
  fi
}

RUNTIME_ROOT="$(resolve_path "$(read_conf system.runtime.root)")"
RUNTIME_DIR="${RUNTIME_ROOT}/system"
BACKEND_PID_FILE="${RUNTIME_DIR}/$(read_conf system.runtime.backend_pid_file)"
FRONTEND_PID_FILE="${RUNTIME_DIR}/$(read_conf system.runtime.frontend_pid_file)"
BACKEND_PORT="$(read_conf service.backend.port)"
FRONTEND_PORT="$(read_conf service.frontend.port)"
SHUTDOWN_GRACE_ATTEMPTS="$(read_conf system.shutdown.grace_attempts)"
SHUTDOWN_INTERVAL_SECONDS="$(read_conf system.shutdown.interval_seconds)"

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
  for ((i = 1; i <= SHUTDOWN_GRACE_ATTEMPTS; i++)); do
    if ! is_pid_running "${pid}"; then
      echo "${name} stopped (PID ${pid})."
      return 0
    fi
    sleep "${SHUTDOWN_INTERVAL_SECONDS}"
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
