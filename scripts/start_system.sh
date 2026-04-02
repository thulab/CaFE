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
BACKEND_LOG_FILE="${RUNTIME_DIR}/$(read_conf system.runtime.backend_log_file)"
FRONTEND_LOG_FILE="${RUNTIME_DIR}/$(read_conf system.runtime.frontend_log_file)"
BACKEND_HOST="$(read_conf service.backend.host)"
BACKEND_PORT="$(read_conf service.backend.port)"
FRONTEND_HOST="$(read_conf service.frontend.host)"
FRONTEND_PORT="$(read_conf service.frontend.port)"
HEALTH_ATTEMPTS="$(read_conf system.healthcheck.attempts)"
HEALTH_INTERVAL_SECONDS="$(read_conf system.healthcheck.interval_seconds)"
HEALTH_TIMEOUT_SECONDS="$(read_conf system.healthcheck.timeout_seconds)"
BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"
FRONTEND_URL="http://${FRONTEND_HOST}:${FRONTEND_PORT}"

mkdir -p "${RUNTIME_DIR}"

is_pid_running() {
  local pid="$1"
  kill -0 "${pid}" 2>/dev/null
}

check_pid_file() {
  local name="$1"
  local pid_file="$2"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if [[ -n "${pid}" ]] && is_pid_running "${pid}"; then
      echo "${name} is already running with PID ${pid}."
      exit 1
    fi
    rm -f "${pid_file}"
  fi
}

wait_for_health() {
  local name="$1"
  local url="$2"
  local attempts="${3:-${HEALTH_ATTEMPTS}}"

  for ((i = 1; i <= attempts; i++)); do
    if command -v curl >/dev/null 2>&1; then
      if curl -fsS "${url}" >/dev/null 2>&1; then
        return 0
      fi
    else
      if python -c "import urllib.request; urllib.request.urlopen('${url}', timeout=${HEALTH_TIMEOUT_SECONDS})" >/dev/null 2>&1; then
        return 0
      fi
    fi
    sleep "${HEALTH_INTERVAL_SECONDS}"
  done

  echo "${name} did not become healthy in time. Check logs:"
  echo "  ${BACKEND_LOG_FILE}"
  echo "  ${FRONTEND_LOG_FILE}"
  exit 1
}

check_pid_file "Backend" "${BACKEND_PID_FILE}"
check_pid_file "Frontend" "${FRONTEND_PID_FILE}"

cd "${REPO_ROOT}"

nohup env TSBENCHMARK_CONF="${CONF_PATH}" python -m backend.app.main >"${BACKEND_LOG_FILE}" 2>&1 &
BACKEND_PID=$!
echo "${BACKEND_PID}" >"${BACKEND_PID_FILE}"

wait_for_health "Backend" "${BACKEND_URL}/health"

nohup env TSBENCHMARK_CONF="${CONF_PATH}" TSBENCHMARK_BACKEND_URL="${BACKEND_URL}" python -m frontend.app >"${FRONTEND_LOG_FILE}" 2>&1 &
FRONTEND_PID=$!
echo "${FRONTEND_PID}" >"${FRONTEND_PID_FILE}"

wait_for_health "Frontend" "${FRONTEND_URL}/health"

echo "System started."
echo "Backend:  ${BACKEND_URL} (PID ${BACKEND_PID})"
echo "Frontend: ${FRONTEND_URL} (PID ${FRONTEND_PID})"
echo "Logs:"
echo "  ${BACKEND_LOG_FILE}"
echo "  ${FRONTEND_LOG_FILE}"
