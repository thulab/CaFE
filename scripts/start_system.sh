#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${REPO_ROOT}/runtime/system"
BACKEND_PID_FILE="${RUNTIME_DIR}/backend.pid"
FRONTEND_PID_FILE="${RUNTIME_DIR}/frontend.pid"
BACKEND_LOG_FILE="${RUNTIME_DIR}/backend.log"
FRONTEND_LOG_FILE="${RUNTIME_DIR}/frontend.log"
BACKEND_URL="http://127.0.0.1:8000"
FRONTEND_URL="http://127.0.0.1:8501"

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
  local attempts="${3:-30}"

  for ((i = 1; i <= attempts; i++)); do
    if command -v curl >/dev/null 2>&1; then
      if curl -fsS "${url}" >/dev/null 2>&1; then
        return 0
      fi
    else
      if python -c "import urllib.request; urllib.request.urlopen('${url}', timeout=1)" >/dev/null 2>&1; then
        return 0
      fi
    fi
    sleep 1
  done

  echo "${name} did not become healthy in time. Check logs:"
  echo "  ${BACKEND_LOG_FILE}"
  echo "  ${FRONTEND_LOG_FILE}"
  exit 1
}

check_pid_file "Backend" "${BACKEND_PID_FILE}"
check_pid_file "Frontend" "${FRONTEND_PID_FILE}"

cd "${REPO_ROOT}"

nohup python -m backend.app.main >"${BACKEND_LOG_FILE}" 2>&1 &
BACKEND_PID=$!
echo "${BACKEND_PID}" >"${BACKEND_PID_FILE}"

wait_for_health "Backend" "${BACKEND_URL}/health"

nohup env TSBENCHMARK_BACKEND_URL="${BACKEND_URL}" python -m frontend.app >"${FRONTEND_LOG_FILE}" 2>&1 &
FRONTEND_PID=$!
echo "${FRONTEND_PID}" >"${FRONTEND_PID_FILE}"

wait_for_health "Frontend" "${FRONTEND_URL}/health"

echo "System started."
echo "Backend:  ${BACKEND_URL} (PID ${BACKEND_PID})"
echo "Frontend: ${FRONTEND_URL} (PID ${FRONTEND_PID})"
echo "Logs:"
echo "  ${BACKEND_LOG_FILE}"
echo "  ${FRONTEND_LOG_FILE}"
