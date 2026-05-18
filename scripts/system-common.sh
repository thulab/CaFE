#!/usr/bin/env bash

tsbenchmark_root_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P
}

TSBENCHMARK_ROOT_DIR="$(tsbenchmark_root_dir)"
TSBENCHMARK_SYSTEM_DIR="${TSBENCHMARK_SYSTEM_DIR:-$TSBENCHMARK_ROOT_DIR/.tsbenchmark-system}"

tsbenchmark_pid_file() {
  printf '%s/%s.pid\n' "$TSBENCHMARK_SYSTEM_DIR" "$1"
}

tsbenchmark_log_file() {
  printf '%s/%s.log\n' "$TSBENCHMARK_SYSTEM_DIR" "$1"
}

tsbenchmark_read_pid() {
  local pid_file
  pid_file="$(tsbenchmark_pid_file "$1")"
  if [[ -s "$pid_file" ]]; then
    cat "$pid_file"
  fi
}

tsbenchmark_is_running() {
  local service="$1"
  local pid err
  pid="$(tsbenchmark_read_pid "$service" || true)"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  err="$(kill -0 "$pid" 2>&1)" && return 0
  [[ "$err" == *"Operation not permitted"* ]]
}

tsbenchmark_remove_stale_pid() {
  local service="$1"
  local pid_file
  pid_file="$(tsbenchmark_pid_file "$service")"
  if [[ -f "$pid_file" ]] && ! tsbenchmark_is_running "$service"; then
    rm -f "$pid_file"
  fi
}

tsbenchmark_backend_cmd() {
  if [[ -n "${TSBENCHMARK_BACKEND_CMD:-}" ]]; then
    printf '%s\n' "$TSBENCHMARK_BACKEND_CMD"
    return
  fi
  printf 'cd "%s/backend" && { [[ -x .venv/bin/uvicorn ]] || uv sync --quiet; } && exec .venv/bin/uvicorn app.main:create_app --factory --host "%s" --port "%s"\n' \
    "$TSBENCHMARK_ROOT_DIR" \
    "${TSBENCHMARK_BACKEND_HOST:-127.0.0.1}" \
    "${TSBENCHMARK_BACKEND_PORT:-8000}"
}

tsbenchmark_frontend_cmd() {
  if [[ -n "${TSBENCHMARK_FRONTEND_CMD:-}" ]]; then
    printf '%s\n' "$TSBENCHMARK_FRONTEND_CMD"
    return
  fi
  printf 'cd "%s/frontend" && { [[ -x node_modules/.bin/vite ]] || npm install --silent; } && exec ./node_modules/.bin/vite --host "%s" --port "%s"\n' \
    "$TSBENCHMARK_ROOT_DIR" \
    "${TSBENCHMARK_FRONTEND_HOST:-127.0.0.1}" \
    "${TSBENCHMARK_FRONTEND_PORT:-5173}"
}
