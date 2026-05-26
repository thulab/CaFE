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

tsbenchmark_node_version_supported() {
  local version="${1#v}"
  local major minor patch
  IFS=. read -r major minor patch <<<"$version"

  if [[ ! "$major" =~ ^[0-9]+$ || ! "$minor" =~ ^[0-9]+$ ]]; then
    return 1
  fi

  (( major == 20 && minor >= 19 )) || (( major == 22 && minor >= 12 )) || (( major > 22 ))
}

tsbenchmark_require_frontend_runtime() {
  local node_version

  if [[ -n "${TSBENCHMARK_FRONTEND_CMD:-}" ]]; then
    return 0
  fi

  if ! command -v node >/dev/null 2>&1; then
    echo "Node.js 20.19+ or 22.12+ is required for the Vue/Vite frontend; node was not found." >&2
    return 1
  fi

  node_version="$(node --version 2>/dev/null || true)"
  if ! tsbenchmark_node_version_supported "$node_version"; then
    echo "Node.js 20.19+ or 22.12+ is required for the Vue/Vite frontend; found ${node_version:-unknown}." >&2
    echo "Install a supported Node.js version, then retry ./scripts/start-system.sh." >&2
    return 1
  fi

  if ! command -v npm >/dev/null 2>&1; then
    echo "npm is required to install and run the Vue/Vite frontend; npm was not found." >&2
    return 1
  fi
}

tsbenchmark_backend_cmd() {
  if [[ -n "${TSBENCHMARK_BACKEND_CMD:-}" ]]; then
    printf '%s\n' "$TSBENCHMARK_BACKEND_CMD"
    return
  fi
  # 鉴权重构后 create_app() 强校验 TSBENCHMARK_AUTH_SECRET；本地一键启动注入一个
  # 32 字节默认 dev secret 和 admin 密码（用户已 export 的会覆盖默认）。
  # 注：format string 是 single-quoted，${VAR:-...} 不会被 printf 时展开，
  # 是在 nohup bash -c 运行时才展开 —— 这是我们想要的运行时绑定。
  printf 'cd "%s/backend" && { [[ -x .venv/bin/uvicorn ]] || uv sync --quiet; } && TSBENCHMARK_AUTH_SECRET="${TSBENCHMARK_AUTH_SECRET:-tsbenchmark-dev-secret-32bytes-padding}" TSBENCHMARK_ADMIN_PASSWORD="${TSBENCHMARK_ADMIN_PASSWORD:-admin}" exec .venv/bin/uvicorn app.main:create_app --factory --host "%s" --port "%s"\n' \
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
