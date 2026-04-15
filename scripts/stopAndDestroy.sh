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

ensure_safe_runtime_root() {
  local runtime_root="$1"
  local repo_root="$2"
  local home_dir
  home_dir="${HOME:-}"

  case "${runtime_root}" in
    ""|"/")
      echo "Refusing to destroy unsafe runtime root: ${runtime_root:-<empty>}"
      exit 1
      ;;
  esac

  if [[ "${runtime_root}" = "${repo_root}" ]]; then
    echo "Refusing to destroy repository root: ${runtime_root}"
    exit 1
  fi

  if [[ -n "${home_dir}" && "${runtime_root}" = "${home_dir}" ]]; then
    echo "Refusing to destroy HOME directory: ${runtime_root}"
    exit 1
  fi
}

RUNTIME_ROOT="$(resolve_path "$(read_conf system.runtime.root)")"

ensure_safe_runtime_root "${RUNTIME_ROOT}" "${REPO_ROOT}"

bash "${SCRIPT_DIR}/stop_system.sh"

mkdir -p "${RUNTIME_ROOT}"

find "${RUNTIME_ROOT}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +

echo "Runtime data removed from ${RUNTIME_ROOT}."
