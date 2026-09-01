#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
remote_host="${CAFE_STABILITY_HOST:-timecho92}"
remote_orchestration="${CAFE_STABILITY_ORCHESTRATION:-/data/xmy/CaFE/runtime/orchestration/short_stability10_inference_3node_78ef32f_20260831}"
remote_experiments="${CAFE_STABILITY_EXPERIMENTS:-/data/xmy/CaFE/runtime/experiments}"

mkdir -p \
  "$script_dir/raw/remote_stability" \
  "$script_dir/raw/suite_summaries"

scp "$remote_host:$remote_orchestration/stability/*" \
  "$script_dir/raw/remote_stability/"

for seed in \
  2026082701 2026082702 2026082703 2026082704 2026082705 \
  2026082706 2026082707 2026082708 2026082709 2026082710
do
  experiment="gift-v15-short-stability10-head78ef32f-seed${seed}"
  scp \
    "$remote_host:$remote_experiments/$experiment/04_analysis_suite/task_equal_summary.json" \
    "$script_dir/raw/suite_summaries/seed${seed}.json"
done

printf 'Downloaded read-only stability inputs to %s/raw\n' "$script_dir"
