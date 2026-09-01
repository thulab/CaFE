from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REMOTE_ROOTS = {
    "gift": (
        "/data/xmy/CaFE/runtime/ablation_trials/"
        "gift-v15-seed2026082701-target-only-v1"
    ),
    "fev": (
        "/data/xmy/CaFE/runtime/ablation_trials/"
        "fev-mini20-full-v6-target-only-v1"
    ),
}


REMOTE_AGGREGATION = r"""
from pathlib import Path
import json
import sys

import numpy as np
import pyarrow.parquet as pq

root = Path(sys.argv[1])
groups = {}
for path in root.rglob("metric_rows.parquet"):
    table = pq.read_table(
        path,
        columns=[
            "term",
            "dataset_id",
            "model_id",
            "capability_id",
            "capability_level",
            "target_only_mase_degradation",
            "full_input_mase",
            "ablated_input_mase",
            "target_only_response_ratio",
        ],
    )
    for row in table.to_pylist():
        key = (
            str(row["term"]),
            str(row["dataset_id"]),
            str(row["model_id"]),
            str(row["capability_id"]),
            int(row["capability_level"]),
        )
        values = groups.setdefault(
            key,
            {
                "degradation": [],
                "full_mase": [],
                "ablated_mase": [],
                "response_ratio": [],
            },
        )
        values["degradation"].append(float(row["target_only_mase_degradation"]))
        values["full_mase"].append(float(row["full_input_mase"]))
        values["ablated_mase"].append(float(row["ablated_input_mase"]))
        values["response_ratio"].append(float(row["target_only_response_ratio"]))

rows = []
for key, values in sorted(groups.items()):
    term, dataset_id, model_id, capability_id, level = key
    degradation = np.asarray(values["degradation"], dtype=float)
    rows.append(
        {
            "term": term,
            "dataset_id": dataset_id,
            "model_id": model_id,
            "capability_id": capability_id,
            "capability_level": level,
            "paired_view_count": int(degradation.size),
            "mase_degradation_mean": float(np.mean(degradation)),
            "mase_degradation_median": float(np.median(degradation)),
            "positive_degradation_fraction": float(np.mean(degradation > 0.0)),
            "full_input_mase_mean": float(np.mean(values["full_mase"])),
            "ablated_input_mase_mean": float(np.mean(values["ablated_mase"])),
            "response_ratio_mean": float(np.mean(values["response_ratio"])),
        }
    )
print(json.dumps(rows, allow_nan=False, separators=(",", ":")))
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="timecho92")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    return parser.parse_args()


def _ssh(host: str, command: str, *, stdin: str | None = None) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, command],
        input=stdin,
        text=True,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_mean(
    values: list[float], *, seed: int, repetitions: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, array.size, size=(repetitions, array.size))
    means = np.mean(array[indexes], axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _summarize(
    task_rows: list[dict[str, Any]], *, seed: int, repetitions: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    level_groups: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    collapsed_groups: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        level_key = (
            row["suite"],
            row["term"],
            row["model_id"],
            row["capability_id"],
            int(row["capability_level"]),
        )
        level_groups[level_key].append(row)
        collapsed_key = level_key[:-1]
        collapsed_groups[collapsed_key].append(row)

    level_summary: list[dict[str, Any]] = []
    for index, (key, members) in enumerate(sorted(level_groups.items())):
        suite, term, model_id, capability_id, level = key
        values = [float(row["mase_degradation_mean"]) for row in members]
        lower, upper = _bootstrap_mean(
            values, seed=seed + index, repetitions=repetitions
        )
        level_summary.append(
            {
                "suite": suite,
                "term": term,
                "model_id": model_id,
                "capability_id": capability_id,
                "capability_level": level,
                "task_count": len(values),
                "task_equal_mase_degradation": float(np.mean(values)),
                "task_bootstrap_95_ci_lower": lower,
                "task_bootstrap_95_ci_upper": upper,
                "task_median_mase_degradation": float(np.median(values)),
                "positive_task_fraction": float(np.mean(np.asarray(values) > 0.0)),
                "paired_view_count": int(
                    sum(int(row["paired_view_count"]) for row in members)
                ),
            }
        )

    collapsed_summary: list[dict[str, Any]] = []
    for index, (key, members) in enumerate(sorted(collapsed_groups.items())):
        suite, term, model_id, capability_id = key
        by_task: defaultdict[str, list[float]] = defaultdict(list)
        for row in members:
            by_task[str(row["dataset_id"])].append(
                float(row["mase_degradation_mean"])
            )
        values = [float(np.mean(task_values)) for task_values in by_task.values()]
        lower, upper = _bootstrap_mean(
            values,
            seed=seed + 100_000 + index,
            repetitions=repetitions,
        )
        array = np.asarray(values, dtype=float)
        leave_one_out = np.asarray(
            [
                float((np.sum(array) - array[i]) / (array.size - 1))
                for i in range(array.size)
            ]
            if array.size > 1
            else values,
            dtype=float,
        )
        collapsed_summary.append(
            {
                "suite": suite,
                "term": term,
                "model_id": model_id,
                "capability_id": capability_id,
                "level_count": len({int(row["capability_level"]) for row in members}),
                "task_count": len(values),
                "task_level_equal_mase_degradation": float(np.mean(array)),
                "task_bootstrap_95_ci_lower": lower,
                "task_bootstrap_95_ci_upper": upper,
                "task_median_mase_degradation": float(np.median(array)),
                "positive_task_fraction": float(np.mean(array > 0.0)),
                "leave_one_task_out_min": float(np.min(leave_one_out)),
                "leave_one_task_out_max": float(np.max(leave_one_out)),
                "paired_view_count": int(
                    sum(int(row["paired_view_count"]) for row in members)
                ),
            }
        )
    return level_summary, collapsed_summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_task_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {
        "schema_version": "cafe.paper_ablation_collection.v1",
        "host": args.host,
        "remote_roots": REMOTE_ROOTS,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_repetitions": args.bootstrap_repetitions,
    }

    for suite, remote_root in REMOTE_ROOTS.items():
        task_rows = json.loads(
            _ssh(
                args.host,
                f"cd /data/xmy/CaFE && uv run python - {remote_root}",
                stdin=REMOTE_AGGREGATION,
            )
        )
        for row in task_rows:
            all_task_rows.append({"suite": suite, **row})

        summary = json.loads(
            _ssh(args.host, f"cat {remote_root}/summary.json")
        )
        for row in summary["rows"]:
            source_rows.append({"suite": suite, **row})

        coverage = json.loads(
            _ssh(args.host, f"cat {remote_root}/coverage.json")
        )
        for row in coverage["rows"]:
            for capability_id, view_count in (
                row.get("exposed_view_count_by_capability") or {}
            ).items():
                coverage_rows.append(
                    {
                        "suite": suite,
                        "experiment_id": row["experiment_id"],
                        "dataset_id": row["dataset_id"],
                        "model_id": row["model_id"],
                        "capability_id": capability_id,
                        "exposed_view_count": int(view_count),
                        "status": row["status"],
                    }
                )
        missing = json.loads(
            _ssh(args.host, f"cat {remote_root}/missing_pairs.json")
        )
        provenance[f"{suite}_missing_pair_count"] = len(missing["rows"])

    level_summary, collapsed_summary = _summarize(
        all_task_rows,
        seed=args.bootstrap_seed,
        repetitions=args.bootstrap_repetitions,
    )
    _write_csv(output_dir / "ablation_source_summary.csv", source_rows)
    _write_csv(output_dir / "ablation_task_level.csv", all_task_rows)
    _write_csv(output_dir / "ablation_level_summary.csv", level_summary)
    _write_csv(output_dir / "ablation_collapsed_summary.csv", collapsed_summary)
    _write_csv(output_dir / "ablation_coverage.csv", coverage_rows)
    (output_dir / "ablation_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
