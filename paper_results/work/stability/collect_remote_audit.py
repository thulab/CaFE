#!/usr/bin/env python3
"""Read-only audit of the ten remote CaFE stability experiments.

The remote Python program only opens JSON/Parquet metadata. It does not create,
modify, or delete any remote artifact. The resulting audit snapshot is written
under this local work directory.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
HOST = os.environ.get("CAFE_STABILITY_HOST", "timecho92")
REMOTE_ROOT = os.environ.get("CAFE_STABILITY_REMOTE_ROOT", "/data/xmy/CaFE")
SEEDS = list(range(2026082701, 2026082711))


REMOTE_PROGRAM = r'''
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
seeds = [int(value) for value in sys.argv[2:]]
experiments = root / "runtime" / "experiments"
prefix = "gift-v15-short-stability10-head78ef32f-seed"

audit = {
    "remote_root": str(root),
    "seeds": seeds,
    "experiments": [],
    "representative_parquet_schemas": [],
}

for seed in seeds:
    experiment = experiments / f"{prefix}{seed}"
    suite_path = experiment / "04_analysis_suite" / "task_equal_summary.json"
    suite = json.loads(suite_path.read_text())
    task_ids = list(suite["task_ids"])
    metrics = sorted({row["metric"] for row in suite["rows"]})
    models = sorted({row["model_id"] for row in suite["rows"]})
    capabilities = sorted(
        {row["capability_id"] for row in suite["rows"] if row["capability_id"]}
    )
    levels = sorted(
        {row["capability_level"] for row in suite["rows"] if row["capability_level"]}
    )

    validation_accepted = 0
    validation_failures = 0
    inference_complete = 0
    inference_model_status_count = 0
    inference_failed_model_status_count = 0
    inference_failure_count = 0
    analysis_manifest_count = 0
    generation_manifest_count = 0
    treatment_count = 0
    input_ablation_count = 0
    official_instance_count = 0

    for task_id in task_ids:
        task_root = experiment / task_id
        generation = json.loads(
            (task_root / "01_generation" / "manifest.json").read_text()
        )
        generation_manifest_count += 1
        treatment_count += int(generation.get("treatment_count", 0))
        input_ablation_count += int(generation.get("input_ablation_count", 0))
        official_instance_count += int(generation.get("official_instance_count", 0))

        validation = json.loads(
            (task_root / "02_validation" / "report.json").read_text()
        )
        validation_accepted += int(bool(validation.get("accepted")))
        validation_failures += int(validation.get("failure_count", 0))

        inference = json.loads(
            (task_root / "03_inference" / "manifest.json").read_text()
        )
        inference_complete += int(bool(inference.get("complete")))
        statuses = inference.get("model_statuses", [])
        inference_model_status_count += len(statuses)
        inference_failed_model_status_count += sum(
            status.get("status") != "complete" for status in statuses
        )
        inference_failure_count += sum(
            int(status.get("failure_count", 0)) for status in statuses
        )

        analysis_manifest_count += int(
            (task_root / "04_analysis" / "manifest.json").is_file()
        )

    counts_by_metric = {}
    for metric in metrics:
        counts_by_metric[metric] = sum(
            row["metric"] == metric for row in suite["rows"]
        )

    audit["experiments"].append(
        {
            "seed": seed,
            "experiment_id": experiment.name,
            "suite_schema_version": suite.get("schema_version"),
            "suite_row_count": len(suite["rows"]),
            "counts_by_metric": counts_by_metric,
            "task_count": len(task_ids),
            "task_ids": task_ids,
            "model_count": len(models),
            "models": models,
            "capability_count": len(capabilities),
            "capabilities": capabilities,
            "levels": levels,
            "generation_manifest_count": generation_manifest_count,
            "validation_accepted_count": validation_accepted,
            "validation_failure_count": validation_failures,
            "inference_complete_count": inference_complete,
            "inference_model_status_count": inference_model_status_count,
            "inference_failed_model_status_count": inference_failed_model_status_count,
            "inference_failure_count": inference_failure_count,
            "analysis_manifest_count": analysis_manifest_count,
            "treatment_count": treatment_count,
            "input_ablation_count": input_ablation_count,
            "official_instance_count": official_instance_count,
        }
    )

representative = experiments / f"{prefix}{seeds[0]}" / audit["experiments"][0]["task_ids"][0]
parquet_paths = {
    "availability": representative / "01_generation" / "availability.parquet",
    "treatment_contracts": representative / "01_generation" / "treatment_contracts.parquet",
    "input_ablation_contracts": representative / "01_generation" / "input_ablation_contracts.parquet",
    "accuracy_rows": representative / "04_analysis" / "accuracy_rows.parquet",
    "capability_effect_rows": representative / "04_analysis" / "capability_effect_rows.parquet",
    "input_ablation_rows": representative / "04_analysis" / "input_ablation_rows.parquet",
}
for artifact, path in parquet_paths.items():
    parquet = pq.ParquetFile(path)
    audit["representative_parquet_schemas"].append(
        {
            "artifact": artifact,
            "path": str(path),
            "row_count": parquet.metadata.num_rows,
            "row_group_count": parquet.metadata.num_row_groups,
            "fields": [
                {"name": field.name, "type": str(field.type)}
                for field in parquet.schema_arrow
            ],
        }
    )

print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
'''


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    command = [
        "ssh",
        HOST,
        f"{REMOTE_ROOT}/.venv/bin/python",
        "-",
        REMOTE_ROOT,
        *[str(seed) for seed in SEEDS],
    ]
    result = subprocess.run(
        command,
        input=REMOTE_PROGRAM,
        text=True,
        capture_output=True,
        check=True,
    )
    audit = json.loads(result.stdout)

    snapshot_path = HERE / "remote_audit.json"
    snapshot_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    coverage_rows = []
    for row in audit["experiments"]:
        coverage_rows.append(
            {
                "seed": row["seed"],
                "experiment_id": row["experiment_id"],
                "suite_rows": row["suite_row_count"],
                "effect_cells": row["counts_by_metric"]["capability_effect_nrmse"],
                "ablation_cells": row["counts_by_metric"]["input_ablation_mase_degradation"],
                "official_cells": row["counts_by_metric"]["official_mase"],
                "tasks": row["task_count"],
                "models": row["model_count"],
                "capabilities": row["capability_count"],
                "levels": "|".join(map(str, row["levels"])),
                "generation_manifests": row["generation_manifest_count"],
                "validations_accepted": row["validation_accepted_count"],
                "validation_failures": row["validation_failure_count"],
                "inference_manifests_complete": row["inference_complete_count"],
                "model_statuses": row["inference_model_status_count"],
                "failed_model_statuses": row["inference_failed_model_status_count"],
                "inference_failures": row["inference_failure_count"],
                "analysis_manifests": row["analysis_manifest_count"],
                "treatments": row["treatment_count"],
                "input_ablations": row["input_ablation_count"],
                "official_instances": row["official_instance_count"],
            }
        )
    write_csv(HERE / "tables" / "remote_run_coverage.csv", coverage_rows)

    schema_rows = []
    for artifact in audit["representative_parquet_schemas"]:
        for position, field in enumerate(artifact["fields"]):
            schema_rows.append(
                {
                    "artifact": artifact["artifact"],
                    "representative_row_count": artifact["row_count"],
                    "row_groups": artifact["row_group_count"],
                    "field_position": position,
                    "field_name": field["name"],
                    "field_type": field["type"],
                }
            )
    write_csv(HERE / "tables" / "artifact_schemas.csv", schema_rows)
    print(f"Wrote {snapshot_path}")


if __name__ == "__main__":
    main()
