#!/usr/bin/env python3
"""Cross-check derived paper tables against the copied remote stability summary."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: str | float, right: str | float, tolerance: float = 1e-11) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    checks: list[dict[str, object]] = []

    remote_models = {
        row["model_id"]: row
        for row in load_csv(HERE / "raw" / "remote_stability" / "model_overall_stability.csv")
    }
    derived_models = {
        row["model_id"]: row
        for row in load_csv(HERE / "tables" / "model_overall_stability_extended.csv")
    }
    for model, remote in remote_models.items():
        derived = derived_models[model]
        for remote_field, derived_field in [
            ("effect_mean", "effect_mean"),
            ("effect_std", "effect_sd"),
            ("effect_cv", "effect_cv"),
            ("rank_mean", "rank_mean"),
            ("rank_min", "rank_min"),
            ("rank_max", "rank_max"),
            ("effect_desired_direction_rate", "effect_desired_rate"),
        ]:
            assert close(remote[remote_field], derived[derived_field]), (
                model,
                remote_field,
                remote[remote_field],
                derived[derived_field],
            )
    checks.append({"check": "model_overall_matches_remote", "passed": True, "rows": 7})

    remote_cells = {
        (row["model_id"], row["capability_id"], row["capability_level"]): row
        for row in load_csv(HERE / "raw" / "remote_stability" / "effect_cell_stability.csv")
    }
    derived_cells = {
        (row["model_id"], row["capability_id"], row["capability_level"]): row
        for row in load_csv(HERE / "tables" / "effect_cell_seed_stability.csv")
    }
    assert remote_cells.keys() == derived_cells.keys()
    for key, remote in remote_cells.items():
        derived = derived_cells[key]
        for remote_field, derived_field in [
            ("mean", "mean"),
            ("std", "sd"),
            ("cv", "cv"),
            ("desired_direction_rate", "desired_rate"),
            ("mean_approximate_within_seed_task_se", "mean_approx_task_bootstrap_se"),
            ("seed_sd_to_task_se_ratio", "seed_sd_to_task_se_ratio"),
        ]:
            assert close(remote[remote_field], derived[derived_field]), (
                key,
                remote_field,
                remote[remote_field],
                derived[derived_field],
            )
    checks.append({"check": "effect_cells_match_remote", "passed": True, "rows": 280})

    summary = json.loads((HERE / "analysis_summary.json").read_text(encoding="utf-8"))
    remote_summary = json.loads(
        (HERE / "raw" / "remote_stability" / "stability_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert close(summary["ranking"]["kendall_w"], remote_summary["ranking_stability"]["kendall_w"])
    assert close(
        summary["seed_vs_task_uncertainty"]["median_ratio"],
        remote_summary["seed_variation_vs_task_uncertainty"]["median"],
    )
    checks.append({"check": "key_summary_statistics_match_remote", "passed": True})

    audit = json.loads((HERE / "remote_audit.json").read_text(encoding="utf-8"))
    assert len(audit["experiments"]) == 10
    for experiment in audit["experiments"]:
        assert experiment["suite_row_count"] == 392
        assert experiment["validation_accepted_count"] == 10
        assert experiment["validation_failure_count"] == 0
        assert experiment["inference_complete_count"] == 10
        assert experiment["inference_model_status_count"] == 70
        assert experiment["inference_failed_model_status_count"] == 0
        assert experiment["inference_failure_count"] == 0
        assert experiment["analysis_manifest_count"] == 10
        assert experiment["treatment_count"] == 239030
        assert experiment["input_ablation_count"] == 39495
        assert experiment["official_instance_count"] == 10528
    checks.append({"check": "remote_run_completeness", "passed": True, "experiments": 10})

    figure_files = sorted((HERE / "figures").glob("fig_*.pdf")) + sorted(
        (HERE / "figures").glob("fig_*.png")
    )
    assert len(figure_files) == 12
    assert all(path.stat().st_size > 1000 for path in figure_files)
    checks.append({"check": "figure_files_nonempty", "passed": True, "files": 12})

    output = {
        "schema_version": "cafe.paper_stability_verification.v1",
        "all_passed": True,
        "checks": checks,
    }
    (HERE / "verification.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
