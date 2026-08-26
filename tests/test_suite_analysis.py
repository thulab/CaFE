from __future__ import annotations

from pathlib import Path

from cafe import core as protocol
from cafe.benchmark_extension.analysis import aggregate_analysis_tasks


def _task_analysis(
    root: Path,
    task_id: str,
    *,
    left_mase: float,
    right_mase: float,
) -> None:
    task_root = root / task_id
    generation = task_root / "01_generation" / "manifest.json"
    analysis_dir = task_root / "04_analysis"
    protocol.write_json(
        generation,
        {
            "config": {
                "benchmark_id": "fev_bench",
                "suite_id": "mini20",
                "task_id": task_id,
            }
        },
    )
    accuracy = analysis_dir / "official_accuracy.json"
    effects = analysis_dir / "capability_effect_summary.json"
    ablations = analysis_dir / "input_ablation_summary.json"
    protocol.write_json(
        accuracy,
        {
            "models": [
                {"model_id": "left", "official_mase_mean": left_mase},
                {"model_id": "right", "official_mase_mean": right_mase},
            ]
        },
    )
    protocol.write_json(effects, {"rows": []})
    protocol.write_json(ablations, {"rows": []})
    protocol.write_json(
        analysis_dir / "manifest.json",
        {
            "files": {
                "official_accuracy": {"path": str(accuracy)},
                "capability_effect_summary": {"path": str(effects)},
                "input_ablation_summary": {"path": str(ablations)},
            }
        },
    )


def test_suite_analysis_is_task_equal_and_pairwise(tmp_path: Path) -> None:
    _task_analysis(tmp_path, "large", left_mase=1.0, right_mase=2.0)
    _task_analysis(tmp_path, "small", left_mase=3.0, right_mase=5.0)

    manifest = aggregate_analysis_tasks(
        tmp_path,
        ["large", "small"],
        bootstrap_seed=3,
        bootstrap_repetitions=100,
    )
    summary = protocol.read_json(
        Path(str(manifest["files"]["task_equal_summary"]["path"]))
    )
    rows = {row["model_id"]: row for row in summary["rows"]}
    assert rows["left"]["task_equal_mean"] == 2.0
    assert rows["right"]["task_equal_mean"] == 3.5
    pair = summary["paired_model_comparisons"][0]
    assert pair["paired_task_count"] == 2
    assert pair["task_equal_mean_difference_left_minus_right"] == -1.5
