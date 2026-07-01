from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "run_synthetic_v2_real_model_experiment.py"


def load_experiment_module():
    repo_root = SCRIPT_PATH.parents[1]
    for path in (repo_root / "backend", repo_root / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("run_synthetic_v2_real_model_experiment", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_models_skips_inactive_and_unknown():
    module = load_experiment_module()
    service_models = [
        {"model_id": "Timer-3.5", "state": "active", "forecast_limits": {"min_input_length": 16, "max_output_length": 720}},
        {"model_id": "timesfm2.5", "state": "inactive", "forecast_limits": {"min_input_length": 16, "max_output_length": 720}},
    ]

    selected, skipped = module.select_models(service_models, ["Timer-3.5", "timesfm2.5", "missing"])

    assert [model["model_id"] for model in selected] == ["Timer-3.5"]
    assert skipped == [{"model_id": "timesfm2.5", "reason": "inactive"}, {"model_id": "missing", "reason": "not_registered"}]


def test_render_report_includes_model_status_and_tables():
    module = load_experiment_module()
    capabilities = ["regime_switching"]
    summaries = []
    for model_id in ("naive", "seasonal_naive", "Timer-3.5"):
        for capability_id in capabilities:
            for difficulty in range(1, 6):
                summaries.append(
                    {
                        "model_id": model_id,
                        "capability_id": capability_id,
                        "difficulty": difficulty,
                        "sample_count": 2,
                        "metrics": {"mae": float(difficulty), "mase": float(difficulty) / 2},
                        "features": {"trend_strength": 0.1},
                    }
                )
    summary = {
        "base_url": "http://127.0.0.1:10810",
        "context_length": 168,
        "horizon": 24,
        "season_length": 24,
        "sample_count_per_capability_difficulty": 2,
        "batch_size": 6,
        "requested_models": ["Timer-3.5", "timesfm2.5"],
        "requested_capabilities": capabilities,
        "selected_models": ["Timer-3.5"],
        "skipped_models": [{"model_id": "timesfm2.5", "reason": "inactive"}],
        "model_run_status": [{"model_id": "Timer-3.5", "status": "succeeded", "failed_count": 0, "elapsed_seconds": 1.2}],
        "summaries": summaries,
        "comparisons": module.build_comparisons(summaries),
        "reproduction_command": module.reproduction_command(["Timer-3.5", "timesfm2.5"], capabilities, 2, 6),
    }

    report = module.render_report(summary, output_dir=Path("runtime/out"))

    assert "Synthetic v2 真实模型响应实验" in report
    assert "Timer-3.5" in report
    assert "timesfm2.5 (inactive)" in report
    assert "`regime_switching`" in report
    assert "--capabilities regime_switching" in report
