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


def test_select_models_checks_target_and_covariate_limits():
    module = load_experiment_module()
    service_models = [
        {"model_id": "single", "state": "active", "forecast_limits": {"min_input_length": 16, "max_output_length": 720, "max_target_count": 1, "max_covariate_count": 50}},
        {"model_id": "multi", "state": "active", "forecast_limits": {"min_input_length": 16, "max_output_length": 720, "max_target_count": None, "max_covariate_count": 0}},
        {"model_id": "cov", "state": "active", "forecast_limits": {"min_input_length": 16, "max_output_length": 720, "max_target_count": 1, "max_covariate_count": 50}},
    ]

    selected, skipped = module.select_models(service_models, ["single", "multi", "cov"], requirements={"target_dim": 3, "covariate_dim": 0})

    assert [model["model_id"] for model in selected] == ["multi"]
    assert [item["reason"] for item in skipped] == ["target_dim_unsupported", "target_dim_unsupported"]

    selected, skipped = module.select_models(service_models, ["single", "multi", "cov"], requirements={"target_dim": 1, "covariate_dim": 2})

    assert [model["model_id"] for model in selected] == ["single", "cov"]
    assert skipped[0]["reason"] == "covariate_dim_unsupported"


def test_probe_samples_include_multi_target_and_covariate_request_shapes():
    module = load_experiment_module()
    multi_sample = module.generate_probe_samples(1, ["common_factor"])[0]
    cov_sample = module.generate_probe_samples(1, ["covariate_response"])[0]

    assert multi_sample.target_column_names == ["target_0", "target_1", "target_2"]
    assert module.forecast_target(multi_sample)["columns"] == ["time", "target_0", "target_1", "target_2"]
    assert len(module.forecast_target(multi_sample)["data"][0]) == 4

    assert cov_sample.target_column_names == ["target_0"]
    assert cov_sample.covariate_column_names == ["weather", "event"]
    assert module.forecast_covariates(cov_sample, history=True)["columns"] == ["time", "weather", "event"]
    assert module.forecast_covariates(cov_sample, history=False)["columns"] == ["time", "weather", "event"]


def test_hierarchical_coherence_extra_metric():
    module = load_experiment_module()
    original_window = (module.CONTEXT_LENGTH, module.HORIZON, module.SEASON_LENGTH)
    module.CONTEXT_LENGTH = 365
    module.HORIZON = 28
    module.SEASON_LENGTH = 7
    try:
        sample = module.generate_probe_samples(1, ["hierarchical_coherence"])[0]
        forecast = [[3.0, 1.0, 2.0] for _ in range(module.HORIZON)]
    finally:
        module.CONTEXT_LENGTH, module.HORIZON, module.SEASON_LENGTH = original_window

    metrics = module.extra_sample_metrics(sample, forecast)

    assert metrics == {"coherence_mae": 0.0}


def test_render_report_includes_model_status_and_tables():
    module = load_experiment_module()
    capabilities = ["regime_switching"]
    summaries = []
    for model_id in ("naive", "seasonal_naive", "Timer-3.5"):
        for capability_id in capabilities:
            for intensity in range(1, 6):
                summaries.append(
                    {
                        "model_id": model_id,
                        "capability_id": capability_id,
                        "intensity": intensity,
                        "difficulty": intensity,
                        "sample_count": 2,
                        "metrics": {"mae": float(intensity), "mase": float(intensity) / 2},
                        "features": {"trend_strength": 0.1},
                    }
                )
    summary = {
        "base_url": "http://127.0.0.1:10810",
        "context_length": 168,
        "horizon": 24,
        "season_length": 24,
        "sample_count_per_capability_intensity": 2,
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
    assert "MAE i1" in report
    assert "--capabilities regime_switching" in report
