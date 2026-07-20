import importlib.util
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT / "scripts/run_paper_v6_e3_covariate_ablation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_paper_v6_e3_covariate_ablation",
    SCRIPT_PATH,
)
assert SPEC and SPEC.loader
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


def master_sample() -> dict:
    length = 504 + 48
    target = np.linspace(-1.0, 1.0, length)[:, None]
    covariates = np.column_stack(
        [
            np.linspace(-2.0, 2.0, length),
            (np.arange(length) % 24 == 0).astype(float),
        ]
    )
    return {
        "sample_id": "sample",
        "master_sample_id": "sample",
        "dataset_id": "gefcom2014_load",
        "task_id": "covariate",
        "profile_id": "covariate_response",
        "capability_id": "covariate_response",
        "intensity": 3,
        "round_index": 0,
        "round_seed": 7,
        "sample_index": 4,
        "paired_group_id": "group",
        "target": target.tolist(),
        "target_dim": 1,
        "covariates": covariates.tolist(),
        "covariate_dim": 2,
        "context_lengths": [96, 168, 336, 504],
        "horizon": 48,
        "season_length": 24,
        "frequency": "h",
        "hierarchy": None,
        "future_sha256": ablation.array_sha256(target[504:]),
    }


def test_build_ablation_view_preserves_history_and_zeros_only_future():
    master = master_sample()
    intact = ablation.e2.master_view(master, 168)
    view = ablation.build_ablation_view(master, context_length=168)

    intact_covariates = np.asarray(intact["covariates"], dtype=float)
    ablated_covariates = np.asarray(view["covariates"], dtype=float)
    assert np.array_equal(
        ablated_covariates[:168],
        intact_covariates[:168],
    )
    assert np.array_equal(
        ablated_covariates[168:],
        np.zeros((48, 2)),
    )
    assert view["ablation"] == "future_covariates_zero"
    assert view["master_sample_id"] == "sample"


def test_ablation_prediction_row_retains_pairing_contract():
    view = ablation.build_ablation_view(
        master_sample(),
        context_length=336,
    )
    forecast = np.zeros((48, 1))

    row = ablation.ablation_prediction_row(
        "model",
        "timer_service",
        view,
        forecast,
    )

    assert row["model_id"] == "model"
    assert row["master_sample_id"] == "sample"
    assert row["context_length"] == 336
    assert row["ablation"] == "future_covariates_zero"
    assert np.asarray(row["forecast"]).shape == (48, 1)


def test_counterfactual_partition_uses_intact_input_adaptation():
    predictions = {
        "native": {
            "input_adaptation": {
                "policy_id": "paper-v7-input-adaptation-v1",
                "adapted": False,
                "target_mode": "native_univariate",
                "covariate_mode": "native",
            }
        },
        "omitted": {
            "input_adaptation": {
                "policy_id": "paper-v7-input-adaptation-v1",
                "adapted": True,
                "target_mode": "native_univariate",
                "covariate_mode": "omitted_unsupported",
            }
        },
    }

    native, reused = ablation.partition_counterfactual_samples(
        predictions
    )

    assert native == {"native"}
    assert reused == {"omitted"}


def test_covariate_omitted_model_reuses_intact_forecast_with_zero_http():
    master = master_sample()
    forecast = np.full((48, 1), 1.25).tolist()
    plan = {
        "policy_id": "paper-v7-input-adaptation-v1",
        "adapted": True,
        "target_mode": "native_univariate",
        "covariate_mode": "omitted_unsupported",
    }

    rows = ablation.reused_counterfactual_rows(
        model_id="model",
        samples={"sample": master},
        oracle_selection={"sample": {"oracle_context": 168}},
        intact_predictions={
            "sample": {
                "forecast": forecast,
                "model_group": "timer_service",
                "input_adaptation": plan,
            }
        },
        master_ids={"sample"},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["forecast"] == forecast
    assert row["counterfactual_mode"] == (
        "reuse_intact_forecast_covariates_omitted"
    )
    assert row["counterfactual_http_request_count"] == 0
    assert row["counterfactual_effect_mae"] == 0.0
    assert row["intact_input_adaptation"] == plan


def test_native_counterfactual_keeps_http_and_adaptation_provenance(
    tmp_path,
):
    path = tmp_path / "native.jsonl"
    path.write_text(
        json.dumps(
            {
                "master_sample_id": "sample",
                "forecast": [[0.0], [0.5]],
                "successful_http_request_count": 3,
                "input_adaptation": {
                    "target_mode": "independent_univariate",
                    "covariate_mode": "native",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    intact_plan = {
        "target_mode": "independent_univariate",
        "covariate_mode": "native",
    }

    rows = ablation.native_counterfactual_rows(
        path,
        intact_predictions={
            "sample": {
                "forecast": [[1.0], [1.5]],
                "input_adaptation": intact_plan,
            }
        },
    )

    assert rows[0]["counterfactual_mode"] == (
        "native_future_covariate_ablation_http"
    )
    assert rows[0]["counterfactual_http_request_count"] == 3
    assert rows[0]["counterfactual_effect_mae"] == 1.0
    assert rows[0]["intact_input_adaptation"] == intact_plan
