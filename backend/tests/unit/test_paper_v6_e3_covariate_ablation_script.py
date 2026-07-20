import importlib.util
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
