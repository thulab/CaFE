from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import paper_v2_transfer_common as transfer  # noqa: E402
import predictive_capability_gate as gate  # noqa: E402
import run_paper_v3_e4_mechanism_gated_transfer as e4v3  # noqa: E402


def test_pseudo_future_origins_are_nonoverlapping_and_inside_history() -> None:
    origins = gate.pseudo_future_origins(
        504,
        season_length=24,
        pseudo_horizon=48,
        fold_count=4,
    )

    assert origins == (312, 360, 408, 456)
    assert origins[-1] + 48 == 504


def test_gate_api_cannot_read_a_benchmark_future() -> None:
    time = np.arange(504, dtype=float)
    history = (
        np.sin(2 * np.pi * time / 24)
        + 0.8 * np.sin(2 * np.pi * time / 48 + 0.2)
        + 0.3 * np.sin(2 * np.pi * time / 12 - 0.4)
    )

    first = gate.evaluate_capability_gate(
        history,
        capability_id="multi_seasonal",
        season_length=24,
        pseudo_horizon=48,
    )
    second = gate.evaluate_capability_gate(
        history.copy(),
        capability_id="multi_seasonal",
        season_length=24,
        pseudo_horizon=48,
    )

    assert first == second
    assert first["uses_benchmark_future"] is False
    assert first["pooled_relative_mae_gain"] > 0.5


def test_periodic_pulse_probe_has_phase_specific_headroom() -> None:
    time = np.arange(504, dtype=float)
    distance = np.abs(np.mod(time - 12 + 12, 24) - 12)
    history = 2.5 * np.exp(-0.5 * (distance / 0.7) ** 2)
    history += 0.05 * np.sin(2 * np.pi * time / 24)

    diagnostics = gate.evaluate_capability_gate(
        history,
        capability_id="predictable_intermittency",
        season_length=24,
        pseudo_horizon=48,
    )

    assert diagnostics["positive_fold_fraction"] == pytest.approx(1.0)
    assert diagnostics["phase_permutation_pvalue"] <= 0.1
    assert diagnostics["pooled_relative_mae_gain"] > 0.5


def test_frozen_gate_decision_requires_every_component() -> None:
    thresholds = {
        "gain_statistic": "pooled_relative_mae_gain",
        "minimum_predictive_gain": 0.1,
        "minimum_positive_fold_fraction": 0.75,
        "minimum_support": 0.2,
        "minimum_parameter_stability": 0.75,
        "maximum_phase_permutation_pvalue": 0.1,
        "minimum_valid_folds": 3,
        "gain_normalization_scale": 0.2,
        "support_normalization_scale": 0.4,
    }
    diagnostics = {
        "valid_fold_count": 4,
        "pooled_relative_mae_gain": 0.3,
        "positive_fold_fraction": 1.0,
        "support_median": 0.5,
        "parameter_stability": 1.0,
        "phase_permutation_pvalue": 0.05,
    }

    accepted = gate.gate_decision(diagnostics, thresholds)
    rejected = gate.gate_decision(
        {**diagnostics, "phase_permutation_pvalue": 0.2},
        thresholds,
    )

    assert accepted["qualified"] is True
    assert accepted["fingerprint_weight"] > 0
    assert rejected["qualified"] is False
    assert rejected["checks"]["phase_permutation"] is False


def test_validation_candidate_precedes_the_official_test_tail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    values = np.arange(1200, dtype=float)
    record = {
        "item_id": "series",
        "start": datetime(2020, 1, 1),
        "target": values,
    }
    monkeypatch.setattr(
        e4v3.e4v2,
        "read_gift_records",
        lambda _path: ("H", [record]),
    )
    monkeypatch.setattr(
        e4v3,
        "gift_eval_short_term_test_holdout_steps",
        lambda _frequency, _records: 96,
    )
    spec = transfer.TRANSFER_PROFILE_SPECS[0]

    tasks, summary = e4v3.build_profile_validation_candidates(
        spec,
        gift_eval_dir=tmp_path,
    )

    assert len(tasks) == 1
    assert tasks[0]["source_origin"] == 1200 - 96 - e4v3.HORIZON
    assert tasks[0]["origin_role"] == "gift_validation_horizon"
    assert tasks[0]["target"][e4v3.CONTEXT_LENGTH][0] == pytest.approx(1056)
    assert summary["validation_origin_policy"].startswith("series_length")


def test_real_cell_scores_use_capability_specific_task_sets() -> None:
    observations = pd.DataFrame.from_records(
        [
            {
                "model_id": model_id,
                "model_group": group,
                "sample_id": sample_id,
                "profile_id": "profile",
                "dataset_name": "dataset",
                "family_id": "family",
                "series_id": sample_id,
                "mase": mase,
                "mae": mase,
            }
            for model_id, group, values in (
                ("seasonal_naive", "baseline", (1.0, 1.0)),
                ("Timer-3.5", "timer_service", (0.5, 1.5)),
            )
            for sample_id, mase in zip(("one", "two"), values, strict=True)
        ]
    )
    mapping = pd.DataFrame.from_records(
        [
            {
                "selection_scope": "inclusive",
                "sample_id": "one",
                "profile_id": "profile",
                "dataset_name": "dataset",
                "family_id": "family",
                "series_id": "one",
                "capability_id": "trend",
                "fingerprint_weight": 1.0,
            },
            {
                "selection_scope": "inclusive",
                "sample_id": "two",
                "profile_id": "profile",
                "dataset_name": "dataset",
                "family_id": "family",
                "series_id": "two",
                "capability_id": "multi_seasonal",
                "fingerprint_weight": 1.0,
            },
        ]
    )
    original_models = e4v3.MODELS
    e4v3.MODELS = ("Timer-3.5",)
    try:
        scores = e4v3.real_cell_score_frame(
            observations,
            cell_map=mapping,
            selection_scope="inclusive",
            weighted=False,
        ).set_index("capability_id")
    finally:
        e4v3.MODELS = original_models

    assert scores.loc["trend", "mase_mean"] == pytest.approx(0.5)
    assert scores.loc["multi_seasonal", "mase_mean"] == pytest.approx(1.5)
    assert scores.loc["trend", "real_log_mase_ratio"] == pytest.approx(
        np.log(0.5)
    )


def test_committed_gate_freeze_is_valid_json() -> None:
    artifact = json.loads(e4v3.GATE_FREEZE_PATH.read_text(encoding="utf-8"))

    validated = e4v3.validated_capability_ids(artifact)

    assert len(validated) == 5
    assert "nonlinear_persistence" not in validated
