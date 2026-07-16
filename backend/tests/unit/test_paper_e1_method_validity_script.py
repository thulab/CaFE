from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "run_paper_e1_method_validity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_paper_e1_method_validity", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_output_uses_versioned_paper_experiment_directory():
    module = load_module()

    assert module.DEFAULT_OUTPUT_DIR.relative_to(module.REPO_ROOT).as_posix() == (
        "runtime/paper_exp/v1/E1_method_validity"
    )
    assert module.DEFAULT_ROUND_SEEDS[0] != module.DEFAULT_ROUND_SEEDS[1]


def test_dose_response_compares_independent_realized_means_to_canonical_targets():
    module = load_module()
    targets = [0.1, 0.2, 0.3, 0.5, 0.8]
    rows = [
        {
            "profile_id": "profile",
            "capability_id": "trend",
            "intensity": intensity,
            "canonical_target_feature": "trend_strength",
            "canonical_target_strength": targets[intensity - 1],
            "realized_features": {"trend_strength": targets[intensity - 1]},
        }
        for intensity in range(1, 6)
        for _ in range(4)
    ]
    artifact = {
        "canonical_intensity": {
            "capabilities": {
                "trend": {
                    "primary_feature": "trend_strength",
                    "target_values": targets,
                }
            }
        }
    }

    cells, summary = module.dose_response_analysis(rows, artifact)

    assert len(cells) == 5
    assert summary["all_passed"] is True
    assert summary["profile_checks"][0]["spearman"] == pytest.approx(1.0)
    assert summary["profile_checks"][0]["max_normalized_absolute_error"] == pytest.approx(0.0)


def test_capability_oracle_does_not_read_target_future():
    module = load_module()
    context = 168
    horizon = 24
    time = np.arange(context + horizon, dtype=float)
    history = np.sin(2 * np.pi * time[:context] / 24)[:, None]
    first = np.vstack([history, np.zeros((horizon, 1))])
    second = np.vstack([history, np.full((horizon, 1), 1000.0)])
    latent = {"periods": [24, 48, 12]}

    first_forecast = module.capability_oracle_forecast(
        capability_id="multi_seasonal",
        target=first,
        covariates=None,
        context_length=context,
        season_length=24,
        latent=latent,
    )
    second_forecast = module.capability_oracle_forecast(
        capability_id="multi_seasonal",
        target=second,
        covariates=None,
        context_length=context,
        season_length=24,
        latent=latent,
    )

    assert first_forecast.shape == (horizon, 1)
    assert np.allclose(first_forecast, second_forecast)


def test_cross_round_repetition_detects_exact_reuse():
    module = load_module()

    def item(round_index: int, index: int, values: list[float]):
        return {
            "row": {
                "profile_id": "profile",
                "capability_id": "trend",
                "intensity": 1,
                "round_index": round_index,
                "sample_index": index,
            },
            "target": np.asarray(values, dtype=float)[:, None],
        }

    clean = [
        item(1, 0, [0, 1, 2]),
        item(1, 1, [2, 1, 0]),
        item(2, 0, [0.5, 1.5, 2.5]),
        item(2, 1, [2.5, 1.5, 0.5]),
    ]
    duplicated = [*clean[:2], item(2, 0, [0, 1, 2]), clean[3]]

    _rows, clean_summary = module.repetition_analysis(clean)
    duplicated_rows, duplicated_summary = module.repetition_analysis(duplicated)

    assert clean_summary["all_passed"] is True
    assert duplicated_summary["all_passed"] is False
    assert duplicated_rows[0]["exact_duplicate_rate"] == pytest.approx(0.5)


def test_fixed_distribution_distances_separate_a_shifted_negative_control():
    module = load_module()
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(128, 3))
    holdout = rng.normal(size=(128, 3))
    shifted = holdout + 3.0
    bandwidth = module.reference_bandwidth(reference, seed=11)

    real_mmd = module.rbf_mmd(reference, holdout, bandwidth=bandwidth)
    shifted_mmd = module.rbf_mmd(reference, shifted, bandwidth=bandwidth)
    real_swd = module.sliced_wasserstein_fixed(reference, holdout, seed=11)
    shifted_swd = module.sliced_wasserstein_fixed(reference, shifted, seed=11)

    assert shifted_mmd > real_mmd
    assert shifted_swd > real_swd


def test_experiment_output_is_immutable(tmp_path):
    module = load_module()
    output = tmp_path / "E1"
    module.prepare_output_dir(output, allow_existing_empty=False)
    (output / "summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable"):
        module.prepare_output_dir(output, allow_existing_empty=True)
