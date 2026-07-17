from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).parents[3] / "scripts" / "run_paper_e3_model_capability_profiles.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_paper_e3_model_capability_profiles", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def observation_rows(
    *,
    model_id: str,
    model_group: str,
    profile_id: str,
    mase: float,
    abs_error_sum: float,
    target_abs_sum: float,
    coherence_abs_sum: float = np.nan,
    coherence_point_count: float = np.nan,
    parent_abs_sum: float = np.nan,
) -> list[dict]:
    return [
        {
            "model_id": model_id,
            "model_group": model_group,
            "sample_id": f"{model_id}-{profile_id}-{index}",
            "profile_id": profile_id,
            "capability_id": "trend",
            "intensity": 1,
            "round_index": 1,
            "sample_index": index,
            "mase": mase,
            "mae": abs_error_sum / 2,
            "abs_error_sum": abs_error_sum,
            "target_abs_sum": target_abs_sum,
            "future_point_count": 2,
            "coherence_abs_sum": coherence_abs_sum,
            "coherence_point_count": coherence_point_count,
            "parent_abs_sum": parent_abs_sum,
        }
        for index in range(2)
    ]


def test_default_paths_and_frozen_source_hash():
    module = load_module()

    assert module.DEFAULT_OUTPUT_DIR.relative_to(module.REPO_ROOT).as_posix() == (
        "runtime/paper_exp/v1/E3_model_capability_profiles"
    )
    assert len(module.CAPABILITY_ORDER) == 9
    assert len(module.UNIVARIATE_CAPABILITIES) == 6
    assert module.DEFAULT_BOOTSTRAP_REPLICATES == 2_000
    assert module.sha256_file(module.DEFAULT_SOURCE_DIR / "manifest.json") == (
        module.EXPECTED_E2_MANIFEST_SHA256
    )


def test_intensity_summary_does_not_assume_level_five_is_worst():
    module = load_module()
    values = np.asarray([1.0, 4.0, 2.0, 3.0, 0.5])

    assert module.normalized_auc(values) == pytest.approx(2.4375)
    assert int(np.argmax(values)) + 1 == 2
    assert module.relative_change(values[0], values[-1]) == pytest.approx(-0.5)
    assert module.linear_intensity_slope([0, 1, 2, 3, 4]) == pytest.approx(4.0)
    assert module.spearman_five_levels([0, 1, 2, 3, 4]) == pytest.approx(1.0)


def test_relative_skill_is_computed_per_bucket_before_macro_average():
    module = load_module()
    rows = []
    rows += observation_rows(
        model_id="Timer-3.5",
        model_group="timer_service",
        profile_id="p1",
        mase=0.5,
        abs_error_sum=1.0,
        target_abs_sum=2.0,
    )
    rows += observation_rows(
        model_id="seasonal_naive",
        model_group="baseline",
        profile_id="p1",
        mase=1.0,
        abs_error_sum=2.0,
        target_abs_sum=2.0,
    )
    rows += observation_rows(
        model_id="Timer-3.5",
        model_group="timer_service",
        profile_id="p2",
        mase=1.5,
        abs_error_sum=3.0,
        target_abs_sum=2.0,
    )
    rows += observation_rows(
        model_id="seasonal_naive",
        model_group="baseline",
        profile_id="p2",
        mase=2.0,
        abs_error_sum=4.0,
        target_abs_sum=2.0,
    )

    cells = module.profile_intensity_score_frame(pd.DataFrame(rows))
    curve = module.intensity_curve_frame(cells)

    assert list(cells["seasonal_naive_skill_mase"]) == pytest.approx([0.5, 0.25])
    assert curve.iloc[0]["seasonal_naive_skill_mase"] == pytest.approx(0.375)
    assert curve.iloc[0]["seasonal_naive_skill_mase"] != pytest.approx(
        1.0 - cells["mase_mean"].mean() / cells["seasonal_naive_mase_mean"].mean()
    )


def test_nmae_pools_absolute_error_and_target_denominator_within_cell():
    module = load_module()
    rows = observation_rows(
        model_id="Timer-3.5",
        model_group="timer_service",
        profile_id="p1",
        mase=0.5,
        abs_error_sum=1.0,
        target_abs_sum=2.0,
    )
    rows[1]["abs_error_sum"] = 9.0
    rows[1]["target_abs_sum"] = 18.0
    rows += observation_rows(
        model_id="seasonal_naive",
        model_group="baseline",
        profile_id="p1",
        mase=1.0,
        abs_error_sum=10.0,
        target_abs_sum=20.0,
    )

    cells = module.profile_intensity_score_frame(pd.DataFrame(rows))

    assert cells.iloc[0]["nmae_abs"] == pytest.approx(10.0 / 20.0)
    assert cells.iloc[0]["seasonal_naive_nmae_abs"] == pytest.approx(20.0 / 40.0)


def test_hierarchical_cell_reports_accuracy_and_coherence_separately():
    module = load_module()
    rows = []
    for model_id, model_group, coherence in [
        ("Chronos-2", "timer_service", 1.0),
        ("seasonal_naive", "baseline", 0.0),
    ]:
        for index in range(2):
            rows.append(
                {
                    "model_id": model_id,
                    "model_group": model_group,
                    "sample_id": f"{model_id}-{index}",
                    "profile_id": "hierarchy",
                    "capability_id": "hierarchical_coherence",
                    "intensity": 1,
                    "round_index": 1,
                    "sample_index": index,
                    "mase": 0.5 if model_group == "timer_service" else 1.0,
                    "mae": 0.5,
                    "abs_error_sum": 3.0,
                    "target_abs_sum": 6.0,
                    "future_point_count": 6,
                    "coherence_abs_sum": coherence,
                    "coherence_point_count": 2.0,
                    "parent_abs_sum": 4.0,
                }
            )

    cells = module.profile_intensity_score_frame(pd.DataFrame(rows))

    assert cells.iloc[0]["mase_mean"] == pytest.approx(0.5)
    assert cells.iloc[0]["coherence_mae"] == pytest.approx(0.5)
    assert cells.iloc[0]["coherence_nmae"] == pytest.approx(0.25)


def test_paired_hierarchical_bootstrap_is_deterministic():
    module = load_module()
    values = np.asarray([[1.0, 1.0], [3.0, 3.0]])
    rng = np.random.default_rng(17)
    round_draws = rng.integers(0, 2, size=(500, 2))
    sample_draws = rng.integers(0, 2, size=(500, 2, 2))

    first = module.bootstrap_mean(values, round_draws, sample_draws)
    second = module.bootstrap_mean(values, round_draws, sample_draws)

    assert np.array_equal(first, second)
    assert np.mean(first) == pytest.approx(2.0, abs=0.1)
    assert set(np.unique(first)).issubset({1.0, 2.0, 3.0})


def test_source_manifest_verification_detects_tampering(tmp_path):
    module = load_module()
    source = tmp_path / "E2"
    source.mkdir()
    (source / "data.txt").write_text("original", encoding="utf-8")
    manifest = {
        "experiment_id": "E2_dynamic_stability",
        "files": {
            "data.txt": {
                "bytes": 8,
                "sha256": module.sha256_file(source / "data.txt"),
            }
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    module.verify_e2_source(source)
    (source / "data.txt").write_text("tampered", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        module.verify_e2_source(source)


def test_output_is_immutable_and_figures_have_three_formats(tmp_path):
    module = load_module()
    output = tmp_path / "E3"
    module.prepare_output_dir(output, allow_existing_empty=False)
    (output / "summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable"):
        module.prepare_output_dir(output, allow_existing_empty=True)

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    stem = tmp_path / "figure"
    module.save_figure(figure, stem)
    assert stem.with_suffix(".png").is_file()
    assert stem.with_suffix(".svg").is_file()
    assert stem.with_suffix(".pdf").is_file()
