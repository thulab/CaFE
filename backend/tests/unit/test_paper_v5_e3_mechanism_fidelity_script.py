import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT / "scripts/analyze_paper_v5_e3_mechanism_fidelity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_paper_v5_e3_mechanism_fidelity",
    SCRIPT_PATH,
)
assert SPEC and SPEC.loader
e3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e3)


def sample_score_frame() -> pd.DataFrame:
    rows = []
    for model_index, model_id in enumerate(("strong", "weak")):
        for paired_group in ("g0", "g1"):
            for intensity in range(1, 6):
                truth_strength = float(intensity)
                forecast_strength = (
                    truth_strength
                    if model_id == "strong"
                    else float(6 - intensity)
                )
                mechanism = 0.9 if model_id == "strong" else 0.3
                rows.append(
                    {
                        "model_id": model_id,
                        "dataset_id": "dataset",
                        "task_id": "univariate",
                        "capability_id": "trend",
                        "intensity": intensity,
                        "paired_group_id": paired_group,
                        "round_index": 0,
                        "sample_index": int(paired_group[-1]),
                        "master_sample_id": (
                            f"{model_id}-{paired_group}-{intensity}"
                        ),
                        "oracle_mase": 0.5 + model_index,
                        "mechanism_fidelity_score": mechanism,
                        "detection_score": mechanism,
                        "timing_score": mechanism,
                        "magnitude_score": mechanism,
                        "selectivity_score": mechanism,
                        "truth_mechanism_strength": truth_strength,
                        "forecast_mechanism_strength": forecast_strength,
                        "formal_score_eligible": True,
                    }
                )
    return pd.DataFrame(rows)


def oracle_score_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": "naive",
                "dataset_id": "dataset",
                "task_id": "univariate",
                "capability_id": "trend",
                "intensity": intensity,
                "oracle_mase": 1.0,
            }
            for intensity in range(1, 6)
            for _ in range(2)
        ]
    )


def test_dose_response_rewards_correct_intensity_order():
    scores = e3.paired_dose_response_scores(sample_score_frame())
    strong = scores[scores["model_id"] == "strong"]
    weak = scores[scores["model_id"] == "weak"]

    assert np.allclose(strong["dose_spearman_rho"], 1.0)
    assert np.allclose(strong["dose_response_score"], 1.0)
    assert np.allclose(weak["dose_spearman_rho"], -1.0)
    assert np.allclose(weak["dose_response_score"], 0.0)


def test_profiles_keep_point_mechanism_and_ability_ranks():
    samples = sample_score_frame()
    cells = e3.intensity_cell_scores(samples, oracle_score_frame())
    dose = e3.paired_dose_response_scores(samples)
    profiles = e3.capability_profiles(cells, dose)

    strong = profiles.loc[profiles["model_id"] == "strong"].iloc[0]
    weak = profiles.loc[profiles["model_id"] == "weak"].iloc[0]
    assert strong["mechanism_fidelity_score"] == pytest.approx(0.93)
    assert strong["ability_score"] == pytest.approx(0.93)
    assert weak["mechanism_fidelity_score"] == pytest.approx(0.21)
    assert weak["ability_score"] == pytest.approx(0.14)
    assert strong["mase_rank"] == 1
    assert strong["mechanism_rank"] == 1
    assert strong["ability_rank"] == 1


def test_covariate_diagnostic_rows_do_not_receive_formal_rank():
    samples = sample_score_frame()
    samples["capability_id"] = "covariate_response"
    samples["task_id"] = "covariate"
    samples["formal_score_eligible"] = False
    oracle = oracle_score_frame()
    oracle["capability_id"] = "covariate_response"
    oracle["task_id"] = "covariate"

    cells = e3.intensity_cell_scores(samples, oracle)
    dose = e3.paired_dose_response_scores(samples)
    profiles = e3.capability_profiles(cells, dose)

    assert not profiles["formal_score_eligible"].any()
    assert profiles["mechanism_rank"].isna().all()
    assert profiles["ability_rank"].isna().all()


def test_covariate_ablation_loader_requires_matching_oracle_context(tmp_path):
    path = tmp_path / "model.jsonl"
    path.write_text(
        json.dumps(
            {
                "master_sample_id": "sample",
                "context_length": 168,
                "forecast": [[0.0], [0.0]],
                "ablation": "future_covariates_zero",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples = {
        "sample": {
            "capability_id": "covariate_response",
        }
    }
    oracle = {
        "model": {
            "sample": {
                "oracle_context": 168,
            }
        }
    }

    result = e3.load_covariate_ablation_predictions(
        tmp_path,
        model_ids=["model"],
        samples=samples,
        oracle_selections=oracle,
    )

    assert result["model"]["sample"]["forecast"] == [[0.0], [0.0]]


def test_covariate_ablation_input_record_freezes_manifest_and_files(
    tmp_path,
):
    (tmp_path / "model.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "ablation": "future_covariates_zero",
                "models": ["model"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    record = e3.covariate_ablation_input_record(
        tmp_path,
        counterfactual_predictions={"model": {"sample": {}}},
    )

    assert record is not None
    assert len(record["manifest_sha256"]) == 64
    assert len(record["files"]["model.jsonl"]["sha256"]) == 64


def test_model_capability_coverage_keeps_unsupported_as_na():
    samples = {
        "u": {
            "dataset_id": "dataset",
            "task_id": "univariate",
            "capability_id": "trend",
        },
        "m": {
            "dataset_id": "panel",
            "task_id": "common_factor",
            "capability_id": "common_factor",
        },
    }
    selections = {
        "univariate-only": {
            "u": {
                "dataset_id": "dataset",
                "task_id": "univariate",
                "capability_id": "trend",
            }
        }
    }

    coverage = e3.model_capability_coverage(
        samples,
        selections,
        ["univariate-only"],
    )

    trend = coverage[coverage["capability_id"] == "trend"].iloc[0]
    common = coverage[
        coverage["capability_id"] == "common_factor"
    ].iloc[0]
    assert bool(trend["supported"])
    assert not bool(common["supported"])
    assert common["unsupported_reason"] == "model_input_contract_unsupported"


def test_bootstrap_pair_states_use_metric_specific_equivalence_scales():
    rows = []
    for model_id, mase, mechanism in (
        ("left", 0.8, 0.61),
        ("right", 1.0, 0.60),
    ):
        for group_index in range(12):
            rows.append(
                {
                    "model_id": model_id,
                    "dataset_id": "dataset",
                    "task_id": "univariate",
                    "capability_id": "trend",
                    "paired_group_id": f"g{group_index:02d}",
                    "round_index": 0,
                    "sample_index": group_index,
                    "mase_group_mean": mase,
                    "blind_mase_group_mean": 1.0,
                    "level_mechanism_group_mean": mechanism,
                    "dose_response_score": mechanism,
                    "formal_score_eligible": True,
                }
            )
    components = pd.DataFrame(rows)

    intervals, pairs = e3.bootstrap_profile_statistics(
        components,
        bank_id="all",
        bootstrap_replicates=100,
        bootstrap_seed=7,
        ci_level=0.95,
        equivalence_margins=(0.02,),
    )

    assert len(intervals) == 6
    primary = pairs.set_index("metric_id")
    assert primary.loc["mase", "effect_scale"] == "relative"
    assert primary.loc["mase", "state"] == "left_better"
    assert primary.loc["mechanism", "effect_scale"] == "absolute"
    assert primary.loc["mechanism", "state"] == "equivalent"
    assert primary.loc["ability", "state"] == "equivalent"


def test_split_half_keeps_whole_paired_groups():
    rows = []
    for model_id in ("left", "right"):
        for group_index in range(8):
            rows.append(
                {
                    "model_id": model_id,
                    "dataset_id": "dataset",
                    "task_id": "univariate",
                    "capability_id": "trend",
                    "paired_group_id": f"g{group_index:02d}",
                    "round_index": 0,
                    "sample_index": group_index,
                }
            )
    split = e3.deterministic_split_components(
        pd.DataFrame(rows),
        split_size=4,
    )

    counts = split.groupby(["bank_id", "model_id"])[
        "paired_group_id"
    ].nunique()
    assert (counts == 4).all()
    first = set(
        split.loc[split["bank_id"] == "first", "paired_group_id"]
    )
    second = set(
        split.loc[split["bank_id"] == "second", "paired_group_id"]
    )
    assert first.isdisjoint(second)
