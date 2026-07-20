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


def test_discovers_all_supported_cells_from_shards_and_frozen_artifact(
    tmp_path,
):
    e2_dir = tmp_path / "e2"
    shard_dir = e2_dir / "sample_shards"
    shard_dir.mkdir(parents=True)
    cells = [
        ("gefcom2012_load", "covariate", "covariate_response"),
        ("m5", "hierarchical", "hierarchical_coherence"),
    ]
    for index, (dataset_id, task_id, capability_id) in enumerate(cells):
        row = {
            "dataset_id": dataset_id,
            "task_id": task_id,
            "capability_id": capability_id,
            "master_sample_id": f"m{index}",
        }
        (shard_dir / f"shard-{index}.jsonl").write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )
    support_path = tmp_path / "support.json"
    support_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "dataset_id": dataset_id,
                        "task_id": task_id,
                        "capability_id": capability_id,
                        "status": "supported",
                    }
                    for dataset_id, task_id, capability_id in cells
                ]
                + [
                    {
                        "dataset_id": "ignored",
                        "task_id": "univariate",
                        "capability_id": "trend",
                        "status": "unsupported",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (e2_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "suite_files": {
                    "support": {"path": str(support_path)}
                }
            }
        ),
        encoding="utf-8",
    )

    discovered = e3.discover_supported_cells(e2_dir)

    assert discovered == tuple(sorted(cells))
    assert e3.discover_supported_cells(
        e2_dir,
        dataset_ids=["m5"],
    ) == (cells[1],)


def test_v7_320_group_split_is_complete_mutually_exclusive_and_audited():
    rows = []
    for model_id in ("native", "adapted"):
        for group_index in range(320):
            rows.append(
                {
                    "model_id": model_id,
                    "dataset_id": "dataset",
                    "task_id": "common_factor",
                    "capability_id": "common_factor",
                    "paired_group_id": f"g{group_index:03d}",
                    "round_index": group_index // 64 + 1,
                    "sample_index": group_index % 64,
                    "analysis_pool_index": group_index,
                    "analysis_block_id": (
                        "A" if group_index < 160 else "B"
                    ),
                }
            )
    components = pd.DataFrame(rows)

    split = e3.deterministic_split_components(
        components,
        split_size=160,
    )
    audit = e3.split_assignment_audit(
        components,
        split,
        split_size=160,
    )

    counts = split.groupby(["bank_id", "model_id"])[
        "paired_group_id"
    ].nunique()
    assert (counts == 160).all()
    assert audit["all_cells_have_two_complete_blocks"]
    assert audit["all_blocks_mutually_exclusive"]
    assert audit["all_blocks_cover_all_selected_groups"]
    assert audit["all_source_analysis_blocks_aligned"]


def test_v7_coverage_uses_original_view_adaptation_provenance():
    samples = {
        "m1": {
            "dataset_id": "panel",
            "task_id": "common_factor",
            "capability_id": "common_factor",
        },
        "m2": {
            "dataset_id": "panel",
            "task_id": "common_factor",
            "capability_id": "common_factor",
        },
    }
    selections = {
        "model": {
            master_id: {
                **sample,
                "master_sample_id": master_id,
            }
            for master_id, sample in samples.items()
        }
    }
    predictions = {
        "model": {
            master_id: {
                "input_adaptation": {
                    "policy_id": "paper-v7-input-adaptation-v1",
                    "adapted": True,
                    "target_mode": "independent_univariate",
                    "covariate_mode": "omitted_unsupported",
                }
            }
            for master_id in samples
        }
    }

    coverage = e3.model_capability_coverage(
        samples,
        selections,
        ["model"],
        predictions,
    ).iloc[0]

    assert bool(coverage["supported"])
    assert coverage["input_execution_mode"] == "adapted"
    assert coverage["prediction_sample_count"] == 2
    assert coverage["adapted_view_count"] == 2
    assert coverage["covariates_omitted_view_count"] == 2


def test_v7_prediction_loader_rejects_missing_adaptation_provenance(
    tmp_path,
):
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    (prediction_dir / "model.jsonl").write_text(
        json.dumps(
            {
                "view_id": "view",
                "forecast": [[0.0]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lacks v7"):
        e3.load_selected_predictions(
            tmp_path,
            model_id="model",
            oracle_selection={
                "master": {"oracle_view_id": "view"}
            },
            require_input_adaptation=True,
        )


def test_covariate_omitted_counterfactual_must_reuse_intact_with_zero_http(
    tmp_path,
):
    forecast = [[1.0], [2.0]]
    (tmp_path / "model.jsonl").write_text(
        json.dumps(
            {
                "master_sample_id": "sample",
                "context_length": 168,
                "forecast": forecast,
                "ablation": "future_covariates_zero",
                "counterfactual_mode": (
                    "reuse_intact_forecast_covariates_omitted"
                ),
                "counterfactual_http_request_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples = {
        "sample": {"capability_id": "covariate_response"}
    }
    oracle = {
        "model": {"sample": {"oracle_context": 168}}
    }
    intact = {
        "model": {
            "sample": {
                "forecast": forecast,
                "input_adaptation": {
                    "policy_id": "paper-v7-input-adaptation-v1",
                    "adapted": True,
                    "target_mode": "native_univariate",
                    "covariate_mode": "omitted_unsupported",
                },
            }
        }
    }

    result = e3.load_covariate_ablation_predictions(
        tmp_path,
        model_ids=["model"],
        samples=samples,
        oracle_selections=oracle,
        intact_predictions=intact,
    )

    assert result["model"]["sample"]["forecast"] == forecast


def test_covariate_omitted_reuse_scores_zero_effect_as_formal():
    length = e3.MAX_CONTEXT_LENGTH + e3.HORIZON
    covariates = np.linspace(-1.0, 1.0, length)[:, None]
    target = (2.0 * covariates + 0.1)[:, 0, None]
    sample = {
        "dataset_id": "dataset",
        "task_id": "covariate",
        "capability_id": "covariate_response",
        "intensity": 3,
        "paired_group_id": "group",
        "master_sample_id": "sample",
        "round_index": 1,
        "sample_index": 0,
        "analysis_pool_index": 0,
        "analysis_block_id": "A",
        "target": target.tolist(),
        "covariates": covariates.tolist(),
        "horizon": e3.HORIZON,
        "season_length": 24,
        "generation_metadata": {},
    }
    target_view, _ = e3.context_view(sample, context_length=168)
    forecast = np.zeros((e3.HORIZON, 1)).tolist()
    predictions = {
        "model": {
            "sample": {
                "forecast": forecast,
                "target_future": target_view[168:].tolist(),
                "input_adaptation": {
                    "policy_id": "paper-v7-input-adaptation-v1",
                    "adapted": True,
                    "target_mode": "native_univariate",
                    "covariate_mode": "omitted_unsupported",
                },
            }
        }
    }
    counterfactual = {
        "model": {
            "sample": {
                "forecast": forecast,
                "counterfactual_mode": (
                    "reuse_intact_forecast_covariates_omitted"
                ),
                "counterfactual_http_request_count": 0,
            }
        }
    }

    scores = e3.evaluate_selected_predictions(
        {"sample": sample},
        {
            "model": {
                "sample": {
                    "oracle_context": 168,
                    "oracle_mase": 1.0,
                }
            }
        },
        predictions,
        counterfactual,
    )

    row = scores.iloc[0]
    assert bool(row["formal_score_eligible"])
    assert row["input_execution_mode"] == "adapted"
    assert row["counterfactual_effect_mae"] == 0.0
    assert row["counterfactual_http_request_count"] == 0
    diagnostics = json.loads(row["diagnostics_json"])
    assert diagnostics["evaluation_mode"] == (
        "paired_future_covariate_ablation"
    )
