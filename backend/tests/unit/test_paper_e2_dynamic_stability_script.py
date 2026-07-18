from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "run_paper_e2_dynamic_stability.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_paper_e2_dynamic_stability", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def default_args(module, tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=tmp_path / "E2",
        base_url="http://127.0.0.1:10810",
        api_prefix="/ai/api/v1",
        models=list(module.DEFAULT_MODELS),
        round_seeds=list(module.DEFAULT_ROUND_SEEDS),
        samples_per_round=module.DEFAULT_SAMPLES_PER_ROUND,
        devices="0,1",
        request_max_attempts=3,
        forecast_timeout_seconds=1200,
        model_load_timeout_seconds=1800,
        bootstrap_replicates=module.DEFAULT_BOOTSTRAP_REPLICATES,
        stage="all",
        resume=False,
        keep_loaded=False,
        skip_real_alignment=False,
    )


def v4_artifact() -> dict:
    capability = {
        "parameters": {"structure_scale": 1.0},
        "intensity_lambdas": [0.1, 0.3, 0.5, 0.7, 0.9],
        "target_feature": "trend_strength",
        "target_percentile_levels": [0.0, 0.25, 0.5, 0.75, 1.0],
        "target_values": [0.05, 0.1, 0.2, 0.35, 0.55],
        "calibrated_realized_strengths": [0.05, 0.1, 0.2, 0.35, 0.55],
        "calibration": {"status": "supported", "max_normalized_error": 0.01},
        "calibration_method": "dataset-local-test",
    }
    return {
        "schema_version": "synthetic_v2_generator_conditioning_artifact.v4",
        "created_at": "2026-07-18T00:00:00+00:00",
        "intensity_policy": {
            "policy_id": "dataset-local-real-bounded-generator-feasible-v1",
            "percentile_levels": [0.0, 0.25, 0.5, 0.75, 1.0],
            "relative_dose_levels": [0.0, 0.25, 0.5, 0.75, 1.0],
            "real_tolerance": {
                "lower_quantile": 0.05,
                "upper_quantile": 0.95,
                "upper_multiplier": 1.2,
            },
            "definition": "dataset-local relative intensity",
        },
        "profiles": {
            "dataset_a__univariate__L504_H48": {
                "profile_id": "dataset_a__univariate__L504_H48",
                "dataset_id": "dataset_a",
                "task_id": "univariate",
                "context_length": 504,
                "horizon": 48,
                "target_dim": 1,
                "season_length": 24,
                "frequency": "h",
                "nuisance_parameters": {"noise_scale_multiplier": 1.0},
                "capabilities": {
                    "trend": capability,
                    "multi_seasonal": {
                        **capability,
                        "target_feature": "multi_period_score",
                    },
                },
            }
        },
    }


def v4_support_matrix() -> dict:
    return {
        "cells": [
            {
                "dataset_id": "dataset_a",
                "task_id": "univariate",
                "capability_id": capability_id,
                "generator_profile_id": "dataset_a__univariate__L504_H48",
                "status": "supported",
                "reason_codes": [],
            }
            for capability_id in ("trend", "multi_seasonal")
        ]
        + [
            {
                "dataset_id": "dataset_a",
                "task_id": "hierarchy",
                "capability_id": "hierarchical_coherence",
                "generator_profile_id": "dataset_a__hierarchy__L504_H48",
                "status": "unsupported",
                "reason_codes": ["variable_structure_not_supported"],
            }
        ]
    }


def test_default_design_uses_v4_dataset_local_supported_cells(tmp_path):
    module = load_module()
    artifact = v4_artifact()

    config = module.experiment_config(
        default_args(module, tmp_path),
        artifact,
        support_matrix=v4_support_matrix(),
    )

    assert config["profile_capability_count"] == 2
    assert config["skipped_profile_capability_count"] == 1
    assert config["expected_generated_sample_count"] == 1_600
    assert config["intensity_policy"]["comparability"] == "within_dataset_only"
    assert config["intensity_policy"]["relative_dose_levels"] == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert "percentile_levels" not in config["intensity_policy"]
    assert "canonical_scale_id" not in config
    assert len(config["round_seeds"]) == 5
    assert config["samples_per_round_per_cell"] == 32
    assert config["requested_models"][-2:] == ["timesfm2.5", "tirex2"]
    assert "tabpfn-ts3" not in config["requested_models"]
    assert config["model_execution"]["Chronos-2"] == {
        "replicas_per_device": 4,
        "http_concurrency": 32,
    }
    assert config["model_execution"]["timesfm2.5"] == {
        "replicas_per_device": 8,
        "http_concurrency": 32,
    }
    assert config["tasks_per_http_request"] == 1


def test_experiment_config_rejects_legacy_canonical_artifact(tmp_path):
    module = load_module()
    with pytest.raises(ValueError, match="v4 dataset-local"):
        module.experiment_config(
            default_args(module, tmp_path),
            {
                "schema_version": "synthetic_v2_generator_conditioning_artifact.v3",
                "canonical_intensity": {"scale_id": "legacy"},
                "profiles": {},
            },
        )


def test_generation_pairs_sample_seed_across_intensities(tmp_path, monkeypatch):
    module = load_module()
    artifact = {
        "profiles": {
            "profile": {
                "dataset_id": "dataset_a",
                "context_length": 4,
                "horizon": 2,
                "target_dim": 1,
                "season_length": 2,
                "frequency": "h",
                "capabilities": {"trend": {}},
            }
        },
    }
    config = {
        "eligible_profile_capability_cells": [
            {
                "dataset_id": "dataset_a",
                "profile_id": "profile",
                "capability_id": "trend",
            }
        ],
        "round_seeds": [101, 202],
        "samples_per_round_per_cell": 2,
        "expected_generated_sample_count": 20,
    }

    monkeypatch.setattr(module, "resolve_generator_conditioning", lambda **_kwargs: {})

    def generate(
        capability_id,
        length,
        context_length,
        target_dim,
        season_length,
        intensity,
        sample_seed,
        **_kwargs,
    ):
        del capability_id, context_length, season_length
        target = np.full((length, target_dim), float(intensity))
        latent = {
            "acceptance": {"attempts": 1},
            "generator_conditioning": {
                "target_feature": "trend_strength",
                "target_strength": float(intensity),
                "target_relative_level": (intensity - 1) / 4,
                "target_percentile_level": (intensity - 1) / 4,
            },
        }
        return target, latent, None, {"trend_strength": float(intensity)}

    monkeypatch.setattr(module, "_generate_accepted_sample_values", generate)
    output = tmp_path / "E2"
    output.mkdir()

    module.generate_samples_if_needed(output, config=config, artifact=artifact)
    rows = list(module.iter_jsonl(output / "samples.jsonl"))
    grouped = module.group_rows(rows, "round_index", "sample_index")

    assert len(rows) == 20
    assert {row["dataset_id"] for row in rows} == {"dataset_a"}
    assert {row["intensity_comparability"] for row in rows} == {
        "within_dataset_only"
    }
    assert "canonical_target_strength" not in rows[0]
    assert rows[0]["target_relative_level"] == 0.0
    assert "target_percentile_level" not in rows[0]
    assert all(len({row["sample_seed"] for row in group}) == 1 for group in grouped.values())
    assert len(
        {
            grouped[(round_index, sample_index)][0]["sample_seed"]
            for round_index in (1, 2)
            for sample_index in (0, 1)
        }
    ) == 4


def test_real_prediction_row_omits_synthetic_coordinates():
    module = load_module()
    sample = {
        "sample_id": "real-1",
        "dataset_id": "dataset_a",
        "task_id": "univariate",
        "context_length": 4,
        "horizon": 2,
        "season_length": 2,
        "target": [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
    }
    row = module.prediction_row(
        "model_a",
        "timer_service",
        sample,
        [[5.0], [6.0]],
    )
    assert row["schema_version"] == "paper_e2_real_prediction.v1"
    assert row["dataset_id"] == "dataset_a"
    assert "capability_id" not in row
    assert "intensity" not in row


def test_real_builder_sample_is_normalized_for_existing_forecast_pipeline(
    tmp_path,
):
    module = load_module()
    path = tmp_path / "real_samples.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "real-1",
                "dataset_id": "dataset_a",
                "lookback": 4,
                "horizon": 2,
                "season_length": 2,
                "frequency": "h",
                "target_dim": 1,
                "target_history": [1.0, 2.0, 3.0, 4.0],
                "target_future": [5.0, 6.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sample = next(module.iter_forecast_samples(path))
    assert sample["context_length"] == 4
    assert sample["covariate_dim"] == 0
    assert np.asarray(sample["target"]).shape == (6, 1)
    assert module.forecast_request_body("model_a", sample)["output_length"] == [2]


def test_real_baselines_use_separate_prediction_directory(tmp_path):
    module = load_module()
    path = tmp_path / "real_samples.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "real-1",
                "dataset_id": "dataset_a",
                "task_id": "univariate",
                "lookback": 4,
                "horizon": 2,
                "season_length": 2,
                "frequency": "h",
                "target_dim": 1,
                "target_history": [1.0, 2.0, 3.0, 4.0],
                "target_future": [5.0, 6.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "real_predictions").mkdir()
    module.run_baselines(
        tmp_path,
        sample_path=path,
        prediction_kind="real",
    )
    for model_id in module.BASELINE_MODELS:
        prediction_path = module.prediction_path_for(
            tmp_path,
            model_id,
            prediction_kind="real",
        )
        rows = list(module.iter_jsonl(prediction_path))
        assert len(rows) == 1
        assert rows[0]["schema_version"] == "paper_e2_real_prediction.v1"


def test_synthetic_real_alignment_is_dataset_local_and_omits_incompatible_models(
    tmp_path,
    monkeypatch,
):
    module = load_module()
    real_sample_path = tmp_path / "real_samples.jsonl"
    support_path = tmp_path / "dataset_support.json"
    real_samples = [
        {
            "sample_id": f"real-{index}",
            "dataset_id": "dataset_a",
            "context_length": 4,
            "horizon": 2,
            "target_dim": 1,
            "covariate_dim": 0,
            "frequency": "h",
        }
        for index in range(2)
    ]
    with real_sample_path.open("w", encoding="utf-8") as handle:
        for row in real_samples:
            handle.write(json.dumps(row) + "\n")
    support_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"dataset_id": "dataset_a", "status": "supported"},
                    {"dataset_id": "dataset_b", "status": "unsupported"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "REAL_SAMPLES_PATH", real_sample_path)
    monkeypatch.setattr(module, "REAL_DATASET_SUPPORT_PATH", support_path)
    (tmp_path / "real_predictions").mkdir()

    model_scores = {
        "model_a": 1.0,
        "model_b": 2.0,
        "naive": 3.0,
        "seasonal_naive": 4.0,
    }
    for model_id, score in model_scores.items():
        path = module.prediction_path_for(
            tmp_path,
            model_id,
            prediction_kind="real",
        )
        with path.open("w", encoding="utf-8") as handle:
            for sample in real_samples:
                handle.write(
                    json.dumps(
                        {
                            "model_id": model_id,
                            "model_group": (
                                "baseline"
                                if model_id in module.BASELINE_MODELS
                                else "timer_service"
                            ),
                            "sample_id": sample["sample_id"],
                            "dataset_id": "dataset_a",
                            "metrics": {"mase": score},
                        }
                    )
                    + "\n"
                )

    synthetic_predictions = []
    for capability_id in ("trend", "multi_seasonal"):
        for intensity in module.INTENSITIES:
            sample_id = f"{capability_id}-{intensity}"
            for model_id, score in {
                **model_scores,
                "model_incompatible": 0.5,
            }.items():
                synthetic_predictions.append(
                    {
                        "model_id": model_id,
                        "model_group": (
                            "baseline"
                            if model_id in module.BASELINE_MODELS
                            else "timer_service"
                        ),
                        "sample_id": sample_id,
                        "dataset_id": "dataset_a",
                        "capability_id": capability_id,
                        "intensity": intensity,
                        "metrics": {"mase": score},
                    }
                )
    compatible_limits = {
        "forecast_limits": {
            "min_input_length": 1,
            "max_input_length": 10,
            "max_output_length": 10,
            "max_target_count": 1,
            "max_covariate_count": 0,
            "max_future_covs_length": None,
        }
    }
    incompatible_limits = {
        "forecast_limits": {
            **compatible_limits["forecast_limits"],
            "max_input_length": 2,
        }
    }
    alignment = module.analyze_synthetic_real_alignment(
        tmp_path,
        config={
            "skip_real_alignment": False,
            "requested_models": [
                "model_a",
                "model_b",
                "model_incompatible",
            ],
        },
        catalog={
            "model_a": compatible_limits,
            "model_b": compatible_limits,
            "model_incompatible": incompatible_limits,
        },
        synthetic_predictions=synthetic_predictions,
    )
    row = alignment["rows"][0]
    assert alignment["real_alignment_lookback"] == 4
    assert row["dataset_id"] == "dataset_a"
    assert row["effective_model_count"] == 4
    assert "model_incompatible" not in row["models"]
    assert row["effective_capability_count"] == 2
    assert row["spearman_rho"] == pytest.approx(1.0)
    assert row["kendall_tau_b"] == pytest.approx(1.0)
    assert row["top_k_overlap_rate"] == pytest.approx(1.0)
    assert row["pairwise_ordering_agreement"] == pytest.approx(1.0)


def test_skip_real_alignment_never_reads_real_suite(tmp_path):
    module = load_module()
    result = module.analyze_synthetic_real_alignment(
        tmp_path,
        config={"skip_real_alignment": True},
        catalog={},
        synthetic_predictions=[],
    )
    assert result["status"] == "skipped"
    assert result["rows"] == []


def test_model_compatibility_uses_target_and_future_covariate_limits():
    module = load_module()
    base = {
        "context_length": 168,
        "horizon": 24,
        "target_dim": 1,
        "covariate_dim": 0,
    }
    univariate_only = {
        "forecast_limits": {
            "min_input_length": 16,
            "max_input_length": 2048,
            "max_output_length": 96,
            "max_target_count": 1,
            "max_covariate_count": 0,
            "max_future_covs_length": None,
        }
    }
    structured = {
        "forecast_limits": {
            "min_input_length": 16,
            "max_input_length": 4096,
            "max_output_length": 720,
            "max_target_count": None,
            "max_covariate_count": 50,
            "max_future_covs_length": 720,
        }
    }

    assert module.model_supports_sample(univariate_only, base)
    assert not module.model_supports_sample(
        univariate_only, {**base, "target_dim": 3}
    )
    assert not module.model_supports_sample(
        univariate_only, {**base, "covariate_dim": 2}
    )
    assert module.model_supports_sample(structured, {**base, "target_dim": 3})
    assert module.model_supports_sample(structured, {**base, "covariate_dim": 2})


def test_model_load_uses_and_verifies_frozen_replica_topology(monkeypatch):
    module = load_module()
    client = object.__new__(module.TimerServiceClient)
    expected = {
        "model_id": "model",
        "status": "loaded",
        "endpoints": [
            {"device": "cuda:0", "worker_pid": 10},
            {"device": "cuda:0", "worker_pid": 11},
            {"device": "cuda:1", "worker_pid": 12},
            {"device": "cuda:1", "worker_pid": 13},
        ],
    }
    states = iter([None, expected])
    posts = []
    monkeypatch.setattr(client, "_loaded_state", lambda _model_id: next(states))
    monkeypatch.setattr(
        client,
        "_post",
        lambda path, body, **kwargs: posts.append((path, body, kwargs)) or {},
    )
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    _seconds, state = client.ensure_loaded(
        "model",
        devices="0,1",
        replicas_per_device=2,
        timeout_seconds=60,
    )

    assert state == expected
    assert posts[0][1] == {
        "model_id": "model",
        "devices": "0,1",
        "replicas_per_device": 2,
    }


def test_model_load_rejects_wrong_replica_topology(monkeypatch):
    module = load_module()
    client = object.__new__(module.TimerServiceClient)
    states = iter(
        [
            None,
            {
                "model_id": "model",
                "status": "loaded",
                "endpoints": [
                    {"device": "cuda:0", "worker_pid": 10},
                    {"device": "cuda:1", "worker_pid": 11},
                ],
            },
        ]
    )
    monkeypatch.setattr(client, "_loaded_state", lambda _model_id: next(states))
    monkeypatch.setattr(client, "_post", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="topology"):
        client.ensure_loaded(
            "model",
            devices="0,1",
            replicas_per_device=2,
            timeout_seconds=60,
        )


def test_request_group_order_finishes_non_covariate_shapes_first():
    module = load_module()
    groups = [
        (365, 28, 1, 2, "d"),
        (168, 24, 3, 0, "h"),
        (168, 24, 1, 2, "h"),
        (365, 28, 3, 0, "d"),
        (168, 24, 1, 0, "h"),
    ]

    assert sorted(groups, key=module.request_group_sort_key) == [
        (168, 24, 1, 0, "h"),
        (168, 24, 3, 0, "h"),
        (365, 28, 3, 0, "d"),
        (168, 24, 1, 2, "h"),
        (365, 28, 1, 2, "d"),
    ]


def test_rank_stability_keeps_foundation_scope_separate_from_baselines():
    module = load_module()
    rows = []
    scores = {
        "model_a": [1.0, 1.1],
        "model_b": [2.0, 2.1],
        "naive": [3.0, 0.5],
        "seasonal_naive": [4.0, 4.1],
    }
    for model_id, values in scores.items():
        for round_index, score in enumerate(values, start=1):
            rows.append(
                {
                    "model_id": model_id,
                    "model_group": (
                        "baseline" if model_id in module.BASELINE_MODELS else "timer_service"
                    ),
                    "profile_id": "profile",
                    "capability_id": "trend",
                    "intensity": 1,
                    "round_index": round_index,
                    "mase_mean": score,
                }
            )

    result = module.rank_stability_rows(rows)
    by_scope = {row["ranking_scope"]: row for row in result}

    assert by_scope["foundation_models"]["models"] == "model_a;model_b"
    assert by_scope["foundation_models"]["kendall_tau_mean"] == pytest.approx(1.0)
    assert by_scope["all_predictors"]["model_count"] == 4
    assert by_scope["all_predictors"]["kendall_tau_mean"] < 1.0


def test_icc_and_kendall_statistics_have_known_limits():
    module = load_module()
    identical_rounds = np.asarray([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])

    assert module.icc_a1(identical_rounds) == pytest.approx(1.0)
    assert module.kendall_tau_b(
        np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0])
    ) == pytest.approx(1.0)
    assert module.kendall_tau_b(
        np.asarray([1.0, 2.0, 3.0]), np.asarray([3.0, 2.0, 1.0])
    ) == pytest.approx(-1.0)


def test_hierarchical_bootstrap_is_deterministic_and_resamples_rounds():
    module = load_module()
    rounds = [np.asarray([1.0, 1.0]), np.asarray([3.0, 3.0])]

    first = module.hierarchical_bootstrap_means(rounds, replicates=500, seed=71)
    second = module.hierarchical_bootstrap_means(rounds, replicates=500, seed=71)

    assert np.array_equal(first, second)
    assert np.mean(first) == pytest.approx(2.0, abs=0.1)
    assert set(np.unique(first)).issubset({1.0, 2.0, 3.0})


def test_cross_round_distance_is_bidirectional_and_detects_duplicates(tmp_path):
    module = load_module()
    path = tmp_path / "samples.jsonl"
    rows = [
        (1, 0, [0.0, 1.0]),
        (1, 1, [5.0, 6.0]),
        (2, 0, [0.0, 1.0]),
        (2, 1, [8.0, 9.0]),
    ]
    with path.open("w", encoding="utf-8") as handle:
        for round_index, sample_index, target in rows:
            handle.write(
                json.dumps(
                    {
                        "profile_id": "profile",
                        "capability_id": "trend",
                        "intensity": 1,
                        "round_index": round_index,
                        "sample_index": sample_index,
                        "target": [[value] for value in target],
                    }
                )
                + "\n"
            )

    result = module.cross_round_distance_rows(path)[0]

    assert result["query_count"] == 4
    assert result["exact_duplicate_rate"] == pytest.approx(0.5)
    assert result["rounded_1e6_duplicate_rate"] == pytest.approx(0.5)
    assert result["near_duplicate_mae_le_1e6_rate"] == pytest.approx(0.5)


def test_resume_requires_identical_config_and_manifest_seals_output(tmp_path):
    module = load_module()
    output = tmp_path / "E2"
    config = {"schema_version": module.SCHEMA_VERSION, "round_seeds": [1, 2]}
    module.prepare_or_resume_output(output, config=config, resume=False)
    assert json.loads(
        (output / "skipped_profile_capability_cells.json").read_text(
            encoding="utf-8"
        )
    )["cells"] == []

    module.prepare_or_resume_output(output, config=config, resume=True)
    with pytest.raises(ValueError, match="does not match"):
        module.prepare_or_resume_output(
            output,
            config={**config, "round_seeds": [3, 4]},
            resume=True,
        )

    (output / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="sealed"):
        module.prepare_or_resume_output(output, config=config, resume=True)
