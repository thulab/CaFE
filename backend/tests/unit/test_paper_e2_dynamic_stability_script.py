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
    )


def test_default_design_is_frozen_to_seven_models_and_18400_samples(tmp_path):
    module = load_module()
    artifact = module.read_json(module.GENERATOR_ARTIFACT_PATH)

    config = module.experiment_config(default_args(module, tmp_path), artifact)

    assert config["profile_capability_count"] == 23
    assert config["expected_generated_sample_count"] == 18_400
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


def test_generation_pairs_sample_seed_across_intensities(tmp_path, monkeypatch):
    module = load_module()
    artifact = {
        "config": {"online_conditioning_profile_ids": ["profile"]},
        "profiles": {
            "profile": {
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
        "online_conditioning_profile_ids": ["profile"],
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
                "canonical_target_feature": "trend_strength",
                "canonical_target_strength": float(intensity),
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
    assert all(len({row["sample_seed"] for row in group}) == 1 for group in grouped.values())
    assert len(
        {
            grouped[(round_index, sample_index)][0]["sample_seed"]
            for round_index in (1, 2)
            for sample_index in (0, 1)
        }
    ) == 4


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
