from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).parents[3] / "scripts"
SCRIPT_PATH = SCRIPT_DIR / "run_paper_v2_e2_dynamic_stability.py"


def load_module():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "run_paper_v2_e2_dynamic_stability_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_args(module, tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=tmp_path / "E2",
        base_url="http://127.0.0.1:10810",
        api_prefix="/ai/api/v1",
        models=list(module.base.DEFAULT_MODELS),
        round_seeds=list(module.DEFAULT_ROUND_SEEDS),
        samples_per_round=module.DEFAULT_SAMPLES_PER_ROUND,
        devices="0,1",
        request_max_attempts=3,
        forecast_timeout_seconds=1200,
        model_load_timeout_seconds=1800,
        bootstrap_replicates=module.base.DEFAULT_BOOTSTRAP_REPLICATES,
        stage="all",
        resume=False,
        keep_loaded=False,
    )


def test_default_v2_design_is_long_context_and_21600_samples(tmp_path) -> None:
    module = load_module()
    artifact = module.base.read_json(module.GENERATOR_ARTIFACT_PATH)

    config = module.experiment_config(default_args(module, tmp_path), artifact)

    assert config["profile_capability_count"] == 54
    assert config["expected_generated_sample_count"] == 21_600
    assert config["samples_per_round_per_cell"] == 16
    assert config["fixed_request_shape"] == {
        "context_length": 504,
        "horizon": 48,
        "season_length": 24,
        "target_dim": 1,
    }
    assert len(config["online_conditioning_profile_ids"]) == 9
    assert config["requested_models"][-1] == "tirex2"
    assert "tabpfn-ts3" not in config["requested_models"]


def test_generation_injects_all_frozen_transfer_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    module = load_module()
    profile = {
        "context_length": 4,
        "horizon": 2,
        "target_dim": 1,
        "season_length": 2,
        "frequency": "h",
        "capabilities": {"trend": {}},
    }
    artifacts = {
        "generator": {
            "config": {"online_conditioning_profile_ids": ["profile"]},
            "profiles": {"profile": profile},
        },
        "feature_gate": {"marker": "feature"},
        "near_distance": {"marker": "near"},
        "manifest": {"experiment_version": "v2"},
    }
    config = {
        "online_conditioning_profile_ids": ["profile"],
        "round_seeds": [101, 202],
        "samples_per_round_per_cell": 1,
        "expected_generated_sample_count": 10,
    }
    monkeypatch.setattr(
        module.base,
        "resolve_generator_conditioning",
        lambda **_kwargs: object(),
    )
    observed_kwargs = []

    def generate(
        capability_id,
        length,
        context_length,
        target_dim,
        season_length,
        intensity,
        sample_seed,
        **kwargs,
    ):
        del capability_id, context_length, season_length, sample_seed
        observed_kwargs.append(kwargs)
        target = np.full((length, target_dim), float(intensity))
        latent = {
            "acceptance": {"attempts": 1},
            "generator_conditioning": {
                "canonical_target_feature": "trend_strength",
                "canonical_target_strength": float(intensity),
            },
        }
        return target, latent, None, {"trend_strength": float(intensity)}

    monkeypatch.setattr(module.base, "_generate_accepted_sample_values", generate)
    output = tmp_path / "E2"
    output.mkdir()

    module.generate_samples_if_needed(output, config=config, artifacts=artifacts)

    rows = list(module.base.iter_jsonl(output / "samples.jsonl"))
    assert len(rows) == 10
    assert all(
        kwargs["generator_conditioning_artifact"] is artifacts["generator"]
        and kwargs["feature_gate_artifact"] is artifacts["feature_gate"]
        and kwargs["near_distance_artifact"] is artifacts["near_distance"]
        and kwargs["acceptance_profile_ids"] == ("profile",)
        for kwargs in observed_kwargs
    )


def test_validate_frozen_design_rejects_short_context() -> None:
    module = load_module()
    artifact = module.base.read_json(module.GENERATOR_ARTIFACT_PATH)
    profile_id = artifact["config"]["online_conditioning_profile_ids"][0]
    artifact["profiles"][profile_id]["context_length"] = 168

    try:
        module.validate_frozen_design(artifact)
    except ValueError as error:
        assert "expected (504, 48, 24, 1)" in str(error)
    else:
        raise AssertionError("short paper-v2 context was accepted")
