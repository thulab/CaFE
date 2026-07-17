from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).parents[3] / "scripts"
SCRIPT_PATH = SCRIPT_DIR / "run_paper_v2_e3_model_capability_profiles.py"


def load_module():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "run_paper_v2_e3_model_capability_profiles_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_source_and_six_capability_design_are_frozen() -> None:
    module = load_module()
    source_config = module.base.read_json(module.DEFAULT_SOURCE_DIR / "config.json")

    module.validate_source_config(source_config)

    assert module.DEFAULT_OUTPUT_DIR.relative_to(module.REPO_ROOT).as_posix() == (
        "runtime/paper_exp/v2/E3_model_capability_profiles"
    )
    assert len(module.CAPABILITY_ORDER) == 6
    assert module.base.UNIVARIATE_CAPABILITIES == module.CAPABILITY_ORDER
    assert module.base.STRUCTURED_CAPABILITIES == ()
    assert module.base.sha256_file(module.DEFAULT_SOURCE_DIR / "manifest.json") == (
        module.EXPECTED_E2_MANIFEST_SHA256
    )


def test_v2_experiment_config_records_long_shape_and_16_samples() -> None:
    module = load_module()
    source_config = module.base.read_json(module.DEFAULT_SOURCE_DIR / "config.json")

    config = module.experiment_config(
        source_dir=module.DEFAULT_SOURCE_DIR,
        source_config=source_config,
        bootstrap_replicates=2000,
    )

    assert config["request_shape"] == {
        "context_length": 504,
        "horizon": 48,
        "season_length": 24,
        "target_dim": 1,
    }
    assert len(config["profiles"]) == 9
    assert config["aggregation"]["within_profile_intensity"] == (
        "equal sample weight across 5 rounds x 16 samples"
    )
    assert "hard ranks are descriptive" in config["ranking_policy"]


def test_capability_contrast_uses_paired_bootstrap_gap(monkeypatch) -> None:
    module = load_module()
    monkeypatch.setattr(module, "CAPABILITY_ORDER", ("trend",))
    profiles = pd.DataFrame(
        [
            {
                "model_id": "leader",
                "capability_id": "trend",
                "five_level_mase_mean": 0.5,
                "five_level_skill_mase_mean": 0.5,
            },
            {
                "model_id": "lagger",
                "capability_id": "trend",
                "five_level_mase_mean": 0.7,
                "five_level_skill_mase_mean": 0.3,
            },
        ]
    )
    bootstrap = {
        ("leader", "trend"): {
            "five_level_mase_mean": np.asarray([0.48, 0.50, 0.52]),
            "five_level_skill_mase_mean": np.asarray([0.52, 0.50, 0.48]),
        },
        ("lagger", "trend"): {
            "five_level_mase_mean": np.asarray([0.68, 0.70, 0.72]),
            "five_level_skill_mase_mean": np.asarray([0.32, 0.30, 0.28]),
        },
    }

    contrasts = module.capability_contrast_frame(profiles, bootstrap)
    lagger = contrasts[contrasts["model_id"] == "lagger"].iloc[0]

    assert lagger["paired_mase_gap_vs_best"] == pytest.approx(0.2)
    assert lagger["paired_mase_gap_vs_best_ci_low"] == pytest.approx(0.2)
    assert bool(lagger["worse_than_best_at_95ci"]) is True
