from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[3]
    / "scripts"
    / "run_paper_v8_deterministic_pilot.py"
)


def load_pilot_module():
    spec = importlib.util.spec_from_file_location(
        "run_paper_v8_deterministic_pilot_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_joint_parameter_sampler_preserves_empirical_rank_and_uses_unique_anchors():
    pilot = load_pilot_module()
    features = (
        "acf1",
        "seasonal_acf",
        "dominant_period",
        "spectral_concentration",
        "trend_strength",
        "slope_abs",
        "curvature_abs",
    )
    rows = [
        {
            feature: float(index + feature_index / 100.0)
            for feature_index, feature in enumerate(features)
        }
        for index in range(20)
    ]
    summary = {
        feature: {"p25": 0.25, "p50": 0.50, "p75": 0.75}
        for feature in features
    }

    anchors = set()
    parameter_vectors = set()
    for sample_index in range(16):
        parameters, _, metadata = pilot.sample_parameters(
            "trend",
            summary,
            feature_rows=rows,
            season_length=24,
            sample_index=sample_index,
        )
        anchors.add(metadata["source_window_index"])
        parameter_vectors.add(
            tuple(sorted((name, round(value, 12)) for name, value in parameters.items()))
        )
        quantiles = metadata["sampled_quantiles"]
        assert max(quantiles.values()) - min(quantiles.values()) == 0.0
        assert metadata["fallback_features"] == []

    assert len(anchors) == 16
    assert len(parameter_vectors) == 16


def test_nuisance_audit_counts_mechanism_path_fingerprints():
    pilot = load_pilot_module()
    rows = [
        {
            "generator_family_role": "primary",
            "intensity": 5,
            "generation_metadata": {
                "slope_jitter_by_target": [0.8 + 0.1 * index],
                "curvature_sign_by_target": [1.0 if index % 2 else -1.0],
            },
        }
        for index in range(4)
    ]

    audit = pilot.nuisance_combination_audit("trend", rows)

    assert audit["unique_nuisance_fingerprint_count"] == 4
    assert audit["expected_independent_path_count"] == 4
    assert audit["unique_nuisance_fingerprint_coverage"] == 1.0
    assert audit["missing_fingerprint_fields"] == []
