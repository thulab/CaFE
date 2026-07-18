from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "build_synthetic_v2_feature_gate_artifact.py"


def load_calibration_module():
    repo_root = SCRIPT_PATH.parents[1]
    for path in (repo_root / "backend", repo_root / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("build_synthetic_v2_feature_gate_artifact", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(index: int, *, group_id: str, start: int) -> dict:
    value = index / 100.0
    return {
        "group_id": group_id,
        "window_start": start,
        "features": {
            "trend_strength": value,
            "slope_abs": value * 0.5,
            "curvature_abs": value * 0.25,
            "seasonal_strength": 0.5 + value * 0.1,
            "noise_ratio": 0.2 + value * 0.05,
            "outlier_rate": value * 0.005,
            "spike_rate": value * 0.01,
        },
    }


def test_group_split_keeps_series_disjoint():
    module = load_calibration_module()
    spec = module.BucketSpec("unit", "tsf_univariate", "unit.zip", 2, 1, 1, 2)
    rows = [
        row(group_index * 10 + offset, group_id=f"series:{group_index}", start=offset)
        for group_index in range(10)
        for offset in range(10)
    ]

    reference, calibration, summary = module.split_real_rows(
        rows,
        spec,
        calibration_fraction=0.2,
        seed=123,
    )

    assert summary["policy"] == "group"
    assert {item["group_id"] for item in reference}.isdisjoint(
        {item["group_id"] for item in calibration}
    )


def test_single_series_split_applies_context_horizon_embargo():
    module = load_calibration_module()
    spec = module.BucketSpec("unit", "tsf_univariate", "unit.zip", 8, 4, 1, 4)
    rows = [row(index, group_id="series:one", start=index * 20) for index in range(100)]

    reference, calibration, summary = module.split_real_rows(
        rows,
        spec,
        calibration_fraction=0.2,
        seed=123,
    )

    assert summary["policy"] == "temporal_embargo"
    assert summary["embargo_steps"] == 12
    assert max(item["window_start"] for item in reference) + 12 <= min(
        item["window_start"] for item in calibration
    )


def test_three_way_split_keeps_generator_fit_out_of_both_gate_partitions():
    module = load_calibration_module()
    spec = module.BucketSpec("unit", "tsf_univariate", "unit.zip", 2, 1, 1, 2)
    rows = [
        row(group_index * 10 + offset, group_id=f"series:{group_index}", start=offset)
        for group_index in range(20)
        for offset in range(10)
    ]

    parameters, reference, calibration, summary = module.split_real_rows_three_way(
        rows,
        spec,
        calibration_fraction=0.2,
        gate_reference_fraction=0.4,
        seed=123,
    )

    parameter_groups = {item["group_id"] for item in parameters}
    reference_groups = {item["group_id"] for item in reference}
    calibration_groups = {item["group_id"] for item in calibration}
    assert parameter_groups.isdisjoint(reference_groups)
    assert parameter_groups.isdisjoint(calibration_groups)
    assert reference_groups.isdisjoint(calibration_groups)
    assert summary["generator_parameter_count"] == len(parameters)


def test_capability_threshold_uses_real_calibration_only():
    module = load_calibration_module()
    rows = [row(index, group_id=f"series:{index // 10}", start=index) for index in range(100)]
    config = module.calibrate_capability("trend", rows[:80], rows[80:], coverage=0.95)
    support = config["control_support"]

    assert support["method"] == "shrunk_robust_mahalanobis"
    assert support["reference_count"] == 80
    assert support["calibration_count"] == 20
    assert support["calibration_acceptance_rate"] >= 0.95
    assert set(config["target_reference"]) == {"trend_strength", "slope_abs", "curvature_abs"}


def test_capability_without_independent_controls_records_explicit_contract(
    monkeypatch,
) -> None:
    module = load_calibration_module()
    rows = [
        row(index, group_id=f"series:{index // 10}", start=index)
        for index in range(100)
    ]
    monkeypatch.setitem(
        module.CONTROL_FEATURES_BY_CAPABILITY,
        "regime_switching",
        (),
    )

    config = module.calibrate_capability(
        "regime_switching",
        rows[:80],
        rows[80:],
        coverage=0.95,
    )
    support = config["control_support"]

    assert support["method"] == (
        "not_applicable_no_independent_observable_controls"
    )
    assert support["feature_names"] == []
    assert support["calibration_acceptance_rate"] == 1.0
