from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "build_paper_v4_nine_capability_suite.py"
)


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location(
        "build_paper_v4_nine_capability_suite",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_all_tasks_use_four_lookbacks_and_h48(module) -> None:
    assert module.CONTEXT_LENGTHS == (96, 168, 336, 504)
    assert module.HORIZON == 48
    assert len(module.ALL_CAPABILITY_IDS) == 9
    assert module.TASK_DESIGNS["hierarchy"].season_length == 7
    assert (
        module.PRIMARY_TARGET_FEATURE["regime_switching"]
        == "regime_clock_history_incremental_r2"
    )


def test_real_paired_views_share_exact_future(module) -> None:
    length = module.MAX_CONTEXT_LENGTH + module.MASTER_LOADER_HORIZON
    target = np.column_stack(
        [
            np.arange(length, dtype=float),
            2.0 * np.arange(length, dtype=float),
        ]
    )
    raw_future_shapes = []
    standardized_futures = []
    for context_length in module.CONTEXT_LENGTHS:
        view, _ = module.paired_view(
            target,
            None,
            context_length=context_length,
            hierarchy=None,
        )
        assert view.shape == (context_length + module.HORIZON, 2)
        raw_future_shapes.append(target[module.MAX_CONTEXT_LENGTH : module.MAX_CONTEXT_LENGTH + module.HORIZON].shape)
        standardized_futures.append(view[-module.HORIZON :])
    assert raw_future_shapes == [(48, 2)] * 4
    # Different lookback normalizations may alter values, but not future indices.
    assert not np.array_equal(standardized_futures[0], standardized_futures[-1])


def test_hierarchy_view_preserves_additivity(module) -> None:
    length = module.MAX_CONTEXT_LENGTH + module.MASTER_LOADER_HORIZON
    child_a = np.sin(np.arange(length) / 7.0) + 3
    child_b = np.cos(np.arange(length) / 5.0) + 4
    target = np.column_stack([child_a + child_b, child_a, child_b])
    for context_length in module.CONTEXT_LENGTHS:
        view, _ = module.paired_view(
            target,
            None,
            context_length=context_length,
            hierarchy="additive_first",
        )
        assert np.max(np.abs(view[:, 0] - view[:, 1] - view[:, 2])) < 1e-10


def test_mapping_assigns_all_nine_capabilities(module) -> None:
    assert {
        module.task_id_for_capability(capability_id)
        for capability_id in module.ALL_CAPABILITY_IDS
    } == {"univariate", "common_factor", "hierarchy", "covariate"}


def test_dataset_local_profile_ids_never_use_global_pool(module) -> None:
    source = module.UNIVARIATE_CALIBRATION_SOURCES[0]
    assert module.generator_profile_id(source.dataset_id, source.task_id) == (
        "m4_hourly__univariate__L504_H48"
    )
    assert module.gate_profile_id(source.dataset_id, source.task_id, 168) == (
        "m4_hourly__univariate__L168_H48"
    )
    spec = module.dataset_bucket_spec(source, 168)
    assert spec.profile_id == "m4_hourly__univariate__L168_H48"
    assert "global" not in spec.profile_id
    assert "pooled" not in spec.profile_id


def test_relative_intensity_policy_is_dataset_local(module) -> None:
    policy = module.intensity_policy()
    assert policy["policy_id"] == "dataset-local-relative-quantiles-v1"
    assert policy["percentile_levels"] == [0.1, 0.3, 0.5, 0.7, 0.9]
    assert "not comparable across datasets" in policy["definition"]


def test_target_spacing_audit_rejects_flat_or_unresolved_levels(module) -> None:
    flat = module.target_spacing_audit([0.2] * 5)
    unresolved = module.target_spacing_audit([0.1, 0.1, 0.2, 0.3, 0.4])
    resolved = module.target_spacing_audit([0.1, 0.2, 0.3, 0.4, 0.5])
    assert flat["reason_code"] == "insufficient_local_target_range"
    assert unresolved["reason_code"] == "insufficient_local_intensity_spacing"
    assert resolved["supported"] is True


def test_structural_support_is_explicit(module) -> None:
    univariate = module.UNIVARIATE_CALIBRATION_SOURCES[0]
    hierarchy = next(
        source
        for source in module.STRUCTURED_SOURCES
        if source.task_id == "hierarchy"
    )
    covariate = next(
        source
        for source in module.STRUCTURED_SOURCES
        if source.task_id == "covariate"
    )
    assert module.structural_support_audit(
        univariate,
        "common_factor",
    )["supported"] is False
    assert module.structural_support_audit(
        hierarchy,
        "hierarchical_coherence",
    )["supported"] is True
    assert module.structural_support_audit(
        covariate,
        "covariate_response",
    )["supported"] is True


def test_dataset_three_way_split_does_not_accept_multiple_datasets(
    module,
    monkeypatch,
) -> None:
    source = module.UNIVARIATE_CALIBRATION_SOURCES[0]
    spec = module.dataset_bucket_spec(source, 96)
    rows = [{"dataset_id": source.dataset_id, "row": index} for index in range(60)]
    captured = {}

    def fake_split(input_rows, input_spec, **kwargs):
        captured["rows"] = input_rows
        captured["profile_id"] = input_spec.profile_id
        return input_rows[:20], input_rows[20:40], input_rows[40:], {"raw": True}

    monkeypatch.setattr(module, "split_real_rows_three_way", fake_split)
    parameter, reference, calibration, summary = module.dataset_three_way_split(
        rows,
        spec,
        seed=17,
    )
    assert captured["rows"] is rows
    assert {row["dataset_id"] for row in parameter + reference + calibration} == {
        source.dataset_id
    }
    assert summary["policy"] == "dataset_local_three_way_no_pooling"
    assert summary["dataset_id"] == source.dataset_id


def test_support_matrix_row_records_unsupported_reason(module) -> None:
    source = module.UNIVARIATE_CALIBRATION_SOURCES[0]
    row = module.support_matrix_row(
        source,
        "trend",
        status="unsupported",
        reason_codes=("insufficient_local_intensity_spacing",),
        view_support={},
        bucket_failures={},
        target_spacing={"supported": False},
    )
    assert row["dataset_id"] == source.dataset_id
    assert row["supported"] is False
    assert row["reason_codes"] == ["insufficient_local_intensity_spacing"]
    assert len(row["gate_profile_ids"]) == 4


def test_standardized_control_vectors_freeze_dataset_local_reference(module) -> None:
    support = {
        "feature_names": ["a", "b"],
        "feature_center": [1.0, 4.0],
        "feature_scale": [2.0, 4.0],
    }
    rows = [
        {"features": {"a": 3.0, "b": 0.0}},
        {"features": {"a": -1.0, "b": 8.0}},
    ]

    vectors = module.standardized_control_vectors(rows, support)

    assert vectors == [[1.0, -1.0], [-1.0, 1.0]]
    assert module.standardized_control_vectors(
        rows,
        {**support, "feature_names": []},
    ) == [[], []]


def test_build_suite_keeps_calibration_failure_as_unsupported(
    module,
    monkeypatch,
    tmp_path,
) -> None:
    source = module.UNIVARIATE_CALIBRATION_SOURCES[0]
    rows = [
        {
            "dataset_id": source.dataset_id,
            "target": np.arange(144, dtype=float),
            "features": {},
        }
        for _ in range(60)
    ]
    views = {context_length: list(rows) for context_length in module.CONTEXT_LENGTHS}
    monkeypatch.setattr(module, "UNIVARIATE_CALIBRATION_SOURCES", (source,))
    monkeypatch.setattr(module, "STRUCTURED_SOURCES", ())
    monkeypatch.setattr(
        module,
        "load_source_views",
        lambda *args, **kwargs: (views, {"dataset": {"dataset_id": source.dataset_id}}),
    )
    monkeypatch.setattr(
        module,
        "dataset_three_way_split",
        lambda input_rows, spec, **kwargs: (
            input_rows[:20],
            input_rows[20:40],
            input_rows[40:],
            {"policy": "dataset_local_three_way_no_pooling"},
        ),
    )
    monkeypatch.setattr(
        module,
        "calibrate_feature_gate_with_rounding_guard",
        lambda capability_id, reference, calibration: {"control_support": {}},
    )
    monkeypatch.setattr(
        module,
        "thresholds_from_split",
        lambda reference, calibration: ({"raw_mae_p01": 0.1}, {}),
    )
    monkeypatch.setattr(
        module,
        "online_artifact_bucket",
        lambda spec, reference, **kwargs: {"profile_id": spec.profile_id},
    )
    feature_names = set(module.PRIMARY_TARGET_FEATURE.values())
    measured = [
        {
            "features": {
                feature_name: float(index + offset / 10)
                for offset, feature_name in enumerate(sorted(feature_names))
            }
        }
        for index in range(20)
    ]
    monkeypatch.setattr(
        module,
        "measurement_rows",
        lambda *args, **kwargs: measured,
    )
    monkeypatch.setattr(
        module,
        "annotate_regime_clock_rows",
        lambda input_rows, _spec: (
            input_rows,
            [{"qualified": True} for _ in input_rows],
        ),
    )
    monkeypatch.setattr(
        module,
        "qualify_regime_reference_rows",
        lambda input_rows, _spec, **kwargs: (
            input_rows,
            {"qualified_window_count": len(input_rows)},
        ),
    )
    monkeypatch.setattr(
        module,
        "summarize_real_features",
        lambda input_rows: {
            feature_name: {
                "p25": 0.2,
                "p50": 0.5,
                "p75": 0.8,
            }
            for feature_name in feature_names
        },
    )
    monkeypatch.setattr(
        module,
        "derive_profile_nuisance",
        lambda *args, **kwargs: {"noise_scale_multiplier": 1.0},
    )

    def fake_calibration(*, capability_id, target_values, **kwargs):
        status = "unsupported" if capability_id == "trend" else "supported"
        return (
            {"structure_scale": 1.0},
            [0.1, 0.3, 0.5, 0.7, 0.9],
            {
                "status": status,
                "realized_values": target_values,
                "max_normalized_error": 1.0 if status == "unsupported" else 0.0,
            },
        )

    monkeypatch.setattr(
        module,
        "calibrate_capability_conditioning",
        fake_calibration,
    )

    module.build_suite(
        tmp_path,
        data_dir=tmp_path,
        gift_eval_dir=tmp_path,
        max_windows=60,
        calibration_samples=4,
        seed=11,
    )

    support = json.loads(
        (tmp_path / "dataset_capability_support_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    cells = {
        cell["capability_id"]: cell for cell in support["cells"]
    }
    artifact = json.loads(
        (tmp_path / "generator_conditioning_artifact.json").read_text(
            encoding="utf-8"
        )
    )
    profile = artifact["profiles"][
        module.generator_profile_id(source.dataset_id, source.task_id)
    ]
    assert cells["trend"]["status"] == "unsupported"
    assert cells["trend"]["reason_codes"] == ["conditioning_calibration_failed"]
    assert len(cells) == 9
    assert cells["common_factor"]["status"] == "unsupported"
    assert cells["common_factor"]["reason_codes"] == [
        "variable_structure_not_supported"
    ]
    assert cells["common_factor"]["task_view_audit"]["required_task_id"] == (
        "common_factor"
    )
    assert cells["common_factor"]["task_view_audit"]["available_task_id"] == (
        "univariate"
    )
    assert "trend" not in profile["capabilities"]
    assert artifact["schema_version"].endswith(".v4")
    assert artifact["intensity_policy"]["policy_id"] == (
        "dataset-local-relative-quantiles-v1"
    )
    conditioning = module.resolve_generator_conditioning(
        capability_id="multi_seasonal",
        profile_id=profile["profile_id"],
        context_length=module.MAX_CONTEXT_LENGTH,
        horizon=module.HORIZON,
        target_dim=1,
        artifact=artifact,
    )
    assert conditioning is not None
    assert conditioning.dataset_id == source.dataset_id


def test_qualification_only_runs_supported_dataset_cells(
    module,
    monkeypatch,
    tmp_path,
) -> None:
    supported = {
        "dataset_id": "dataset_a",
        "task_id": "univariate",
        "capability_id": "trend",
        "status": "supported",
        "generator_profile_id": "dataset_a__univariate__L504_H48",
    }
    unsupported = {
        "dataset_id": "dataset_b",
        "task_id": "univariate",
        "capability_id": "trend",
        "status": "unsupported",
        "generator_profile_id": "dataset_b__univariate__L504_H48",
    }
    generator = {
        "profiles": {
            supported["generator_profile_id"]: {
                "target_dim": 1,
                "season_length": 24,
                "hierarchy": None,
            }
        }
    }
    for name, payload in {
        "generator_conditioning_artifact.json": generator,
        "feature_gate_artifact.json": {},
        "near_distance_artifact.json": {},
        "dataset_capability_support_matrix.json": {
            "cells": [supported, unsupported]
        },
        "profile_suite.json": {"support_matrix": [supported, unsupported]},
    }.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        module,
        "resolve_generator_conditioning",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "_generate_sample_values",
        lambda *args, **kwargs: (
            np.arange(module.MAX_CONTEXT_LENGTH + module.HORIZON, dtype=float),
            {"predictability": {"construction_validated": True}},
            None,
        ),
    )
    monkeypatch.setattr(
        module,
        "_realized_features",
        lambda *args, **kwargs: {"trend_strength": 0.5},
    )
    monkeypatch.setattr(
        module,
        "evaluate_feature_support_gate",
        lambda **kwargs: {
            "enforced": True,
            "accepted": True,
            "status": "accepted",
        },
    )
    monkeypatch.setattr(
        module,
        "evaluate_near_distance_gate",
        lambda **kwargs: {
            "enforced": True,
            "accepted": True,
            "status": "accepted",
        },
    )
    monkeypatch.setattr(module, "render_report", lambda *args: "# test\n")
    monkeypatch.setattr(module, "write_manifest", lambda output_dir: None)

    result = module.qualify_suite(
        tmp_path,
        samples_per_cell=1,
        max_attempts=1,
        seed=7,
    )

    assert result["supported_cell_count"] == 1
    assert result["unsupported_cell_count"] == 1
    assert result["expected_sample_count"] == 5
    assert result["accepted_sample_count"] == 5
    assert {
        row["dataset_id"] for row in result["accepted_samples"]
    } == {"dataset_a"}
    assert result["all_supported_cells_qualified"] is True
