from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).parents[3]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_script(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_calibration_strata_are_nonoverlapping_and_use_floor_capacity():
    common = load_script("paper_v8_pipeline_common")

    strata = common.nonoverlapping_strata(2 * 504 + 37)

    assert len(strata) == 2
    assert all(upper >= lower for lower, upper in strata)
    assert strata[0][1] + 504 <= strata[1][0]


def test_direct_anchor_summary_does_not_compress_feature_values():
    common = load_script("paper_v8_pipeline_common")
    anchor = {"acf1": 0.93, "slope_abs": 0.17}

    summary = common.anchor_summary(anchor)

    assert summary == {
        "acf1": {"p50": 0.93},
        "slope_abs": {"p50": 0.17},
    }


def test_sensitivity_seed_selection_is_prefix_stable():
    generation = load_script("generate_paper_v8_samples")

    first = generation.selected_sensitivity_seeds(
        "gift_electricity_h",
        list(range(1)),
        4,
    )
    expanded = generation.selected_sensitivity_seeds(
        "gift_electricity_h",
        list(range(16)),
        4,
    )

    assert first == expanded.intersection(range(1))
    assert first == set()


def test_response_support_detects_sustained_foldback_without_magic_bound():
    common = load_script("paper_v8_pipeline_common")
    grid = np.linspace(0.0, 1.0, 11)
    response = np.asarray(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.62, 0.55, 0.58]
    )

    index, audit = common.stable_monotone_support(grid, response)

    assert index == 7
    assert audit["foldback_detected"] is True
    assert audit["effective_lambda_support"] == pytest.approx([0.0, 0.7])


def test_master_views_share_exact_future_and_l504_mase_scale():
    common = load_script("paper_v8_pipeline_common")
    time = np.arange(common.MASTER_LENGTH, dtype=float)
    target = (
        np.sin(2 * np.pi * time / 24.0) + 0.002 * time
    )[:, None]
    scale, scale_by_target = common.mase_scales(target, season_length=24)
    master = {
        "sample_id": "v8__demo",
        "master_sample_id": "v8__demo",
        "counterfactual_pair_id": None,
        "capability_id": "trend",
        "context_length": common.CONTEXT_LENGTH,
        "horizon": common.HORIZON,
        "target_dim": 1,
        "covariate_dim": 0,
        "target": target.tolist(),
        "covariates": None,
        "generation_metadata": {},
        "mase_scale": scale,
        "mase_scale_by_target": scale_by_target,
        "future_sha256": "future",
    }

    views = [
        common.master_view(master, context)
        for context in common.VIEW_CONTEXT_LENGTHS
    ]

    futures = [
        np.asarray(view["target"], dtype=float)[view["context_length"] :]
        for view in views
    ]
    assert all(np.array_equal(futures[0], future) for future in futures[1:])
    assert {view["mase_scale"] for view in views} == {scale}
    assert all(
        view["view_standardization_policy"]
        == "slice_exact_l504_standardized_master_without_restandardization"
        for view in views
    )


@pytest.mark.parametrize(
    ("capability_id", "metadata", "field", "expected"),
    [
        (
            "regime_switching",
            {"cut_points": [480, 510]},
            "cut_points",
            [72, 102],
        ),
        (
            "predictable_intermittency",
            {"pulse_centers": [480, 520]},
            "pulse_centers",
            [72, 112],
        ),
    ],
)
def test_short_views_shift_indexed_generation_metadata(
    capability_id,
    metadata,
    field,
    expected,
):
    common = load_script("paper_v8_pipeline_common")
    target = np.arange(common.MASTER_LENGTH, dtype=float)[:, None]
    scale, scale_by_target = common.mase_scales(target, season_length=24)
    master = {
        "sample_id": f"v8__{capability_id}",
        "counterfactual_pair_id": None,
        "capability_id": capability_id,
        "context_length": common.CONTEXT_LENGTH,
        "horizon": common.HORIZON,
        "target_dim": 1,
        "covariate_dim": 0,
        "target": target.tolist(),
        "covariates": None,
        "generation_metadata": metadata,
        "mase_scale": scale,
        "mase_scale_by_target": scale_by_target,
        "future_sha256": "future",
    }

    view = common.master_view(master, 96)

    assert view["generation_metadata"][field] == expected


def test_oracle_context_uses_one_context_for_both_pair_members():
    analysis = load_script("analyze_paper_v8")
    rows = []
    for context, member_mase in (
        (96, (0.5, 1.5)),
        (168, (0.7, 0.7)),
        (336, (0.9, 0.9)),
        (504, (1.0, 1.0)),
    ):
        for member, mase in enumerate(member_mase):
            rows.append(
                {
                    "model_id": "demo",
                    "master_sample_id": f"member-{member}",
                    "master_counterfactual_pair_id": "pair",
                    "counterfactual_member": member,
                    "context_length": context,
                    "metrics": {"mase": mase},
                }
            )

    selected, pair_context = analysis.selected_context_rows(rows)

    oracle = [
        row for row in selected if row["context_policy"] == "oracle_context"
    ]
    assert pair_context[("demo", "pair")] == 168
    assert len(oracle) == 2
    assert {row["context_length"] for row in oracle} == {168}


def test_split_bank_requires_two_batches_for_stability_statistics():
    analysis = load_script("analyze_paper_v8")
    rows = []
    for seed in range(64):
        batch_scale = 1.0 if seed < 32 else 2.0
        for model_id, model_scale in (("a", 1.0), ("b", 2.0)):
            for policy in ("fixed_l504", "oracle_context"):
                rows.append(
                    {
                        "dataset_id": "dataset",
                        "context_policy": policy,
                        "evaluation_table": "main",
                        "generator_family_role": "primary",
                        "capability_id": "trend",
                        "model_id": model_id,
                        "seed_index": seed,
                        "intensity": 5,
                        "metrics": {
                            "mase": batch_scale * model_scale,
                            "trend_slope_relative_abs_error": (
                                batch_scale * model_scale
                            ),
                        },
                    }
                )

    split = analysis.split_bank(
        rows,
        [],
        models=["a", "b"],
        seed_start=0,
        seed_count=64,
    )
    by_key = {
        (
            row["batch_size"],
            row["context_policy"],
            row["capability_id"],
            row["score_kind"],
        ): row
        for row in split
    }
    two_batches = by_key[(32, "fixed_l504", "trend", "accuracy")]
    one_batch = by_key[(64, "fixed_l504", "trend", "accuracy")]

    assert two_batches["mean_kendall_tau_b"] == pytest.approx(1.0)
    assert two_batches["top1_consistency"] == pytest.approx(1.0)
    assert two_batches["mean_top3_overlap"] == pytest.approx(1.0)
    assert two_batches[
        "mean_pairwise_relative_score_difference"
    ] == pytest.approx(2.0 / 3.0)
    assert one_batch["mean_kendall_tau_b"] is None
    assert one_batch["top1_consistency"] is None
    assert one_batch["mean_top3_overlap"] is None
    assert one_batch[
        "mean_pairwise_relative_score_difference"
    ] is None


def test_matched_comparison_excludes_unmatched_clean_seeds_and_intensities():
    analysis = load_script("analyze_paper_v8")

    def row(
        *,
        seed,
        intensity,
        mase,
        family="primary",
        table="main",
    ):
        return {
            "dataset_id": "dataset",
            "context_policy": "fixed_l504",
            "evaluation_table": table,
            "generator_family_role": family,
            "capability_id": "trend",
            "model_id": "model",
            "seed_index": seed,
            "intensity": intensity,
            "metrics": {
                "mase": mase,
                "trend_slope_relative_abs_error": mase,
            },
        }

    comparisons = analysis.matched_comparison_rows(
        [
            row(seed=0, intensity=1, mase=100.0),
            row(seed=1, intensity=5, mase=1.0),
            row(
                seed=1,
                intensity=5,
                mase=2.0,
                family="secondary",
            ),
        ],
        [],
    )

    secondary = next(
        item
        for item in comparisons
        if item["comparison_id"] == "secondary_family"
    )
    assert secondary["matched_seed_count"] == 1
    assert secondary["matched_intensities"] == [5]
    assert secondary["control_accuracy_score"] == pytest.approx(1.0)
    assert secondary["treatment_accuracy_score"] == pytest.approx(2.0)
    assert secondary["accuracy_relative_delta"] == pytest.approx(1.0)


def test_inference_prediction_uses_frozen_mase_scale():
    inference = load_script("run_paper_v8_inference")
    target = np.arange(12, dtype=float)[:, None]
    sample = {
        "sample_id": "view",
        "master_sample_id": "master",
        "dataset_id": "dataset",
        "config_id": "config",
        "profile_id": "profile",
        "capability_id": "trend",
        "generator_family_role": "primary",
        "generator_family_id": "family",
        "evaluation_table": "main",
        "intensity": 1,
        "seed_index": 0,
        "counterfactual_pair_id": None,
        "counterfactual_member": None,
        "context_length": 8,
        "horizon": 4,
        "target_dim": 1,
        "covariate_dim": 0,
        "mase_scale": 2.0,
        "target": target.tolist(),
        "future_sha256": "future",
    }
    forecast = np.zeros((4, 1), dtype=float)

    row = inference.prediction_row(
        "model",
        "foundation",
        sample,
        forecast,
    )

    expected_mae = float(np.mean(np.arange(8, 12)))
    assert row["metrics"]["mae"] == pytest.approx(expected_mae)
    assert row["metrics"]["mase"] == pytest.approx(expected_mae / 2.0)


def test_tail_model_is_partitioned_across_idle_services(tmp_path):
    common = load_script("paper_v8_pipeline_common")
    inference = load_script("run_paper_v8_inference")
    task_path = tmp_path / "tasks.jsonl"
    common.write_jsonl(
        task_path,
        ({"sample_id": f"sample-{index}"} for index in range(30)),
    )
    model_ids = ["Chronos-2", "toto2.0", "tirex2", "timesfm2.5"]
    services = [
        (
            f"http://service-{index}",
            {model_id: {"model_id": model_id} for model_id in model_ids},
        )
        for index in range(3)
    ]

    work, assignments, manifest = inference.plan_inference_work(
        model_ids,
        services,
        task_path=task_path,
        inference_dir=tmp_path / "inference",
        enable_tail_sharding=True,
    )

    assert manifest is not None
    assert manifest["model_id"] == "timesfm2.5"
    assert manifest["part_count"] == 3
    assert sum(part["row_count"] for part in manifest["parts"]) == 30
    assert sorted(
        item.tail_part_index
        for items in work.values()
        for item in items
        if item.model_id == "timesfm2.5"
    ) == [0, 1, 2]
    assert all(
        any(item.model_id == "timesfm2.5" for item in items)
        for items in work.values()
    )
    assert sum(
        model_id == "timesfm2.5"
        for model_ids_for_endpoint in assignments.values()
        for model_id in model_ids_for_endpoint
    ) == 1


def test_tail_predictions_are_merged_only_after_complete_coverage(tmp_path):
    common = load_script("paper_v8_pipeline_common")
    inference = load_script("run_paper_v8_inference")
    model_id = "timesfm2.5"
    model_root = tmp_path / "model_shards" / "timesfm2_5"
    manifest = {
        "model_id": model_id,
        "source_task_row_count": 4,
        "parts": [
            {"part_index": 0, "row_count": 2},
            {"part_index": 1, "row_count": 2},
        ],
    }
    for part_index, sample_ids in enumerate(
        (("sample-0", "sample-2"), ("sample-1", "sample-3"))
    ):
        part_root = (
            model_root / "tail_parts" / f"part_{part_index:03d}"
        )
        common.write_jsonl(
            inference.prediction_path_for(part_root, model_id),
            (
                {"model_id": model_id, "sample_id": sample_id}
                for sample_id in sample_ids
            ),
        )

    inference.consolidate_tail_predictions(tmp_path, manifest)

    canonical = list(
        common.iter_jsonl(
            inference.prediction_path_for(model_root, model_id)
        )
    )
    assert [row["sample_id"] for row in canonical] == [
        "sample-0",
        "sample-1",
        "sample-2",
        "sample-3",
    ]
    statuses = inference.aggregate_model_statuses(
        [model_id],
        [
            {
                "model_id": model_id,
                "status": "complete",
                "endpoint": "service-a",
                "native_view_count": 2,
            },
            {
                "model_id": model_id,
                "status": "complete",
                "endpoint": "service-b",
                "native_view_count": 2,
            },
        ],
        inference_dir=tmp_path,
        expected_view_count=4,
    )
    assert statuses[0]["status"] == "complete"
    assert statuses[0]["native_view_count"] == 4
    assert statuses[0]["endpoints"] == ["service-a", "service-b"]
