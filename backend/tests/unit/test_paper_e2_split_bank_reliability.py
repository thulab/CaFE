from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT / "scripts/analyze_paper_e2_split_bank_reliability.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_paper_e2_split_bank_reliability",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fake_oracle(group_count: int = 8) -> pd.DataFrame:
    rows = []
    for capability_index, capability_id in enumerate(("cap1", "cap2")):
        for intensity in (1, 2):
            for group_index in range(group_count):
                for model_index, model_id in enumerate(("m1", "m2", "m3")):
                    rows.append(
                        {
                            "model_id": model_id,
                            "master_sample_id": (
                                f"{capability_id}-g{group_index}-"
                                f"i{intensity}"
                            ),
                            "dataset_id": "dataset",
                            "task_id": "task",
                            "capability_id": capability_id,
                            "intensity": intensity,
                            "paired_group_id": (
                                f"{capability_id}-g{group_index:03d}"
                            ),
                            "pool_index": group_index,
                            "oracle_mase": (
                                1.0
                                + model_index
                                + capability_index * 0.2
                                + intensity * 0.05
                                + group_index * 0.001
                            ),
                            "fixed_l504_mase": (
                                1.1
                                + model_index
                                + capability_index * 0.2
                                + intensity * 0.05
                                + group_index * 0.001
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def test_ordered_split_uses_front_and_back_paired_groups() -> None:
    module = load_module()
    oracle = fake_oracle()

    assignments = module.split_assignments(
        module.pool_catalog(oracle),
        bank_size=3,
        split_kind="ordered",
        repeat_index=0,
        split_seed=7,
    )
    cap1 = assignments[assignments["capability_id"] == "cap1"]

    assert set(cap1[cap1["bank_id"] == "A"]["paired_group_id"]) == {
        "cap1-g000",
        "cap1-g001",
        "cap1-g002",
    }
    assert set(cap1[cap1["bank_id"] == "B"]["paired_group_id"]) == {
        "cap1-g005",
        "cap1-g006",
        "cap1-g007",
    }


def test_random_splits_are_deterministic_disjoint_and_nested() -> None:
    module = load_module()
    catalog = module.pool_catalog(fake_oracle())
    small = module.split_assignments(
        catalog,
        bank_size=2,
        split_kind="random",
        repeat_index=3,
        split_seed=71,
    )
    large = module.split_assignments(
        catalog,
        bank_size=3,
        split_kind="random",
        repeat_index=3,
        split_seed=71,
    )
    again = module.split_assignments(
        catalog,
        bank_size=2,
        split_kind="random",
        repeat_index=3,
        split_seed=71,
    )

    pd.testing.assert_frame_equal(small, again)
    for profile, small_group in small.groupby(module.PROFILE_KEYS):
        large_group = large
        for column, value in zip(module.PROFILE_KEYS, profile, strict=True):
            large_group = large_group[large_group[column] == value]
        for bank_id in ("A", "B"):
            small_ids = set(
                small_group[small_group["bank_id"] == bank_id][
                    "paired_group_id"
                ]
            )
            large_ids = set(
                large_group[large_group["bank_id"] == bank_id][
                    "paired_group_id"
                ]
            )
            assert small_ids <= large_ids
        assert not (
            set(
                small_group[small_group["bank_id"] == "A"][
                    "paired_group_id"
                ]
            )
            & set(
                small_group[small_group["bank_id"] == "B"][
                    "paired_group_id"
                ]
            )
        )


def test_split_analysis_reports_score_rank_and_tie_reliability() -> None:
    module = load_module()
    oracle = fake_oracle()
    assignments = module.split_assignments(
        module.pool_catalog(oracle),
        bank_size=4,
        split_kind="ordered",
        repeat_index=0,
        split_seed=7,
    )

    summary, frames = module.analyze_split(
        oracle,
        assignments,
        score_column="oracle_mase",
        bank_size=4,
        minimum_agreement=0.8,
        equivalence_margins=(0.01, 0.02, 0.05),
        primary_equivalence_margin=0.02,
        pair_bootstrap_replicates=200,
        pair_ci_level=0.95,
        bootstrap_seed=17,
    )

    rank = summary["formal_rank_reliability"]
    assert rank["cell_count"] == 4
    assert rank["pairwise_ordering_agreement"]["mean"] == pytest.approx(1.0)
    assert rank["top1_agreement_rate"] == pytest.approx(1.0)
    pair_summary = summary["tie_aware_model_contrasts"]
    assert pair_summary["primary"]["state_match_rate"] == pytest.approx(1.0)
    assert set(pair_summary["by_margin"]) == {"0.01", "0.02", "0.05"}
    assert summary["partial_order_rank_reliability"][
        "rank_interval_overlap_rate"
    ] == pytest.approx(1.0)
    assert summary["practical_tie_rank_reliability"]["primary"][
        "tie_pair_state_agreement"
    ]["mean"] == pytest.approx(1.0)
    assert set(frames) == {
        "cell_model_scores",
        "cell_model_reliability",
        "capability_profiles",
        "capability_profile_reliability",
        "tie_aware_pair_states",
        "tie_aware_model_contrasts",
        "partial_order_ranks",
        "partial_order_rank_reliability",
        "partial_order_top_tier_reliability",
        "practical_tie_ranks",
        "practical_tie_rank_reliability",
        "practical_tie_top_tier_reliability",
        "rank_reliability",
    }


def test_practical_equivalence_distinguishes_ties_from_uncertainty() -> None:
    module = load_module()

    assert module.practical_equivalence_state(
        ci_low=-0.01,
        ci_high=0.015,
        margin=0.02,
    ) == "equivalent"
    assert module.practical_equivalence_state(
        ci_low=-0.04,
        ci_high=0.01,
        margin=0.02,
    ) == "unresolved"
    assert module.practical_equivalence_state(
        ci_low=-0.08,
        ci_high=-0.03,
        margin=0.02,
    ) == "left_better"


def test_practical_tie_rank_absorbs_small_point_estimate_reversal() -> None:
    module = load_module()
    rows = []
    for bank_id, values in (
        ("A", {"m1": 1.00, "m2": 1.01, "m3": 1.20}),
        ("B", {"m1": 1.01, "m2": 1.00, "m3": 1.20}),
    ):
        for point_rank, (model_id, score) in enumerate(
            sorted(values.items(), key=lambda item: item[1]),
            start=1,
        ):
            rows.append(
                {
                    "bank_id": bank_id,
                    "model_id": model_id,
                    "dataset_id": "dataset",
                    "task_id": "task",
                    "capability_id": "capability",
                    "intensity": 1,
                    "mase_mean": score,
                    "model_rank": point_rank,
                }
            )
    ranks = module.practical_tie_ranks(
        pd.DataFrame(rows),
        equivalence_margins=(0.02,),
    )

    _, _, summaries = module.compare_practical_tie_ranks(ranks)

    assert summaries["0.02"]["tie_pair_state_agreement"]["mean"] == (
        pytest.approx(1.0)
    )
    assert summaries["0.02"]["exact_tie_rank_vector_rate"] == (
        pytest.approx(1.0)
    )
    assert summaries["0.02"]["conclusion_compatibility_rate"] == (
        pytest.approx(1.0)
    )
    assert summaries["0.02"]["top_tier_size_mean"] == pytest.approx(2.0)
    top = ranks[ranks["practical_tie_tier"] == 1]
    assert set(top["model_id"]) == {"m1", "m2"}


def test_pool_validation_rejects_incomplete_model_cell() -> None:
    module = load_module()
    oracle = fake_oracle()
    incomplete = oracle.drop(
        oracle[
            (oracle["model_id"] == "m1")
            & (oracle["capability_id"] == "cap1")
            & (oracle["intensity"] == 1)
        ].index[0]
    )

    with pytest.raises(ValueError, match="complete paired-group pool"):
        module.validate_oracle_pool(incomplete)


def test_end_to_end_writes_flexible_bank_size_outputs(tmp_path) -> None:
    module = load_module()
    e2_dir = tmp_path / "e2"
    oracle_dir = e2_dir / "oracle_sample_scores"
    oracle_dir.mkdir(parents=True)
    oracle = fake_oracle()
    models = ["m1", "m2", "m3"]
    for model_id in models:
        path = oracle_dir / f"{model_id}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in oracle[oracle["model_id"] == model_id].to_dict(
                orient="records"
            ):
                row.pop("pool_index")
                row["prediction_kind"] = "synthetic"
                handle.write(json.dumps(row) + "\n")
    (e2_dir / "inference_config.json").write_text(
        json.dumps({"requested_models": models}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    summary = module.analyze(
        e2_dir,
        output_dir,
        bank_sizes=[2, 4],
        models=models,
        datasets=None,
        random_repeats=2,
        split_seed=11,
        minimum_agreement=0.8,
        equivalence_margins=[0.01, 0.02, 0.05],
        primary_equivalence_margin=0.02,
        pair_bootstrap_replicates=100,
        pair_ci_level=0.95,
    )

    assert summary["bank_sizes"] == [2, 4]
    assert summary["round_interpretation"].startswith(
        "round fields are ignored"
    )
    assert summary["random_split"]["oracle_context"]["4"][
        "repeat_count"
    ] == 2
    comparison = pd.read_csv(output_dir / "split_comparison_summary.csv")
    assert len(comparison) == 12
    assert set(comparison["split_kind"]) == {"ordered", "random"}
    assert (
        output_dir / "ordered_rank_reliability_oracle_context.csv"
    ).is_file()
    assert (
        output_dir
        / "ordered_partial_order_ranks_oracle_context.csv"
    ).is_file()
    assert (
        output_dir
        / "ordered_practical_tie_ranks_oracle_context.csv"
    ).is_file()
    assert (
        output_dir
        / "ordered_rank_reliability_by_capability_oracle_context.csv"
    ).is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "report.md").is_file()
    assert (output_dir / "manifest.json").is_file()
