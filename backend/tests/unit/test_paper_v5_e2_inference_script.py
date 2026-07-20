from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = REPO_ROOT / "scripts/run_paper_v5_e2_inference.py"
SOURCE_BUILDER_PATH = (
    REPO_ROOT / "scripts/build_paper_v5_real_source_window_suite.py"
)
ANALYSIS_PATH = REPO_ROOT / "scripts/analyze_paper_v5_e2.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_master(module) -> dict:
    time = np.arange(module.MAX_CONTEXT_LENGTH + module.HORIZON)
    target = np.sin(2 * np.pi * time / 24)[:, None]
    return {
        "sample_id": "master-i1",
        "master_sample_id": "master-i1",
        "paired_group_id": "group",
        "profile_id": "dataset__univariate__L504_H48",
        "dataset_id": "dataset",
        "task_id": "univariate",
        "capability_id": "multi_seasonal",
        "intensity": 1,
        "round_index": 1,
        "round_seed": 101,
        "sample_index": 0,
        "context_length": 504,
        "context_lengths": list(module.CONTEXT_LENGTHS),
        "horizon": module.HORIZON,
        "season_length": 24,
        "frequency": "h",
        "target_dim": 1,
        "covariate_dim": 0,
        "hierarchy": None,
        "target": target.tolist(),
        "covariates": None,
        "future_sha256": module.array_sha256(
            target[module.MAX_CONTEXT_LENGTH :]
        ),
    }


def test_master_expands_to_four_independently_standardized_views():
    module = load_module(RUNNER_PATH, "paper_v5_e2_inference_views")
    master = synthetic_master(module)

    views = [
        module.master_view(master, context_length)
        for context_length in module.CONTEXT_LENGTHS
    ]

    assert [view["context_length"] for view in views] == list(
        module.CONTEXT_LENGTHS
    )
    assert len({view["view_id"] for view in views}) == 4
    assert {view["master_sample_id"] for view in views} == {"master-i1"}
    assert {
        view["master_future_sha256"] for view in views
    } == {master["future_sha256"]}
    for view in views:
        context = view["context_length"]
        target = np.asarray(view["target"], dtype=float)
        assert target.shape == (context + module.HORIZON, 1)
        assert abs(float(target[:context].mean())) < 1e-10
        assert abs(float(target[:context].std()) - 1.0) < 1e-10


def test_master_view_slices_explicit_timestamps_with_the_target_suffix():
    module = load_module(RUNNER_PATH, "paper_v5_e2_inference_timestamps")
    master = synthetic_master(module)
    master["timestamps"] = [
        f"2026-01-{1 + index // 48:02d}T{(index % 48) // 2:02d}:"
        f"{30 * (index % 2):02d}:00+00:00"
        for index in range(module.MAX_CONTEXT_LENGTH + module.HORIZON)
    ]

    view = module.master_view(master, 168)

    assert len(view["timestamps"]) == 168 + module.HORIZON
    assert view["timestamps"][0] == master["timestamps"][
        module.MAX_CONTEXT_LENGTH - 168
    ]
    assert view["timestamps"][-1] == master["timestamps"][-1]


def test_prediction_row_retains_view_and_master_identity():
    module = load_module(RUNNER_PATH, "paper_v5_e2_inference_prediction")
    view = module.master_view(synthetic_master(module), 168)
    target = np.asarray(view["target"], dtype=float)
    forecast = target[168:]

    row = module.prediction_row(
        "model",
        "timer_service",
        view,
        forecast,
    )

    assert row["sample_id"] == "master-i1__L168"
    assert row["view_id"] == "master-i1__L168"
    assert row["master_sample_id"] == "master-i1"
    assert row["context_length"] == 168
    assert row["metrics"]["mase"] == 0.0
    assert row["capability_id"] == "multi_seasonal"
    assert row["round_index"] == 1
    assert len(row["target_future"]) == module.HORIZON


def test_inference_config_freezes_v7_input_adaptation_policy(tmp_path):
    module = load_module(RUNNER_PATH, "paper_v5_e2_inference_adaptation_config")
    args = type(
        "Args",
        (),
        {
            "models": ["Timer-3.5"],
            "capabilities": None,
            "devices": "0,1",
            "request_max_attempts": 3,
            "forecast_timeout_seconds": 1200,
            "model_load_timeout_seconds": 1800,
            "base_url": "http://127.0.0.1:10810",
            "api_prefix": "/ai/api/v1",
            "input_adaptation_policy": (
                module.engine.INPUT_ADAPTATION_POLICY_ID
            ),
        },
    )()

    config = module.inference_config(
        args,
        synthetic={"path": "samples.jsonl", "sha256": "sample"},
        real_source=None,
    )

    assert config["input_adaptation_policy"]["policy_id"] == (
        module.engine.INPUT_ADAPTATION_POLICY_ID
    )
    assert config["input_adaptation_policy"]["target_fallback"] == (
        "independent_univariate_then_column_order_reassembly"
    )
    assert config["input_adaptation_policy"]["covariate_fallback"] == (
        "omit_all_covariates_when_model_contract_is_insufficient"
    )


def test_tabpfn_execution_configuration_is_frozen():
    module = load_module(RUNNER_PATH, "paper_v5_e2_inference_config")

    assert module.DEFAULT_MODELS[-1] == "tabpfn-ts3"
    assert module.MODEL_EXECUTION_CONFIG["tabpfn-ts3"] == {
        "replicas_per_device": 8,
        "http_concurrency": 24,
    }


def test_preflight_selects_one_master_per_cell(tmp_path):
    module = load_module(RUNNER_PATH, "paper_v5_e2_inference_preflight")
    source = tmp_path / "source.jsonl"
    rows = []
    for capability in ("trend", "regime_switching"):
        for index in range(2):
            row = synthetic_master(module)
            row["sample_id"] = f"{capability}-{index}"
            row["master_sample_id"] = row["sample_id"]
            row["capability_id"] = capability
            rows.append(row)
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    destination = module.build_preflight_master_file(
        source,
        tmp_path / "preflight.jsonl",
    )
    selected = list(module.iter_jsonl(destination))

    assert len(selected) == 2
    assert {row["capability_id"] for row in selected} == {
        "trend",
        "regime_switching",
    }


def test_capability_subset_retains_every_requested_master(tmp_path):
    module = load_module(
        RUNNER_PATH,
        "paper_v5_e2_inference_capability_subset",
    )
    source = tmp_path / "source.jsonl"
    rows = []
    for capability in ("trend", "regime_switching"):
        for index in range(3):
            row = synthetic_master(module)
            row["sample_id"] = f"{capability}-{index}"
            row["master_sample_id"] = row["sample_id"]
            row["capability_id"] = capability
            rows.append(row)
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    destination = module.build_capability_subset_master_file(
        source,
        tmp_path / "subset.jsonl",
        ["regime_switching"],
    )
    selected = list(module.iter_jsonl(destination))

    assert len(selected) == 3
    assert {
        row["capability_id"] for row in selected
    } == {"regime_switching"}


def test_capability_subset_rejects_missing_capability(tmp_path):
    module = load_module(
        RUNNER_PATH,
        "paper_v5_e2_inference_missing_capability",
    )
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(synthetic_master(module)) + "\n",
        encoding="utf-8",
    )

    try:
        module.build_capability_subset_master_file(
            source,
            tmp_path / "subset.jsonl",
            ["nonlinear_persistence"],
        )
    except ValueError as error:
        assert "nonlinear_persistence" in str(error)
    else:
        raise AssertionError("missing capability should be rejected")


def test_real_source_rows_use_multi_capability_univariate_references():
    module = load_module(
        SOURCE_BUILDER_PATH,
        "paper_v5_real_source_builder",
    )
    references = np.vstack(
        [
            np.arange(module.MAX_CONTEXT_LENGTH + module.HORIZON),
            np.arange(module.MAX_CONTEXT_LENGTH + module.HORIZON) + 1,
        ]
    ).astype(float)
    profile_id = "dataset__univariate__L504_H48"
    rows, support = module.source_rows(
        near_artifact={
            "buckets": {
                profile_id: {
                    "context_length": 504,
                    "horizon": 48,
                    "target_dim": 1,
                    "covariate_dim": 0,
                    "reference_count": 2,
                    "reference_raw": references.tolist(),
                    "season_length": 24,
                    "split": {"policy": "dataset_local"},
                }
            }
        },
        eligible={"dataset": ["trend", "regime_switching"]},
    )

    assert len(rows) == 2
    assert support[0]["status"] == "supported"
    assert support[0]["supported_capability_count"] == 2
    assert rows[0]["context_lengths"] == [96, 168, 336, 504]
    assert np.asarray(rows[0]["target"]).shape == (552, 1)
    assert rows[0]["source_role"].startswith(
        "dataset-local near-distance reference_raw"
    )


def test_cell_rank_stability_is_evaluated_without_cross_cell_pooling():
    module = load_module(ANALYSIS_PATH, "paper_v5_e2_analysis_rank")
    rows = []
    for round_index in range(1, 6):
        for model_id, score in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
            rows.append(
                {
                    "model_id": model_id,
                    "dataset_id": "dataset",
                    "task_id": "univariate",
                    "capability_id": "trend",
                    "intensity": 3,
                    "round_index": round_index,
                    "mase_mean": score + round_index * 0.01,
                    "mase_std": 0.1,
                    "master_sample_count": 32,
                    "score_policy": "oracle_context",
                    "model_rank": float(ord(model_id) - ord("a") + 1),
                    "compatible_model_count": 3,
                }
            )
    frame = module.pd.DataFrame(rows)

    stability = module.rank_stability_rows(frame)

    assert len(stability) == 1
    assert stability.iloc[0]["pairwise_agreement_min"] == 1.0
    assert stability.iloc[0]["exact_rank_vector_pair_rate"] == 1.0
    assert bool(
        stability.iloc[0]["passed_min_pairwise_agreement"]
    )


def test_v7_analysis_protocol_reads_64_by_5_generation_config():
    module = load_module(ANALYSIS_PATH, "paper_v7_e2_analysis_protocol")
    config = {
        "generation_mode": "formal_v7_round_pool",
        "round_seeds": list(module.FORMAL_ROUND_SEEDS),
        "samples_per_round_per_cell": 64,
        "total_per_intensity": 320,
        "analysis_block_contract": {
            "ordering": ["round_index", "sample_index"],
            "total_per_intensity": 320,
            "block_size": 160,
            "mutually_exclusive": True,
            "complete_partition": True,
            "blocks": [
                {"analysis_block_id": "A"},
                {"analysis_block_id": "B"},
            ],
        },
    }

    protocol = module.validate_formal_generation_config(config)

    assert protocol["round_count"] == 5
    assert protocol["samples_per_round"] == 64
    assert protocol["total_per_intensity"] == 320
    assert protocol["analysis_block_size"] == 160
    stale = dict(config, samples_per_round_per_cell=32)
    with pytest.raises(ValueError, match="64 samples"):
        module.validate_formal_generation_config(stale)


def test_v7_full_pool_and_two_block_scores_validate_identity():
    module = load_module(ANALYSIS_PATH, "paper_v7_e2_analysis_blocks")
    protocol = {
        "round_seeds": list(module.FORMAL_ROUND_SEEDS),
        "round_count": 5,
        "samples_per_round": 64,
        "total_per_intensity": 320,
        "analysis_block_size": 160,
        "analysis_block_ids": ["A", "B"],
        "analysis_blocks_mutually_exclusive": True,
    }
    rows = []
    for model_index, model_id in enumerate(("a", "b")):
        for round_index, round_seed in enumerate(
            module.FORMAL_ROUND_SEEDS,
            start=1,
        ):
            for sample_index in range(64):
                pool_index = (round_index - 1) * 64 + sample_index
                rows.append(
                    {
                        "model_id": model_id,
                        "master_sample_id": f"master-{pool_index}",
                        "paired_group_id": f"group-{pool_index}",
                        "dataset_id": "dataset",
                        "task_id": "univariate",
                        "capability_id": "trend",
                        "intensity": 1,
                        "round_index": round_index,
                        "round_seed": round_seed,
                        "sample_index": sample_index,
                        "oracle_mase": (
                            1.0 + model_index + pool_index / 10_000
                        ),
                    }
                )
    oracle = module.validate_formal_oracle_identity(
        module.pd.DataFrame(rows),
        protocol=protocol,
    )

    full = module.cell_full_pool_scores(
        oracle,
        score_column="oracle_mase",
        score_policy="oracle_context",
        expected_samples_per_cell=320,
    )
    blocks = module.cell_analysis_block_scores(
        oracle,
        score_column="oracle_mase",
        score_policy="oracle_context",
        expected_block_size=160,
    )
    stability = module.analysis_block_stability_rows(blocks)

    assert set(full["master_sample_count"]) == {320}
    assert set(blocks["master_sample_count"]) == {160}
    assert set(blocks["analysis_block_id"]) == {"A", "B"}
    assert stability.iloc[0]["pairwise_ordering_agreement"] == 1.0
    assert bool(stability.iloc[0]["exact_rank_vector"])
    incomplete = oracle.iloc[:-1].copy()
    with pytest.raises(ValueError, match="incomplete"):
        module.validate_formal_oracle_identity(
            incomplete,
            protocol=protocol,
        )


def test_oracle_compaction_selects_minimum_mase_and_shorter_tie(tmp_path):
    module = load_module(ANALYSIS_PATH, "paper_v5_e2_analysis_oracle")
    source = tmp_path / "predictions.jsonl"
    rows = []
    for context, mase in ((96, None), (168, 0.5), (336, 0.5), (504, 0.9)):
        rows.append(
            {
                "model_id": "model",
                "model_group": "timer_service",
                "master_sample_id": "master",
                "view_id": f"master__L{context}",
                "dataset_id": "dataset",
                "task_id": "univariate",
                "profile_id": "profile",
                "context_length": context,
                "metrics": {
                    "mase": mase,
                    "mae": 0.4 if mase is None else mase / 2,
                    "mse": 0.8 if mase is None else mase,
                },
                "mase_unavailable_reason": (
                    "flat_history" if mase is None else None
                ),
                "capability_id": "trend",
                "intensity": 2,
                "round_index": 1,
                "round_seed": 101,
                "sample_index": 0,
                "paired_group_id": "group",
            }
        )
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    destination = tmp_path / "oracle.jsonl"

    audit = module.compact_prediction_file(
        source,
        destination,
        model_id="model",
        prediction_kind="synthetic",
    )
    record = next(module.iter_jsonl(destination))

    assert audit["view_count"] == 4
    assert audit["mase_unavailable_view_count"] == 1
    assert audit["master_count"] == 1
    assert record["oracle_context"] == 168
    assert record["oracle_mase"] == 0.5
    assert record["fixed_l504_mase"] == 0.9
    assert record["context_mase"]["96"] is None
    assert record["context_mase_unavailable_reason"] == {
        "96": "flat_history"
    }


def test_source_alignment_reports_top_rank_and_real_mase_gaps():
    module = load_module(ANALYSIS_PATH, "paper_v5_e2_analysis_alignment")
    synthetic = module.pd.DataFrame(
        [
            {
                "dataset_id": "dataset",
                "model_id": model,
                "synthetic_average_rank": rank,
                "effective_capability_count": 3,
                "score_policy": "oracle_context",
            }
            for model, rank in (("a", 1.0), ("b", 2.0), ("c", 3.0))
        ]
    )
    real = module.pd.DataFrame(
        [
            {
                "dataset_id": "dataset",
                "model_id": model,
                "real_source_mean_mase": mase,
                "real_source_rank": rank,
                "real_master_count": 40,
                "score_policy": "oracle_context",
            }
            for model, mase, rank in (
                ("a", 0.5, 1.0),
                ("b", 0.75, 2.0),
                ("c", 1.0, 3.0),
            )
        ]
    )

    row = module.source_alignment_rows(synthetic, real).iloc[0]

    assert row["spearman_rho"] == 1.0
    assert row["synthetic_top1_model"] == "a"
    assert row["synthetic_top1_top2_rank_gap"] == 1.0
    assert row["real_top1_model"] == "a"
    assert row["real_top1_top2_mase_gap"] == 0.25
    assert row["real_top1_top2_relative_mase_gap"] == 0.5
