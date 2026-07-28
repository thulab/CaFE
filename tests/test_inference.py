from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).parents[3]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_script(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_parallel_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parallel_calibration_merge_preserves_declared_capability_order():
    calibration = load_script("calibrate_paper_v8")
    results = {
        "second": {
            "schema_version": "schema",
            "generator_version": "generator",
            "capabilities": {
                "second": {
                    "value": 2,
                    "available_for_generation": False,
                    "unavailable_reason_codes": ["unsupported"],
                }
            },
        },
        "first": {
            "schema_version": "schema",
            "generator_version": "generator",
            "capabilities": {
                "first": {
                    "value": 1,
                    "available_for_generation": True,
                    "unavailable_reason_codes": [],
                }
            },
        },
    }

    merged = calibration.merge_capability_calibrations(
        results,
        ("first", "second"),
    )

    assert list(merged["capabilities"]) == ["first", "second"]
    assert merged["available_capabilities"] == ["first"]
    assert merged["unavailable_capabilities"] == {
        "second": ["unsupported"]
    }


def test_preparation_submission_prioritizes_slow_capabilities():
    common = load_script("paper_v8_pipeline_common")

    order = common.preparation_capability_order(
        ("trend", "common_factor", "cross_series_dependence")
    )

    assert order == (
        "cross_series_dependence",
        "common_factor",
        "trend",
    )


def test_parallel_generation_merge_preserves_shard_order(tmp_path):
    generation = load_script("generate_paper_v8_samples")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "merged.jsonl"
    first.write_bytes(b'{"capability":"first"}\n')
    second.write_bytes(b'{"capability":"second"}\n')

    generation.merge_jsonl_shards(output, (first, second))

    assert output.read_bytes() == (
        b'{"capability":"first"}\n{"capability":"second"}\n'
    )


def test_pipeline_passes_nonsemantic_preparation_worker_count(tmp_path):
    pipeline = load_script("run_paper_v8_pipeline")
    args = SimpleNamespace(
        output_root=tmp_path,
        gift_eval_dir=tmp_path,
        max_anchors=256,
        calibration_seeds=32,
        max_calibration_seeds=96,
        max_generation_attempts=3,
        near_distance_gate=True,
        preparation_workers=8,
        capabilities=["trend"],
        seed_start=0,
        seed_count=64,
        models=["Chronos-2"],
        endpoints=["http://127.0.0.1:10810"],
        devices="0,1",
        endpoint_preset=[],
        endpoint_devices=[],
        endpoint_capacity=[],
        endpoint_concurrency_scale=[],
        endpoint_model_capacity=[],
        endpoint_model_concurrency=[],
        resume_inference=False,
    )

    commands = pipeline.commands_for_dataset(
        args,
        "gift_electricity_h",
        experiment_root=tmp_path,
    )

    calibration_arguments = commands["calibration"][1]
    worker_index = calibration_arguments.index("--workers")
    assert calibration_arguments[worker_index : worker_index + 2] == [
        "--workers",
        "8",
    ]
    generation_arguments = commands["generation"][1]
    worker_index = generation_arguments.index("--workers")
    assert generation_arguments[worker_index : worker_index + 2] == [
        "--workers",
        "8",
    ]
    assert "--max-generation-attempts" in generation_arguments
    assert "--near-distance-gate" in generation_arguments


def test_endpoint_topology_cli_arguments_forwards_default_endpoint_preset():
    inference = load_script("run_paper_v8_inference")
    endpoint = "http://192.168.99.89:10810"
    args = SimpleNamespace(
        endpoints=["http://127.0.0.1:10810", endpoint],
        devices="0,1",
        endpoint_preset=[],
        endpoint_devices=[],
        endpoint_capacity=[],
        endpoint_concurrency_scale=[],
        endpoint_model_capacity=[],
        endpoint_model_concurrency=[],
    )

    arguments = inference.endpoint_topology_cli_arguments(args)

    preset_option = arguments.index("--endpoint-preset")
    assert arguments[preset_option + 1] == (
        f"{endpoint}={inference.RTX5090X8_H48_B1_PRESET}"
    )


def test_dataset_parallel_preparation_continues_after_one_failure(
    monkeypatch,
    tmp_path,
):
    pipeline = load_script("run_paper_v8_pipeline")
    calls: list[str] = []

    def fake_commands(_args, dataset_id, *, experiment_root):
        assert experiment_root == tmp_path
        return {
            step: (f"{dataset_id}:{step}", [])
            for step in ("calibration", "generation", "validation")
        }

    def fake_run(script, _arguments, *, log_path=None):
        assert log_path is not None
        calls.append(script)
        if script == "first:generation":
            raise RuntimeError("expected failure")

    monkeypatch.setattr(pipeline, "commands_for_dataset", fake_commands)
    monkeypatch.setattr(pipeline, "run", fake_run)
    args = SimpleNamespace(
        dataset_workers=2,
        start_at="calibration",
        stop_after="validation",
    )

    completed, failed = pipeline.run_parallel_preparation(
        args,
        ["first", "second"],
        experiment_root=tmp_path,
        experiment_id="experiment",
        protocol_sha256="protocol",
        steps=["calibration", "generation", "validation"],
    )

    assert [item["dataset_id"] for item in completed] == ["second"]
    assert [item["dataset_id"] for item in failed] == ["first"]
    assert failed[0]["failed_step"] == "generation"
    assert "second:validation" in calls
    assert pipeline.v8.read_json(
        tmp_path / "first" / "preparation_status.json"
    )["state"] == "failed"
    assert pipeline.v8.read_json(
        tmp_path / "second" / "preparation_status.json"
    )["state"] == "complete"


def test_dataset_parallelism_accepts_analysis_only_and_rejects_inference():
    pipeline = load_script("run_paper_v8_pipeline")

    assert pipeline.STEPS.index("inference") > pipeline.STEPS.index("validation")
    analysis_index = pipeline.STEPS.index("analysis")
    pipeline.validate_dataset_parallelism(
        dataset_workers=8,
        start_index=analysis_index,
        stop_index=analysis_index,
    )
    with pytest.raises(ValueError, match="analysis-only"):
        pipeline.validate_dataset_parallelism(
            dataset_workers=2,
            start_index=pipeline.STEPS.index("inference"),
            stop_index=pipeline.STEPS.index("inference"),
        )


def test_experiment_analysis_arguments_request_separate_aggregate():
    pipeline = load_script("run_paper_v8_pipeline")
    args = SimpleNamespace(
        seed_start=4,
        seed_count=8,
        models=["a", "b"],
        resume_analysis=True,
    )

    arguments = pipeline.experiment_analysis_arguments(
        args,
        experiment_root=Path("/tmp/experiment"),
    )

    assert arguments == [
        "--aggregate-experiment",
        "--output-root",
        "/tmp/experiment",
        "--seed-start",
        "4",
        "--seed-count",
        "8",
        "--models",
        "a",
        "b",
        "--analysis-profile",
        "full",
        "--reuse-existing-aggregate",
    ]


def test_analysis_only_reuse_forwards_immutable_source_experiment(tmp_path):
    pipeline = load_script("run_paper_v8_pipeline")
    source = tmp_path / "source"
    args = SimpleNamespace(
        output_root=tmp_path,
        gift_eval_dir=tmp_path,
        source_root=None,
        max_anchors=256,
        calibration_seeds=32,
        max_calibration_seeds=96,
        max_generation_attempts=3,
        near_distance_gate=True,
        preparation_workers=1,
        capabilities=["hierarchical_coherence"],
        seed_start=0,
        seed_count=64,
        models=["Chronos-2"],
        endpoints=["http://127.0.0.1:10810"],
        devices="0,1",
        endpoint_preset=[],
        endpoint_devices=[],
        endpoint_capacity=[],
        endpoint_concurrency_scale=[],
        endpoint_model_capacity=[],
        endpoint_model_concurrency=[],
        resume_inference=False,
        resume_analysis=False,
        analysis_profile="scores_only",
        analysis_source_experiment_root=source,
    )

    commands = pipeline.commands_for_dataset(
        args,
        "gift_hierarchical_sales_d",
        experiment_root=tmp_path / "derived",
    )
    aggregate = pipeline.experiment_analysis_arguments(
        args,
        experiment_root=tmp_path / "derived",
    )

    for arguments in (commands["analysis"][1], aggregate):
        option = arguments.index("--source-experiment-root")
        assert arguments[option + 1] == str(source.resolve())
        profile_option = arguments.index("--analysis-profile")
        assert arguments[profile_option + 1] == "scores_only"


def test_reusable_analysis_manifest_validates_binding_and_files(tmp_path):
    pipeline = load_script("run_paper_v8_pipeline")
    dataset_id = "gift_electricity_h"
    shard_name = "seed_000000_000064"
    inference_dir = tmp_path / dataset_id / "03_inference" / shard_name
    analysis_dir = tmp_path / dataset_id / "04_analysis" / shard_name
    inference_manifest_path = inference_dir / "inference_manifest.json"
    result_path = analysis_dir / "scores.json"
    pipeline.v8.write_json(inference_manifest_path, {"complete": True})
    pipeline.v8.write_json(result_path, {"scores": []})
    pipeline.v8.write_json(
        analysis_dir / "analysis_manifest.json",
        {
            "schema_version": "paper_v8_analysis_manifest.v5",
            "dataset_id": dataset_id,
            "inference_manifest_sha256": pipeline.v8.file_sha256(
                inference_manifest_path
            ),
            "models": ["Chronos-2"],
            "analysis_profile": "full",
            "coverage": [
                {
                    "model_id": "Chronos-2",
                    "missing_prediction_count": 0,
                }
            ],
            "files": {"scores": pipeline.v8.file_record(result_path)},
        },
    )

    assert pipeline.reusable_analysis_manifest(
        tmp_path,
        dataset_id=dataset_id,
        seed_start=0,
        seed_count=64,
        models=["Chronos-2"],
    )

    result_path.write_text('{"scores": ["changed"]}\n', encoding="utf-8")
    assert not pipeline.reusable_analysis_manifest(
        tmp_path,
        dataset_id=dataset_id,
        seed_start=0,
        seed_count=64,
        models=["Chronos-2"],
    )


def test_parallel_analysis_reuses_complete_dataset_and_computes_rest(
    monkeypatch,
    tmp_path,
):
    pipeline = load_script("run_paper_v8_pipeline")
    calls: list[str] = []

    def fake_reusable(_root, *, dataset_id, **_arguments):
        return dataset_id == "first"

    def fake_commands(_args, dataset_id, *, experiment_root):
        assert experiment_root == tmp_path
        return {"analysis": (f"{dataset_id}:analysis", [])}

    def fake_run(script, _arguments, *, log_path=None):
        assert log_path is not None
        calls.append(script)

    monkeypatch.setattr(
        pipeline,
        "reusable_analysis_manifest",
        fake_reusable,
    )
    monkeypatch.setattr(pipeline, "commands_for_dataset", fake_commands)
    monkeypatch.setattr(pipeline, "run", fake_run)
    args = SimpleNamespace(
        dataset_workers=2,
        start_at="analysis",
        stop_after="analysis",
        resume_analysis=True,
        seed_start=0,
        seed_count=64,
        models=["Chronos-2"],
    )

    completed, failed = pipeline.run_parallel_analysis(
        args,
        ["first", "second"],
        experiment_root=tmp_path,
        experiment_id="experiment",
        protocol_sha256="protocol",
    )

    assert not failed
    assert [item["dataset_id"] for item in completed] == ["first", "second"]
    assert completed[0]["analysis_status"] == "already_complete"
    assert completed[1]["analysis_status"] == "computed"
    assert calls == ["second:analysis"]


def test_structural_pair_failure_is_available_to_generation_retry(
    monkeypatch,
):
    generation = load_script("generate_paper_v8_samples")
    rows = [
        {
            "evaluation_table": "strict_counterfactual_audit",
            "counterfactual_member": member,
            "counterfactual_pair_id": "pair",
            "capability_id": "cross_series_dependence",
            "context_length": 2,
            "generation_metadata": {},
            "target": [[0.0], [0.0], [0.0]],
        }
        for member in (0, 1)
    ]
    monkeypatch.setattr(
        generation,
        "cross_series_identifiability_gate",
        lambda **_arguments: {
            "accepted": False,
            "enforced": True,
            "reason": "test",
        },
    )

    gate = generation.structural_seed_bundle_gate(rows)

    assert gate is not None
    assert gate["accepted"] is False
    assert gate["pair_id"] == "pair"
    assert all("structural_generation_gate" in row for row in rows)
