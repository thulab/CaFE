from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


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
            "capabilities": {"second": {"value": 2}},
        },
        "first": {
            "schema_version": "schema",
            "generator_version": "generator",
            "capabilities": {"first": {"value": 1}},
        },
    }

    merged = calibration.merge_capability_calibrations(
        results,
        ("first", "second"),
    )

    assert list(merged["capabilities"]) == ["first", "second"]
    assert merged["capabilities"]["first"] == {"value": 1}
    assert merged["capabilities"]["second"] == {"value": 2}


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
