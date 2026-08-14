from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from cafe import protocol
from cafe.inference import runner as inference


def _file_record(path: Path, *, row_count: int) -> dict[str, object]:
    return {
        **protocol.file_record(path),
        "row_count": row_count,
    }


def _empty_generation_manifest(tmp_path: Path) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    for name in ("clean", "robustness", "input_ablations"):
        path = tmp_path / f"{name}.jsonl"
        row_count = protocol.write_jsonl(path, ())
        files[name] = _file_record(path, row_count=row_count)
    return {
        "config_sha256": "generation-config-sha256",
        "files": files,
    }


def _real_anchored_master(sample_id: str, member: int) -> dict[str, object]:
    length = protocol.CONTEXT_LENGTH + protocol.HORIZON
    target = (np.arange(length, dtype=float) + 1000.0 * member)[:, None]
    future = np.asarray(target[protocol.CONTEXT_LENGTH :], dtype="<f8")
    return {
        "schema_version": "cafe.master_sample.v1",
        "sample_id": sample_id,
        "master_sample_id": sample_id,
        "counterfactual_pair_id": "pair-0",
        "counterfactual_member": member,
        "context_length": protocol.CONTEXT_LENGTH,
        "horizon": protocol.HORIZON,
        "target_dim": 1,
        "covariate_dim": 0,
        "capability_id": "multi_seasonal",
        "evaluation_table": "real_anchored_counterfactual",
        "generation_metadata": {},
        "future_sha256": hashlib.sha256(future.tobytes()).hexdigest(),
        "target": target.tolist(),
        "covariates": None,
    }


def test_prepare_view_tasks_adds_fixed_l168_real_anchored_component(
    tmp_path: Path,
) -> None:
    generation_manifest = _empty_generation_manifest(tmp_path)
    masters_path = tmp_path / "real_anchored__seed_000000_000001.jsonl"
    masters = [
        _real_anchored_master("anchored-m0", 0),
        _real_anchored_master("anchored-m1", 1),
    ]
    row_count = protocol.write_jsonl(masters_path, masters)
    generation_manifest["files"][
        inference.REAL_ANCHORED_GENERATION_FILE_KEY
    ] = _file_record(masters_path, row_count=row_count)

    task_path, manifest = inference.prepare_view_tasks(
        generation_manifest,
        inference_dir=tmp_path / "inference",
    )

    component = manifest["task_components"][
        inference.REAL_ANCHORED_GENERATION_FILE_KEY
    ]
    component_path = Path(component["path"])
    views = list(protocol.iter_jsonl(component_path))
    combined = list(protocol.iter_jsonl(task_path))
    assert manifest["real_anchored_view_count"] == 2
    assert manifest["view_count"] == 2
    assert component["row_count"] == 2
    assert component["sha256"] == protocol.file_sha256(component_path)
    assert combined == views
    assert {row["context_length"] for row in views} == {
        protocol.FIXED_CONTEXT_LENGTH
    }
    assert {row["benchmark_track"] for row in views} == {
        inference.REAL_ANCHORED_BENCHMARK_TRACK
    }
    assert {row["evaluation_table"] for row in views} == {
        inference.REAL_ANCHORED_BENCHMARK_TRACK
    }
    assert all(
        row["context_policy_candidates"] == [protocol.FIXED_CONTEXT_LENGTH]
        for row in views
    )
    assert all(
        len(row["target"])
        == protocol.FIXED_CONTEXT_LENGTH + protocol.HORIZON
        for row in views
    )
    assert views[0]["target"][0] == [
        float(protocol.CONTEXT_LENGTH - protocol.FIXED_CONTEXT_LENGTH)
    ]
    assert inference.validate_inference_task_manifest_files(manifest) == task_path


def test_prepare_view_tasks_preserves_legacy_two_component_manifest(
    tmp_path: Path,
) -> None:
    generation_manifest = _empty_generation_manifest(tmp_path)

    task_path, manifest = inference.prepare_view_tasks(
        generation_manifest,
        inference_dir=tmp_path / "inference",
    )

    assert manifest["real_anchored_view_count"] == 0
    assert inference.REAL_ANCHORED_GENERATION_FILE_KEY not in manifest[
        "task_components"
    ]
    legacy_manifest = dict(manifest)
    legacy_manifest.pop("real_anchored_view_count")
    legacy_manifest.pop("real_anchored_source")
    assert (
        inference.validate_inference_task_manifest_files(legacy_manifest)
        == task_path
    )


def test_prepare_view_tasks_preserves_real_anchored_auxiliary_table(
    tmp_path: Path,
) -> None:
    generation_manifest = _empty_generation_manifest(tmp_path)
    masters_path = tmp_path / "real_anchored_ablation.jsonl"
    masters = [
        _real_anchored_master("ablation-m0", 0),
        _real_anchored_master("ablation-m1", 1),
    ]
    for row in masters:
        row["evaluation_table"] = "real_anchored_input_ablation"
    row_count = protocol.write_jsonl(masters_path, masters)
    generation_manifest["files"][
        inference.REAL_ANCHORED_GENERATION_FILE_KEY
    ] = _file_record(masters_path, row_count=row_count)

    _task_path, manifest = inference.prepare_view_tasks(
        generation_manifest,
        inference_dir=tmp_path / "inference",
    )

    component = manifest["task_components"][
        inference.REAL_ANCHORED_GENERATION_FILE_KEY
    ]
    views = list(protocol.iter_jsonl(Path(component["path"])))
    assert {row["evaluation_table"] for row in views} == {
        "real_anchored_input_ablation"
    }
    assert {row["benchmark_track"] for row in views} == {
        inference.REAL_ANCHORED_BENCHMARK_TRACK
    }


def test_real_anchored_component_hash_is_validated(tmp_path: Path) -> None:
    generation_manifest = _empty_generation_manifest(tmp_path)
    masters_path = tmp_path / "real_anchored.jsonl"
    row_count = protocol.write_jsonl(
        masters_path,
        [_real_anchored_master("anchored-m0", 0)],
    )
    generation_manifest["files"][
        inference.REAL_ANCHORED_GENERATION_FILE_KEY
    ] = _file_record(masters_path, row_count=row_count)
    _task_path, manifest = inference.prepare_view_tasks(
        generation_manifest,
        inference_dir=tmp_path / "inference",
    )
    component_path = Path(
        manifest["task_components"][
            inference.REAL_ANCHORED_GENERATION_FILE_KEY
        ]["path"]
    )
    component_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="byte-size mismatch|hash mismatch"):
        inference.validate_inference_task_manifest_files(manifest)


def test_generation_real_anchored_source_requires_matching_row_count(
    tmp_path: Path,
) -> None:
    generation_manifest = _empty_generation_manifest(tmp_path)
    masters_path = tmp_path / "real_anchored.jsonl"
    protocol.write_jsonl(
        masters_path,
        [_real_anchored_master("anchored-m0", 0)],
    )
    generation_manifest["files"][
        inference.REAL_ANCHORED_GENERATION_FILE_KEY
    ] = _file_record(masters_path, row_count=2)

    with pytest.raises(ValueError, match="row-count mismatch"):
        inference.prepare_view_tasks(
            generation_manifest,
            inference_dir=tmp_path / "inference",
        )
