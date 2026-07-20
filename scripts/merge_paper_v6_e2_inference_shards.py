#!/usr/bin/env python3
"""Merge disjoint model-level Paper v7 E2 inference shards.

Each shard is produced by ``run_paper_v5_e2_inference.py`` with the same
frozen inputs and a disjoint ``--models`` list.  Prediction files are linked
into the primary output when possible (copied across filesystems), while
statuses and provenance are rebuilt deterministically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge disjoint Paper v7 E2 model inference shards."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("shard_dirs", nargs="+", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def safe_filename(value: str) -> str:
    return "".join(
        character
        if character.isalnum() or character in "-_"
        else "_"
        for character in value
    )


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def input_identity(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "synthetic": config.get("synthetic_input"),
        "real_source": config.get("real_source_input"),
        "context_lengths": config.get("context_lengths"),
        "horizon": config.get("horizon"),
        "input_adaptation_policy": config.get(
            "input_adaptation_policy"
        ),
    }


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha256(destination) != file_sha256(source):
            raise ValueError(f"destination already differs: {destination}")
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def validate_status(
    status: dict[str, Any],
    *,
    model_id: str,
    prediction_path: Path,
) -> None:
    if status.get("status") != "complete":
        raise ValueError(f"incomplete inference shard for {model_id}")
    succeeded = int(status.get("succeeded_count", -1))
    compatible = int(status.get("compatible_sample_count", -2))
    if succeeded != compatible:
        raise ValueError(
            f"inference count mismatch for {model_id}: "
            f"{succeeded}/{compatible}"
        )
    expected_views = status.get("expected_original_view_count")
    unsupported_windows = int(
        status.get("unsupported_window_view_count", 0)
    )
    if (
        expected_views is not None
        and unsupported_windows == 0
        and succeeded != int(expected_views)
    ):
        raise ValueError(
            f"original view coverage mismatch for {model_id}: "
            f"{succeeded}/{expected_views}"
        )
    expected_requests = status.get("expected_http_request_count")
    successful_requests = status.get("successful_http_request_count")
    if (
        expected_requests is not None
        and successful_requests is not None
        and int(expected_requests) != int(successful_requests)
    ):
        raise ValueError(
            f"adapted HTTP request coverage mismatch for {model_id}: "
            f"{successful_requests}/{expected_requests}"
        )
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    observed = count_jsonl(prediction_path)
    if observed != succeeded:
        raise ValueError(
            f"prediction row mismatch for {model_id}: "
            f"{observed}/{succeeded}"
        )


def merge_prediction_kind(
    *,
    output_dir: Path,
    shard_dir: Path,
    model_id: str,
    source_status: dict[str, Any],
    prediction_kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    real = prediction_kind == "real"
    prediction_directory = (
        "real_source_predictions" if real else "predictions"
    )
    failure_directory = "real_failures" if real else "failures"
    filename = f"{safe_filename(model_id)}.jsonl"
    recorded_path = source_status.get("prediction_path")
    if recorded_path:
        source_prediction = Path(str(recorded_path))
        if not source_prediction.is_absolute():
            source_prediction = REPO_ROOT / source_prediction
        source_prediction = source_prediction.resolve()
        expected_parent = (shard_dir / prediction_directory).resolve()
        if source_prediction.parent != expected_parent:
            raise ValueError(
                f"status prediction path escapes shard for {model_id}: "
                f"{source_prediction}"
            )
        filename = source_prediction.name
    else:
        source_prediction = shard_dir / prediction_directory / filename
    validate_status(
        source_status,
        model_id=model_id,
        prediction_path=source_prediction,
    )
    destination_prediction = output_dir / prediction_directory / filename
    transfer = link_or_copy(source_prediction, destination_prediction)
    source_hash = file_sha256(source_prediction)
    destination_hash = file_sha256(destination_prediction)
    if source_hash != destination_hash:
        raise AssertionError(f"prediction hash changed for {model_id}")

    source_failure = shard_dir / failure_directory / filename
    destination_failure = output_dir / failure_directory / filename
    failure_count = 0
    failure_transfer: str | None = None
    failure_hash: str | None = None
    if source_failure.is_file():
        failure_count = count_jsonl(source_failure)
        failure_transfer = link_or_copy(
            source_failure,
            destination_failure,
        )
        failure_hash = file_sha256(source_failure)

    merged_status = {
        **source_status,
        "prediction_path": display_path(destination_prediction),
        "result_origin": display_path(shard_dir),
    }
    provenance = {
        "prediction_kind": prediction_kind,
        "source_path": display_path(source_prediction),
        "primary_path": display_path(destination_prediction),
        "row_count": int(source_status["succeeded_count"]),
        "size_bytes": destination_prediction.stat().st_size,
        "sha256": destination_hash,
        "transfer": transfer,
        "failure_path": (
            None
            if not source_failure.is_file()
            else display_path(destination_failure)
        ),
        "failure_row_count": failure_count,
        "failure_sha256": failure_hash,
        "failure_transfer": failure_transfer,
    }
    return merged_status, provenance


def merge_shards(output_dir: Path, shard_dirs: list[Path]) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    primary_config_path = output_dir / "inference_config.json"
    if not primary_config_path.is_file():
        raise FileNotFoundError(primary_config_path)
    primary_config = read_json(primary_config_path)
    expected_models = [str(value) for value in primary_config["requested_models"]]
    expected_identity = input_identity(primary_config)

    synthetic_models: dict[str, Any] = {}
    real_models: dict[str, Any] = {}
    catalog_models: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    observed_models: set[str] = set()
    for shard_dir_value in shard_dirs:
        shard_dir = shard_dir_value.resolve()
        config_path = shard_dir / "inference_config.json"
        catalog_path = shard_dir / "inference_model_catalog.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        shard_config = read_json(config_path)
        if input_identity(shard_config) != expected_identity:
            raise ValueError(f"inference input mismatch: {shard_dir}")
        shard_models = [str(value) for value in shard_config["requested_models"]]
        if not catalog_path.is_file():
            raise FileNotFoundError(catalog_path)
        shard_catalog = read_json(catalog_path)
        shard_catalog_by_id = {
            str(model["model_id"]): model
            for model in shard_catalog.get("models", [])
        }
        missing_catalog_models = sorted(
            set(shard_models) - set(shard_catalog_by_id)
        )
        if missing_catalog_models:
            raise ValueError(
                f"catalog is missing requested models in {shard_dir}: "
                + ", ".join(missing_catalog_models)
            )
        duplicates = observed_models.intersection(shard_models)
        if duplicates:
            raise ValueError(
                "models appear in multiple inference shards: "
                + ", ".join(sorted(duplicates))
            )
        observed_models.update(shard_models)
        synthetic_statuses = read_json(shard_dir / "model_status.json")[
            "models"
        ]
        real_status_path = shard_dir / "real_source_model_status.json"
        real_statuses = (
            read_json(real_status_path)["models"]
            if real_status_path.is_file()
            else {}
        )
        shard_record: dict[str, Any] = {
            "shard_dir": display_path(shard_dir),
            "service": shard_config.get("service"),
            "models": {},
            "config_sha256": file_sha256(config_path),
            "catalog_sha256": (
                file_sha256(catalog_path)
                if catalog_path.is_file()
                else None
            ),
        }
        for model_id in shard_models:
            catalog_models[model_id] = shard_catalog_by_id[model_id]
            if model_id not in synthetic_statuses:
                raise ValueError(
                    f"missing synthetic status for {model_id}: {shard_dir}"
                )
            synthetic_status, synthetic_record = merge_prediction_kind(
                output_dir=output_dir,
                shard_dir=shard_dir,
                model_id=model_id,
                source_status=synthetic_statuses[model_id],
                prediction_kind="synthetic",
            )
            synthetic_models[model_id] = synthetic_status
            model_record = {"synthetic": synthetic_record}
            if primary_config.get("real_source_input") is not None:
                if model_id not in real_statuses:
                    raise ValueError(
                        f"missing real status for {model_id}: {shard_dir}"
                    )
                real_status, real_record = merge_prediction_kind(
                    output_dir=output_dir,
                    shard_dir=shard_dir,
                    model_id=model_id,
                    source_status=real_statuses[model_id],
                    prediction_kind="real",
                )
                real_models[model_id] = real_status
                model_record["real"] = real_record
            shard_record["models"][model_id] = model_record
        records.append(shard_record)

    if observed_models != set(expected_models):
        missing = sorted(set(expected_models) - observed_models)
        extra = sorted(observed_models - set(expected_models))
        raise ValueError(
            f"inference shard model coverage mismatch: "
            f"missing={missing}, extra={extra}"
        )
    ordered_synthetic = {
        model_id: synthetic_models[model_id]
        for model_id in expected_models
    }
    ordered_real = {
        model_id: real_models[model_id]
        for model_id in expected_models
        if model_id in real_models
    }
    write_json(
        output_dir / "model_status.json",
        {
            "schema_version": "paper_v5_e2_model_status.v1",
            "models": ordered_synthetic,
        },
    )
    if primary_config.get("real_source_input") is not None:
        write_json(
            output_dir / "real_source_model_status.json",
            {
                "schema_version": "paper_v5_e2_model_status.v1",
                "models": ordered_real,
            },
        )
    write_json(
        output_dir / "inference_model_catalog.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "models": [
                catalog_models[model_id]
                for model_id in expected_models
            ],
            "source_catalogs": [
                {
                    "shard_dir": record["shard_dir"],
                    "sha256": record["catalog_sha256"],
                }
                for record in records
            ],
        },
    )
    result = {
        "schema_version": "paper_v7_e2_distributed_inference.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_output_dir": display_path(output_dir),
        "input_identity": expected_identity,
        "expected_models": expected_models,
        "shards": records,
    }
    write_json(output_dir / "inference_shards.json", result)
    return result


def main() -> int:
    args = parse_args()
    result = merge_shards(args.output_dir, args.shard_dirs)
    print(
        f"merged {len(result['shards'])} inference shards for "
        f"{len(result['expected_models'])} models",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
