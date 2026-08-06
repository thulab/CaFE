from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pa_parquet

from cafe import protocol
from cafe.data import real
from cafe.data.fev_bench import FEV_BENCH_CONFIGS
from cafe.data.fev_bench import FevBenchConfig


def discover_categorical_levels(
    parquet_path: Path,
    columns: list[str],
) -> dict[str, dict[str, Any]]:
    """Discover full categorical support without materializing Python lists."""

    if not columns:
        return {}
    table = pa_parquet.read_table(parquet_path, columns=columns)
    output: dict[str, dict[str, Any]] = {}
    for column in columns:
        levels: set[str] = set()
        null_count = 0
        value_count = 0
        for chunk in table.column(column).chunks:
            if not (
                pa.types.is_list(chunk.type)
                or pa.types.is_large_list(chunk.type)
            ):
                raise ValueError(
                    f"FEV categorical dynamic column {column!r} is not a list"
                )
            null_count += int(chunk.null_count)
            values = chunk.values
            value_count += len(values)
            null_count += int(values.null_count)
            levels.update(
                str(value)
                for value in pc.unique(values).to_pylist()
                if value is not None
            )
        output[column] = {
            "levels": sorted(levels),
            "level_count": len(levels),
            "value_count": value_count,
            "null_count": null_count,
            "null_fraction": (
                null_count / value_count if value_count else 0.0
            ),
        }
    return output


def _task_config(
    task_row: dict[str, Any],
    file_row: dict[str, Any],
    category_scan: dict[str, dict[str, Any]],
) -> FevBenchConfig:
    return FevBenchConfig(
        config_id=str(task_row["config_id"]),
        source_path=str(file_row["path"]),
        frequency=str(task_row["frequency"]),
        target_columns=tuple(str(value) for value in task_row["target_columns"]),
        known_dynamic_columns=tuple(
            str(value) for value in task_row["known_dynamic_columns"]
        ),
        past_dynamic_columns=tuple(
            str(value) for value in task_row["past_dynamic_columns"]
        ),
        static_columns=tuple(str(value) for value in task_row["static_columns"]),
        categorical_dynamic_levels=tuple(
            (
                column,
                tuple(str(value) for value in category_scan[column]["levels"]),
            )
            for column in task_row["categorical_known_columns"]
        ),
        sha256=str(file_row["sha256"]),
        size_bytes=int(file_row["size_bytes"]),
    )


def _feature_qualification(
    anchors: list[dict[str, Any]],
    capability_id: str,
) -> dict[str, Any]:
    summary, qualification = protocol.real_intensity_feature_summary(
        anchors,
        capability_id=capability_id,
    )
    return {
        **qualification,
        "summary": summary,
    }


def qualify_task_view(
    *,
    task_row: dict[str, Any],
    file_row: dict[str, Any],
    data_root: Path,
    maximum_anchors: int,
    minimum_observed_fraction: float,
) -> dict[str, Any]:
    config_id = str(task_row["config_id"])
    parquet_path = data_root / config_id / Path(str(file_row["path"])).name
    category_scan = discover_categorical_levels(
        parquet_path,
        [str(value) for value in task_row["categorical_known_columns"]],
    )
    config = _task_config(task_row, file_row, category_scan)
    previous = FEV_BENCH_CONFIGS.get(config_id)
    FEV_BENCH_CONFIGS[config_id] = config
    try:
        bundle = real.load_real_dataset(
            "fev_parquet",
            data_root / config_id,
        )
        dataset = protocol.DatasetSpec(
            dataset_id=f"fev_phase2_task_{int(task_row['task_index']):03d}",
            logical_name=f"FEV Phase 2 {config_id}",
            config_id=f"fev/{config_id}",
            asset_name=config_id,
            domain="FEV-Bench qualification",
            task_view_id=str(task_row["task_view_id"]),
            real_data_adapter="fev_parquet",
        )
        anchor_error: str | None = None
        anchors: list[dict[str, Any]] = []
        source_metadata: dict[str, Any] = {}
        try:
            anchors, source_metadata = protocol.build_calibration_anchors(
                dataset,
                source_root=data_root,
                maximum_anchors=maximum_anchors,
                minimum_observed_fraction=minimum_observed_fraction,
                real_bundle=bundle,
            )
        except ValueError as error:
            anchor_error = str(error)

        capability_status: dict[str, str] = {}
        feature_qualification: dict[str, dict[str, Any]] = {}
        metadata_statuses = task_row["capability_status"]
        for capability_id in protocol.CAPABILITIES:
            metadata_status = str(metadata_statuses[capability_id])
            if metadata_status.startswith("not_applicable"):
                capability_status[capability_id] = metadata_status
                continue
            if capability_id == "hierarchical_coherence":
                capability_status[capability_id] = (
                    "eligible_via_existing_canonical_adapter"
                    if metadata_status == "use_existing_canonical_adapter"
                    else metadata_status
                )
                continue
            if not anchors:
                capability_status[capability_id] = "rejected_no_usable_anchors"
                continue
            qualification = _feature_qualification(anchors, capability_id)
            feature_qualification[capability_id] = qualification
            capability_status[capability_id] = (
                "eligible"
                if qualification["usable"]
                else "rejected_real_feature_support"
            )

        adapter_metadata = bundle.metadata
        stratum_count = sum(
            (record.values.shape[-1] // protocol.REAL_FORECAST_MASTER_LENGTH)
            * (record.values.shape[0] if record.values.ndim == 2 else 1)
            for record in bundle.records
        )
        return {
            "task_index": int(task_row["task_index"]),
            "task_view_id": str(task_row["task_view_id"]),
            "config_id": config_id,
            "frequency": str(task_row["frequency"]),
            "frequency_class": str(task_row["frequency_class"]),
            "target_columns": list(task_row["target_columns"]),
            "target_count": int(task_row["target_count"]),
            "known_dynamic_columns": list(task_row["known_dynamic_columns"]),
            "categorical_known_columns": list(
                task_row["categorical_known_columns"]
            ),
            "categorical_scan": category_scan,
            "source_is_gift_eval": bool(task_row["source_is_gift_eval"]),
            "existing_cafe_source_overlaps": list(
                task_row["existing_cafe_source_overlaps"]
            ),
            "asset_path": str(parquet_path),
            "asset_sha256": str(file_row["sha256"]),
            "asset_size_bytes": int(file_row["size_bytes"]),
            "native_record_count": len(bundle.records),
            "minimum_length": int(adapter_metadata["minimum_length"]),
            "median_length": float(adapter_metadata["median_length"]),
            "maximum_length": int(adapter_metadata["maximum_length"]),
            "stratum_count": int(stratum_count),
            "accepted_anchor_count": len(anchors),
            "anchor_error": anchor_error,
            "target_value_count": int(adapter_metadata["target_value_count"]),
            "target_nonfinite_count": int(
                adapter_metadata["target_nonfinite_count"]
            ),
            "target_nonfinite_fraction": float(
                adapter_metadata["target_nonfinite_fraction"]
            ),
            "known_covariate_value_count": int(
                adapter_metadata["known_covariate_value_count"]
            ),
            "known_covariate_nonfinite_count": int(
                adapter_metadata["known_covariate_nonfinite_count"]
            ),
            "known_covariate_nonfinite_fraction": float(
                adapter_metadata["known_covariate_nonfinite_fraction"]
            ),
            "rejected_missing_count": int(
                source_metadata.get("rejected_missing_count", 0)
            ),
            "rejected_uninformative_count": int(
                source_metadata.get("rejected_uninformative_count", 0)
            ),
            "feature_qualification": feature_qualification,
            "capability_status": capability_status,
        }
    finally:
        if previous is None:
            del FEV_BENCH_CONFIGS[config_id]
        else:
            FEV_BENCH_CONFIGS[config_id] = previous


def summarize_qualification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = {
        capability_id: dict(
            sorted(
                Counter(
                    row["capability_status"][capability_id] for row in rows
                ).items()
            )
        )
        for capability_id in protocol.CAPABILITIES
    }
    eligible_cells = sum(
        status.startswith("eligible")
        for row in rows
        for status in row["capability_status"].values()
    )
    return {
        "task_count": len(rows),
        "config_count": len({row["config_id"] for row in rows}),
        "task_with_usable_anchor_count": sum(
            int(row["accepted_anchor_count"]) > 0 for row in rows
        ),
        "task_with_minimum_anchor_count": sum(
            int(row["accepted_anchor_count"])
            >= protocol.MIN_REAL_FEATURE_COUNT
            for row in rows
        ),
        "task_with_target_missingness_count": sum(
            int(row["target_nonfinite_count"]) > 0 for row in rows
        ),
        "task_with_known_covariate_missingness_count": sum(
            int(row["known_covariate_nonfinite_count"]) > 0 for row in rows
        ),
        "categorical_task_count": sum(
            bool(row["categorical_known_columns"]) for row in rows
        ),
        "eligible_capability_cells": eligible_cells,
        "default_phase3_qualification_path_units": (
            eligible_cells * protocol.DEFAULT_CALIBRATION_PATH_COUNT
        ),
        "capability_status_counts": status_counts,
    }


def qualification_matrix_csv(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    fields = [
        "task_index",
        "task_view_id",
        "config_id",
        "frequency",
        "target_count",
        "known_dynamic_count",
        "native_record_count",
        "minimum_length",
        "median_length",
        "maximum_length",
        "stratum_count",
        "accepted_anchor_count",
        "target_nonfinite_fraction",
        "known_covariate_nonfinite_fraction",
        "existing_cafe_source_overlaps",
        "anchor_error",
        *protocol.CAPABILITIES,
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                **{field: row.get(field) for field in fields},
                "known_dynamic_count": len(row["known_dynamic_columns"]),
                "existing_cafe_source_overlaps": ";".join(
                    row["existing_cafe_source_overlaps"]
                ),
                **row["capability_status"],
            }
        )
    return output.getvalue()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
