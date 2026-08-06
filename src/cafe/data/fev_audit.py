from __future__ import annotations

import csv
import fnmatch
import io
import json
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from cafe.data.fev_bench import FEV_BENCH_CONFIGS
from cafe.protocol import CAPABILITIES
from cafe.protocol import MIN_REAL_FEATURE_COUNT
from cafe.protocol import REAL_FORECAST_MASTER_LENGTH


BACKGROUND_CAPABILITIES = (
    "trend",
    "multi_seasonal",
    "time_varying_seasonality",
    "regime_switching",
    "nonlinear_persistence",
    "predictable_intermittency",
)
MULTIVARIATE_CAPABILITIES = (
    "common_factor",
    "cross_series_dependence",
)

# Exact-frequency overlaps with the non-FEV registry. ETT_1H contains the two
# native ETT panels represented separately by gift_ett1_h and gift_ett2_h.
EXISTING_CAFE_SOURCE_OVERLAPS = {
    "ETT_1H": ("gift_ett1_h", "gift_ett2_h"),
    "LOOP_SEATTLE_1H": ("gift_loop_seattle_h",),
    "M_DENSE_1H": ("gift_m_dense_h",),
    "SZ_TAXI_1H": ("gift_sz_taxi_h",),
    "bizitobs_l2c_1H": ("gift_bizitobs_l2c_h",),
    "hierarchical_sales_1D": ("gift_hierarchical_sales_d",),
    "jena_weather_1H": ("gift_jena_weather_h",),
    "restaurant": ("gift_restaurant_d",),
}


def parse_fev_readme(
    text: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not text.startswith("---\n"):
        raise ValueError("FEV dataset README is missing YAML front matter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("FEV dataset README front matter is unterminated")
    metadata = yaml.safe_load(text[4:end])
    if not isinstance(metadata, dict):
        raise ValueError("FEV dataset README front matter is not a mapping")

    statistics: dict[str, dict[str, Any]] = {}
    for line in text[end + 5 :].splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 9:
            raise ValueError(f"unsupported FEV statistics row: {line}")
        config_id = cells[0].strip("`")
        statistics[config_id] = {
            "config_id": config_id,
            "frequency": cells[1],
            "item_count": _parse_table_integer(cells[2]),
            "median_length": _parse_table_integer(cells[3]),
            "observation_count": _parse_table_integer(cells[4]),
            "dynamic_column_count": _parse_table_integer(cells[5]),
            "static_column_count": _parse_table_integer(cells[6]),
            "source": cells[7],
            "citation": cells[8],
        }
    if not statistics:
        raise ValueError("FEV dataset README contains no statistics table")
    return metadata, statistics


def _parse_table_integer(value: str) -> int:
    cleaned = value.replace(",", "").strip()
    if not cleaned or not re.fullmatch(r"[0-9]+", cleaned):
        raise ValueError(f"invalid integer in FEV statistics table: {value!r}")
    return int(cleaned)


def parse_fev_tasks(text: str) -> list[dict[str, Any]]:
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError("FEV task YAML must contain a tasks list")
    tasks = payload["tasks"]
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or "dataset_config" not in task:
            raise ValueError(f"invalid FEV task at index {index}")
    return tasks


def _feature_map(dataset_info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(feature["name"]): feature
        for feature in dataset_info.get("features", [])
    }


def _sequence_dtype(feature: dict[str, Any] | None) -> str | None:
    if not isinstance(feature, dict) or "sequence" not in feature:
        return None
    sequence = feature["sequence"]
    if isinstance(sequence, dict):
        return str(sequence.get("dtype"))
    return str(sequence)


def _is_numeric_sequence(feature: dict[str, Any] | None) -> bool:
    dtype = (_sequence_dtype(feature) or "").lower()
    return bool(re.match(r"^(float|int|uint)", dtype))


def _is_categorical_sequence(feature: dict[str, Any] | None) -> bool:
    return (_sequence_dtype(feature) or "").lower() in {
        "string",
        "bool",
        "boolean",
    }


def _target_columns(task: dict[str, Any]) -> tuple[str, ...]:
    target = task.get("target", "target")
    if isinstance(target, str):
        return (target,)
    if isinstance(target, list) and target and all(
        isinstance(column, str) for column in target
    ):
        return tuple(target)
    raise ValueError(
        f"unsupported target declaration for {task['dataset_config']}: {target!r}"
    )


def _task_view_id(task: dict[str, Any], task_index: int) -> str:
    return str(
        task.get("task_name")
        or f"{task['dataset_config']}__task_{task_index:03d}"
    )


def _frequency_class(frequency: str) -> str:
    text = frequency.strip()
    if re.fullmatch(r"(?i)(?:[0-9]+)?(?:min|t|h|d)", text):
        return "fixed_interval"
    if re.match(r"(?i)^(?:w|m|ms|me|q|qs|qe|y|ys|ye)", text):
        return "calendar_offset"
    return "unknown"


def _base_length_status(median_length: int) -> str:
    if median_length >= REAL_FORECAST_MASTER_LENGTH:
        return "candidate"
    return "requires_length_scan"


def _capability_statuses(
    *,
    base_status: str,
    schema_errors: list[str],
    target_count: int,
    known_dynamic_count: int,
    categorical_known_count: int,
    config_id: str,
) -> dict[str, str]:
    if schema_errors:
        return {capability: "blocked_metadata" for capability in CAPABILITIES}
    statuses = {
        capability: base_status for capability in BACKGROUND_CAPABILITIES
    }
    for capability in MULTIVARIATE_CAPABILITIES:
        statuses[capability] = (
            base_status
            if target_count >= 2
            else "not_applicable_single_target"
        )
    if known_dynamic_count == 0:
        statuses["covariate_response"] = "not_applicable_no_known_covariates"
    elif categorical_known_count:
        statuses["covariate_response"] = (
            "candidate_requires_category_scan"
            if base_status == "candidate"
            else "requires_length_and_category_scan"
        )
    else:
        statuses["covariate_response"] = base_status
    statuses["hierarchical_coherence"] = (
        "use_existing_canonical_adapter"
        if config_id == "hierarchical_sales_1D"
        else "not_applicable_no_explicit_hierarchy"
    )
    return statuses


def _config_file_patterns(metadata: dict[str, Any]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for config in metadata.get("configs", []):
        config_id = str(config["config_name"])
        patterns: list[str] = []
        for data_file in config.get("data_files", []):
            if data_file.get("split") == "train":
                patterns.append(str(data_file["path"]))
        output[config_id] = patterns
    return output


def _parquet_files_for_config(
    config_id: str,
    *,
    patterns: dict[str, list[str]],
    tree_files: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matched = [
        file
        for file in tree_files
        if any(
            fnmatch.fnmatch(str(file["path"]), pattern)
            for pattern in patterns.get(config_id, [])
        )
    ]
    return sorted(matched, key=lambda row: str(row["path"]))


def build_fev_metadata_audit(
    *,
    tasks: list[dict[str, Any]],
    readme_metadata: dict[str, Any],
    statistics: dict[str, dict[str, Any]],
    tree_entries: list[dict[str, Any]],
    maximum_anchors: int = 256,
    calibration_path_count: int = 32,
) -> dict[str, Any]:
    dataset_info = {
        str(row["config_name"]): row
        for row in readme_metadata.get("dataset_info", [])
    }
    patterns = _config_file_patterns(readme_metadata)
    tree_files = [
        row
        for row in tree_entries
        if row.get("type") == "file" and str(row.get("path", "")).endswith(".parquet")
    ]
    task_rows: list[dict[str, Any]] = []
    download_files: dict[str, dict[str, Any]] = {}

    for task_index, task in enumerate(tasks):
        config_id = str(task["dataset_config"])
        info = dataset_info.get(config_id)
        stats = statistics.get(config_id)
        schema_errors: list[str] = []
        if info is None:
            schema_errors.append("missing_dataset_info")
            info = {"features": []}
        if stats is None:
            schema_errors.append("missing_dataset_statistics")
            stats = {
                "frequency": "unknown",
                "item_count": 0,
                "median_length": 0,
                "observation_count": 0,
                "dynamic_column_count": 0,
                "static_column_count": 0,
                "source": "",
                "citation": "",
            }
        features = _feature_map(info)
        targets = _target_columns(task)
        known = tuple(str(value) for value in task.get("known_dynamic_columns", []))
        past = tuple(str(value) for value in task.get("past_dynamic_columns", []))
        static = tuple(str(value) for value in task.get("static_columns", []))
        missing_role_columns = sorted(
            set((*targets, *known, *past, *static)) - set(features)
        )
        if missing_role_columns:
            schema_errors.append(
                "missing_role_columns:" + ",".join(missing_role_columns)
            )
        nonnumeric_targets = [
            column for column in targets if not _is_numeric_sequence(features.get(column))
        ]
        if nonnumeric_targets:
            schema_errors.append(
                "nonnumeric_target_columns:" + ",".join(nonnumeric_targets)
            )
        unsupported_known = [
            column
            for column in known
            if not (
                _is_numeric_sequence(features.get(column))
                or _is_categorical_sequence(features.get(column))
            )
        ]
        if unsupported_known:
            schema_errors.append(
                "unsupported_known_columns:" + ",".join(unsupported_known)
            )
        categorical_known = [
            column
            for column in known
            if _is_categorical_sequence(features.get(column))
        ]
        parquet_files = _parquet_files_for_config(
            config_id,
            patterns=patterns,
            tree_files=tree_files,
        )
        if not parquet_files:
            schema_errors.append("missing_parquet_assets")
        for file in parquet_files:
            file_path = str(file["path"])
            download_files[file_path] = {
                "path": file_path,
                "size_bytes": int(file.get("size", 0)),
                "sha256": str(file.get("lfs", {}).get("oid") or file.get("oid")),
                "configs": sorted(
                    {
                        *download_files.get(file_path, {}).get("configs", []),
                        config_id,
                    }
                ),
            }
        median_length = int(stats["median_length"])
        base_status = _base_length_status(median_length)
        capability_status = _capability_statuses(
            base_status=base_status,
            schema_errors=schema_errors,
            target_count=len(targets),
            known_dynamic_count=len(known),
            categorical_known_count=len(categorical_known),
            config_id=config_id,
        )
        approximate_candidate_windows = min(
            maximum_anchors,
            int(stats["item_count"])
            * len(targets)
            * (median_length // REAL_FORECAST_MASTER_LENGTH),
        )
        task_rows.append(
            {
                "task_index": task_index,
                "task_view_id": _task_view_id(task, task_index),
                "config_id": config_id,
                "frequency": str(stats["frequency"]),
                "frequency_class": _frequency_class(str(stats["frequency"])),
                "horizon_native": int(task["horizon"]),
                "window_count_native": int(task.get("num_windows", 1)),
                "item_count": int(stats["item_count"]),
                "median_length": median_length,
                "observation_count": int(stats["observation_count"]),
                "target_columns": list(targets),
                "target_count": len(targets),
                "known_dynamic_columns": list(known),
                "past_dynamic_columns": list(past),
                "static_columns": list(static),
                "categorical_known_columns": categorical_known,
                "source": str(stats["source"]),
                "citation": str(stats["citation"]),
                "source_is_gift_eval": (
                    "Salesforce/GiftEval" in str(stats["source"])
                ),
                "existing_cafe_source_overlaps": list(
                    EXISTING_CAFE_SOURCE_OVERLAPS.get(config_id, ())
                ),
                "pilot_adapter_registered": config_id in FEV_BENCH_CONFIGS,
                "parquet_files": [str(file["path"]) for file in parquet_files],
                "download_bytes": sum(int(file.get("size", 0)) for file in parquet_files),
                "dataset_bytes": int(info.get("dataset_size", 0)),
                "approximate_candidate_windows_capped": (
                    approximate_candidate_windows
                ),
                "minimum_required_finite_windows": MIN_REAL_FEATURE_COUNT,
                "schema_errors": schema_errors,
                "capability_status": capability_status,
            }
        )

    config_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        config_tasks[str(row["config_id"])].append(row)
    config_rows: list[dict[str, Any]] = []
    for config_id, rows in sorted(config_tasks.items()):
        statuses = {
            capability: sorted(
                {row["capability_status"][capability] for row in rows}
            )
            for capability in CAPABILITIES
        }
        config_rows.append(
            {
                "config_id": config_id,
                "task_view_count": len(rows),
                "frequency": rows[0]["frequency"],
                "frequency_class": rows[0]["frequency_class"],
                "item_count": rows[0]["item_count"],
                "median_length": rows[0]["median_length"],
                "observation_count": rows[0]["observation_count"],
                "source": rows[0]["source"],
                "source_is_gift_eval": rows[0]["source_is_gift_eval"],
                "existing_cafe_source_overlaps": rows[0][
                    "existing_cafe_source_overlaps"
                ],
                "pilot_adapter_registered": rows[0]["pilot_adapter_registered"],
                "download_bytes": rows[0]["download_bytes"],
                "dataset_bytes": rows[0]["dataset_bytes"],
                "schema_errors": sorted(
                    {error for row in rows for error in row["schema_errors"]}
                ),
                "capability_status": statuses,
            }
        )

    status_counts = {
        capability: dict(
            sorted(
                Counter(
                    row["capability_status"][capability] for row in task_rows
                ).items()
            )
        )
        for capability in CAPABILITIES
    }
    candidate_cells = sum(
        status.startswith("candidate")
        for row in task_rows
        for status in row["capability_status"].values()
    )
    conditional_cells = sum(
        status.startswith("requires_length")
        for row in task_rows
        for status in row["capability_status"].values()
    )
    benchmark_configs = set(config_tasks)
    benchmark_files = [
        row
        for row in download_files.values()
        if set(row["configs"]) & benchmark_configs
    ]
    return {
        "task_rows": task_rows,
        "config_rows": config_rows,
        "download_files": sorted(benchmark_files, key=lambda row: row["path"]),
        "summary": {
            "task_count": len(task_rows),
            "config_count": len(config_rows),
            "dataset_card_config_count": len(dataset_info),
            "dataset_card_statistics_count": len(statistics),
            "parquet_file_count": len(benchmark_files),
            "download_bytes": sum(row["size_bytes"] for row in benchmark_files),
            "dataset_bytes": sum(
                row["dataset_bytes"] for row in config_rows
            ),
            "observation_count": sum(
                row["observation_count"] for row in config_rows
            ),
            "metadata_candidate_capability_cells": candidate_cells,
            "conditional_length_scan_cells": conditional_cells,
            "default_qualification_path_units": (
                candidate_cells * calibration_path_count
            ),
            "source_overlap_config_count": sum(
                bool(row["existing_cafe_source_overlaps"])
                for row in config_rows
            ),
            "gift_eval_source_config_count": sum(
                bool(row["source_is_gift_eval"]) for row in config_rows
            ),
            "pilot_registered_config_count": sum(
                bool(row["pilot_adapter_registered"])
                for row in config_rows
            ),
            "schema_error_task_count": sum(
                bool(row["schema_errors"]) for row in task_rows
            ),
            "multivariate_task_count": sum(
                int(row["target_count"]) >= 2 for row in task_rows
            ),
            "known_dynamic_task_count": sum(
                bool(row["known_dynamic_columns"]) for row in task_rows
            ),
            "any_covariate_task_count": sum(
                bool(
                    row["known_dynamic_columns"]
                    or row["past_dynamic_columns"]
                    or row["static_columns"]
                )
                for row in task_rows
            ),
            "categorical_known_task_count": sum(
                bool(row["categorical_known_columns"]) for row in task_rows
            ),
            "median_below_window_config_count": sum(
                int(row["median_length"]) < REAL_FORECAST_MASTER_LENGTH
                for row in config_rows
            ),
            "frequency_class_counts": dict(
                sorted(Counter(row["frequency_class"] for row in config_rows).items())
            ),
            "capability_status_counts": status_counts,
        },
    }


def task_matrix_csv(task_rows: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    fields = [
        "task_index",
        "task_view_id",
        "config_id",
        "frequency",
        "horizon_native",
        "item_count",
        "median_length",
        "target_count",
        "known_dynamic_count",
        "past_dynamic_count",
        "static_count",
        "download_bytes",
        "source_is_gift_eval",
        "existing_cafe_source_overlaps",
        "pilot_adapter_registered",
        "schema_errors",
        *CAPABILITIES,
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in task_rows:
        writer.writerow(
            {
                **{field: row.get(field) for field in fields},
                "known_dynamic_count": len(row["known_dynamic_columns"]),
                "past_dynamic_count": len(row["past_dynamic_columns"]),
                "static_count": len(row["static_columns"]),
                "existing_cafe_source_overlaps": ";".join(
                    row["existing_cafe_source_overlaps"]
                ),
                "schema_errors": ";".join(row["schema_errors"]),
                **row["capability_status"],
            }
        )
    return output.getvalue()


def config_inventory_csv(config_rows: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    fields = [
        "config_id",
        "task_view_count",
        "frequency",
        "frequency_class",
        "item_count",
        "median_length",
        "observation_count",
        "download_bytes",
        "dataset_bytes",
        "source_is_gift_eval",
        "existing_cafe_source_overlaps",
        "pilot_adapter_registered",
        "schema_errors",
        *CAPABILITIES,
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in config_rows:
        writer.writerow(
            {
                **{field: row.get(field) for field in fields},
                "existing_cafe_source_overlaps": ";".join(
                    row["existing_cafe_source_overlaps"]
                ),
                "schema_errors": ";".join(row["schema_errors"]),
                **{
                    capability: ";".join(statuses)
                    for capability, statuses in row["capability_status"].items()
                },
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
