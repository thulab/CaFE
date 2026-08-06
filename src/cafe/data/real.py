from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pa_parquet

from cafe.data.fev_bench import (
    FEV_CATEGORICAL_MISSING_LEVEL,
    FEV_BENCH_CONFIGS,
    FEV_DATASET_REPOSITORY,
    FEV_DATASET_REVISION,
    FEV_TASK_REPOSITORY,
    FEV_TASK_REVISION,
    FEV_TASKS_SHA256,
    FevBenchConfig,
)

from cafe.features.primitives import (
    M5_COVARIATE_PROVENANCE,
    M5_KNOWN_FUTURE_COVARIATES,
    file_sha256,
    m5_covariate_matrix,
    read_gift_arrow_targets,
)


@dataclass(frozen=True)
class RealSeriesRecord:
    """One forecastable record plus optional structural calibration views."""

    item_id: str
    values: np.ndarray
    channel_ids: tuple[str, ...] = ()
    covariates: np.ndarray | None = None
    covariate_names: tuple[str, ...] = ()
    covariate_kind: str | None = None
    hierarchy_values: np.ndarray | None = None
    hierarchy_kind: str | None = None
    structural_group_id: str | None = None


@dataclass(frozen=True)
class RealDatasetBundle:
    frequency: str
    records: tuple[RealSeriesRecord, ...]
    asset_files: tuple[Path, ...]
    adapter_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


RealDataLoader = Callable[[Path, int | None], RealDatasetBundle]
_LOADERS: dict[str, RealDataLoader] = {}


def register_real_data_adapter(
    adapter_id: str,
) -> Callable[[RealDataLoader], RealDataLoader]:
    def decorator(loader: RealDataLoader) -> RealDataLoader:
        if adapter_id in _LOADERS:
            raise ValueError(f"duplicate CaFE real-data adapter {adapter_id!r}")
        _LOADERS[adapter_id] = loader
        return loader

    return decorator


def load_real_dataset(
    adapter_id: str,
    source_path: Path,
    *,
    record_limit: int | None = None,
) -> RealDatasetBundle:
    try:
        loader = _LOADERS[adapter_id]
    except KeyError as error:
        raise ValueError(
            f"unknown CaFE real-data adapter {adapter_id!r}; "
            f"registered={sorted(_LOADERS)}"
        ) from error
    bundle = loader(source_path.resolve(), record_limit)
    if not bundle.records:
        raise ValueError(
            f"CaFE real-data adapter {adapter_id!r} returned no records"
        )
    return bundle


@register_real_data_adapter("gift_arrow")
def load_gift_arrow(
    asset_path: Path,
    record_limit: int | None,
) -> RealDatasetBundle:
    frequency, native_records = read_gift_arrow_targets(asset_path)
    selected = (
        native_records
        if record_limit is None
        else native_records[: int(record_limit)]
    )
    records = tuple(
        RealSeriesRecord(
            item_id=str(item_id),
            values=np.asarray(values, dtype=float),
        )
        for item_id, values in selected
    )
    return RealDatasetBundle(
        frequency=frequency,
        records=records,
        asset_files=tuple(sorted(asset_path.glob("data-*.arrow"))),
        adapter_id="gift_arrow",
        metadata={
            "record_selection": (
                "source_order_prefix" if record_limit is not None else "all"
            ),
        },
    )


def _fev_config_for_path(asset_path: Path) -> FevBenchConfig:
    try:
        return FEV_BENCH_CONFIGS[asset_path.name]
    except KeyError as error:
        raise ValueError(
            f"unsupported FEV-Bench config directory {asset_path.name!r}; "
            f"registered={sorted(FEV_BENCH_CONFIGS)}"
        ) from error


def _fev_sequence(
    table: pa.Table,
    *,
    row_index: int,
    column: str,
    expected_length: int,
) -> np.ndarray:
    values = np.asarray(table.column(column)[row_index].as_py(), dtype=float)
    if values.ndim != 1 or values.shape != (expected_length,):
        raise ValueError(
            f"FEV column {column!r} row {row_index} has shape "
            f"{values.shape}; expected {(expected_length,)}"
        )
    return values


def _validate_fev_timestamps(
    timestamps: list[Any],
    *,
    config: FevBenchConfig,
    row_index: int,
) -> int:
    index = pd.DatetimeIndex(timestamps)
    if index.empty or index.hasnans:
        raise ValueError(
            f"FEV {config.config_id} row {row_index} has empty/invalid timestamps"
        )
    if len(index) > 1:
        try:
            expected = pd.date_range(
                start=index[0],
                periods=len(index),
                freq=config.frequency,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"FEV {config.config_id} has unsupported frequency "
                f"{config.frequency!r}"
            ) from error
        if not index.equals(expected):
            raise ValueError(
                f"FEV {config.config_id} row {row_index} is not regular "
                f"at frequency {config.frequency!r}"
            )
    return len(index)


def _fev_known_covariates(
    table: pa.Table,
    *,
    row_index: int,
    config: FevBenchConfig,
    expected_length: int,
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    if not config.known_dynamic_columns:
        return None, ()
    categorical_levels = dict(config.categorical_dynamic_levels)
    columns: list[np.ndarray] = []
    names: list[str] = []
    for column in config.known_dynamic_columns:
        levels = categorical_levels.get(column)
        if levels is None:
            columns.append(
                _fev_sequence(
                    table,
                    row_index=row_index,
                    column=column,
                    expected_length=expected_length,
                )
            )
            names.append(column)
            continue
        raw_values = table.column(column)[row_index].as_py()
        if len(raw_values) != expected_length:
            raise ValueError(
                f"FEV categorical column {column!r} row {row_index} has "
                "an invalid length"
            )
        values = np.asarray(
            [
                FEV_CATEGORICAL_MISSING_LEVEL
                if value is None
                else str(value)
                for value in raw_values
            ],
            dtype=object,
        )
        unexpected = sorted(set(values) - set(levels))
        if unexpected:
            raise ValueError(
                f"FEV categorical column {column!r} has unexpected levels: "
                f"{unexpected}"
            )
        for level in levels:
            encoded = (values == level).astype(float)
            columns.append(encoded)
            names.append(f"{column}={level}")
    return np.column_stack(columns), tuple(names)


@register_real_data_adapter("fev_parquet")
def load_fev_parquet(
    asset_path: Path,
    record_limit: int | None,
) -> RealDatasetBundle:
    """Load one pinned FEV-Bench task view from local Parquet shards."""

    config = _fev_config_for_path(asset_path)
    if not asset_path.is_dir():
        raise FileNotFoundError(f"FEV config directory not found: {asset_path}")
    parquet_files = tuple(sorted(asset_path.glob("train-*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(
            f"FEV config {config.config_id} has no train-*.parquet shards"
        )
    expected_files = (asset_path / config.parquet_name,)
    if parquet_files != expected_files:
        raise ValueError(
            f"FEV {config.config_id} assets do not match the pinned contract; "
            f"expected={[path.name for path in expected_files]}, "
            f"actual={[path.name for path in parquet_files]}"
        )
    actual_sha256 = file_sha256(parquet_files[0])
    if actual_sha256 != config.sha256:
        raise ValueError(
            f"FEV {config.config_id} checksum mismatch: expected "
            f"{config.sha256}, got {actual_sha256}"
        )
    if parquet_files[0].stat().st_size != config.size_bytes:
        raise ValueError(
            f"FEV {config.config_id} size mismatch: expected "
            f"{config.size_bytes}, got {parquet_files[0].stat().st_size}"
        )
    required_columns = {
        "id",
        "timestamp",
        *config.target_columns,
        *config.known_dynamic_columns,
        *config.past_dynamic_columns,
        *config.static_columns,
    }
    available_columns = {
        column
        for path in parquet_files
        for column in pa_parquet.ParquetFile(path).schema_arrow.names
    }
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise ValueError(
            f"FEV {config.config_id} is missing columns: {missing_columns}"
        )
    consumed_columns = sorted(
        {
            "id",
            "timestamp",
            *config.target_columns,
            *config.known_dynamic_columns,
        }
    )
    tables = [
        pa_parquet.read_table(path, columns=consumed_columns)
        for path in parquet_files
    ]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)

    item_ids = [str(value) for value in table.column("id").to_pylist()]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError(f"FEV {config.config_id} contains duplicate item IDs")
    records: list[RealSeriesRecord] = []
    lengths: list[int] = []
    target_value_count = 0
    target_nonfinite_count = 0
    covariate_value_count = 0
    covariate_nonfinite_count = 0
    for row_index, item_id in enumerate(item_ids):
        timestamps = table.column("timestamp")[row_index].as_py()
        target_length = _validate_fev_timestamps(
            timestamps,
            config=config,
            row_index=row_index,
        )
        target_values = np.vstack(
            [
                _fev_sequence(
                    table,
                    row_index=row_index,
                    column=column,
                    expected_length=target_length,
                )
                for column in config.target_columns
            ]
        )
        target_value_count += int(target_values.size)
        target_nonfinite_count += int(
            target_values.size - np.count_nonzero(np.isfinite(target_values))
        )
        covariates, covariate_names = _fev_known_covariates(
            table,
            row_index=row_index,
            config=config,
            expected_length=target_length,
        )
        if covariates is not None:
            covariate_value_count += int(covariates.size)
            covariate_nonfinite_count += int(
                covariates.size
                - np.count_nonzero(np.isfinite(covariates))
            )
        records.append(
            RealSeriesRecord(
                item_id=item_id,
                values=target_values,
                channel_ids=config.target_columns,
                covariates=covariates,
                covariate_names=covariate_names,
                covariate_kind=(
                    "known_future" if covariates is not None else None
                ),
            )
        )
        lengths.append(target_length)

    selected = (
        records if record_limit is None else records[: int(record_limit)]
    )
    config_payload = json.dumps(
        asdict(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return RealDatasetBundle(
        frequency=config.frequency,
        records=tuple(selected),
        asset_files=parquet_files,
        adapter_id="fev_parquet",
        metadata={
            "dataset_repository": FEV_DATASET_REPOSITORY,
            "dataset_revision": FEV_DATASET_REVISION,
            "task_repository": FEV_TASK_REPOSITORY,
            "task_revision": FEV_TASK_REVISION,
            "tasks_sha256": FEV_TASKS_SHA256,
            "config_id": config.config_id,
            "config_contract_sha256": hashlib.sha256(config_payload).hexdigest(),
            "target_columns": list(config.target_columns),
            "known_dynamic_columns": list(config.known_dynamic_columns),
            "categorical_dynamic_levels": {
                column: list(levels)
                for column, levels in config.categorical_dynamic_levels
            },
            "known_covariate_output_columns": list(
                records[0].covariate_names
            ),
            "past_dynamic_columns": list(config.past_dynamic_columns),
            "past_dynamic_policy": (
                "retained_in_source_provenance_not_exposed_as_known_future"
            ),
            "static_columns": list(config.static_columns),
            "static_policy": "retained_in_source_provenance_not_used_by_cafe_v1",
            "source_record_count": len(records),
            "selected_record_count": len(selected),
            "record_selection": (
                "source_order_prefix" if record_limit is not None else "all"
            ),
            "minimum_length": min(lengths),
            "median_length": float(np.median(lengths)),
            "maximum_length": max(lengths),
            "target_value_count": target_value_count,
            "target_nonfinite_count": target_nonfinite_count,
            "target_nonfinite_fraction": (
                target_nonfinite_count / target_value_count
                if target_value_count
                else 0.0
            ),
            "known_covariate_value_count": covariate_value_count,
            "known_covariate_nonfinite_count": covariate_nonfinite_count,
            "known_covariate_nonfinite_fraction": (
                covariate_nonfinite_count / covariate_value_count
                if covariate_value_count
                else 0.0
            ),
            "asset_sha256": {
                path.name: file_sha256(path) for path in parquet_files
            },
        },
    )


_HIERARCHICAL_SALES_ITEM_PATTERN = re.compile(
    r"QTY_(B[1-4])_([1-9][0-9]*)"
)
_HIERARCHICAL_SALES_PROMO_PATTERN = re.compile(
    r"PROMO_(B[1-4])_([1-9][0-9]*)"
)
_HIERARCHICAL_SALES_EXPECTED_BRAND_COUNTS = {
    "B1": 42,
    "B2": 45,
    "B3": 21,
    "B4": 10,
}


def _hierarchical_sales_arrow_table(
    asset_path: Path,
) -> tuple[pa.Table, Path]:
    if not asset_path.is_dir():
        raise FileNotFoundError(
            f"Hierarchical Sales directory not found: {asset_path}"
        )
    arrow_files = sorted(asset_path.glob("data-*.arrow"))
    if len(arrow_files) != 1:
        raise ValueError(
            "expected exactly one canonical data-*.arrow in "
            f"{asset_path}, got {len(arrow_files)}"
        )
    arrow_path = arrow_files[0]
    with pa.memory_map(str(arrow_path), "r") as source:
        table = pa_ipc.open_stream(source).read_all()
    required = {"item_id", "start", "target", "freq"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(
            f"Hierarchical Sales Arrow is missing columns: {', '.join(missing)}"
        )
    return table, arrow_path


def _hierarchical_sales_promotion_covariates(
    asset_path: Path,
    *,
    expected_targets: dict[str, np.ndarray],
    start: pd.Timestamp,
    target_length: int,
) -> tuple[dict[str, np.ndarray] | None, Path | None, dict[str, Any]]:
    expected_item_ids = set(expected_targets)
    promotion_path = asset_path / "hierarchical_sales_data.csv"
    if not promotion_path.is_file():
        return (
            None,
            None,
            {
                "available": False,
                "reason": "hierarchical_sales_data.csv not found",
                "expected_asset": promotion_path.name,
                "missing_date_fill_policy": "not_applied_without_source",
            },
        )

    frame = pd.read_csv(promotion_path)
    if "DATE" not in frame.columns:
        raise ValueError("Hierarchical Sales promotion CSV is missing DATE")
    qty_columns = {
        str(column)
        for column in frame.columns
        if _HIERARCHICAL_SALES_ITEM_PATTERN.fullmatch(str(column))
    }
    expected_promo_columns = {
        item_id.replace("QTY_", "PROMO_", 1)
        for item_id in expected_item_ids
    }
    promo_columns = {
        str(column)
        for column in frame.columns
        if _HIERARCHICAL_SALES_PROMO_PATTERN.fullmatch(str(column))
    }
    if qty_columns != expected_item_ids:
        raise ValueError(
            "Hierarchical Sales promotion CSV QTY columns do not match the "
            f"118-leaf contract; missing={sorted(expected_item_ids - qty_columns)}, "
            f"unexpected={sorted(qty_columns - expected_item_ids)}"
        )
    if promo_columns != expected_promo_columns:
        raise ValueError(
            "Hierarchical Sales promotion CSV PROMO columns do not map "
            "one-to-one to QTY leaves; "
            f"missing={sorted(expected_promo_columns - promo_columns)}, "
            f"unexpected={sorted(promo_columns - expected_promo_columns)}"
        )

    parsed_dates = pd.to_datetime(frame["DATE"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError(
            "Hierarchical Sales promotion CSV contains invalid DATE values"
        )
    normalized_dates = pd.DatetimeIndex(parsed_dates).normalize()
    if normalized_dates.duplicated().any():
        raise ValueError(
            "Hierarchical Sales promotion CSV contains duplicate DATE values"
        )

    ordered_promo_columns = sorted(
        expected_promo_columns,
        key=lambda value: (
            int(_HIERARCHICAL_SALES_PROMO_PATTERN.fullmatch(value).group(1)[1:]),
            int(_HIERARCHICAL_SALES_PROMO_PATTERN.fullmatch(value).group(2)),
        ),
    )
    raw_promotions = frame[ordered_promo_columns]
    numeric_promotions = raw_promotions.apply(pd.to_numeric, errors="coerce")
    invalid_cells = raw_promotions.notna() & numeric_promotions.isna()
    if invalid_cells.to_numpy().any():
        raise ValueError(
            "Hierarchical Sales promotion CSV contains non-numeric PROMO values"
        )
    numeric_promotions.index = normalized_dates
    finite_promotion_values = numeric_promotions.to_numpy(dtype=float)
    finite_promotion_values = finite_promotion_values[
        np.isfinite(finite_promotion_values)
    ]
    if not np.isin(finite_promotion_values, (0.0, 1.0)).all():
        raise ValueError(
            "Hierarchical Sales PROMO values must be binary presence indicators"
        )

    expected_dates = pd.date_range(
        start=start.normalize(),
        periods=target_length,
        freq="D",
    )
    overlap_count = int(expected_dates.isin(normalized_dates).sum())
    if overlap_count == 0:
        raise ValueError(
            "Hierarchical Sales promotion CSV has no DATE overlap with Arrow"
        )
    aligned = numeric_promotions.reindex(expected_dates)
    numeric_qty = frame[sorted(qty_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    invalid_qty_cells = frame[sorted(qty_columns)].notna() & numeric_qty.isna()
    if invalid_qty_cells.to_numpy().any():
        raise ValueError(
            "Hierarchical Sales promotion CSV contains non-numeric QTY values"
        )
    numeric_qty.index = normalized_dates
    aligned_qty = numeric_qty.reindex(expected_dates)
    mismatched_qty = [
        item_id
        for item_id, arrow_values in sorted(expected_targets.items())
        if not np.array_equal(
            aligned_qty[item_id].to_numpy(dtype=float),
            np.asarray(arrow_values, dtype=float),
            equal_nan=True,
        )
    ]
    if mismatched_qty:
        raise ValueError(
            "Hierarchical Sales promotion CSV QTY values do not match Arrow "
            f"after daily DATE alignment: {mismatched_qty}"
        )
    missing_date_count = int((~expected_dates.isin(normalized_dates)).sum())
    missing_value_count = int(aligned.isna().to_numpy().sum())
    aligned = aligned.fillna(0.0)
    aligned_values = aligned.to_numpy(dtype=float)
    if not np.isfinite(aligned_values).all():
        raise ValueError(
            "Hierarchical Sales promotion CSV contains non-finite PROMO values"
        )

    promotion_by_item_id = {
        promo_name.replace("PROMO_", "QTY_", 1): aligned[promo_name].to_numpy(
            dtype=float,
        )
        for promo_name in ordered_promo_columns
    }
    return (
        promotion_by_item_id,
        promotion_path,
        {
            "available": True,
            "source": "original hierarchical_sales_data.csv promotion indicator",
            "source_dataset_doi": "10.17632/njdkntcpc9.1",
            "source_download_url": (
                "https://data.mendeley.com/public-files/datasets/"
                "njdkntcpc9/files/08bb4f43-6dfa-4995-b268-"
                "42fa0690ba6b/file_downloaded"
            ),
            "covariate_kind": "known_future",
            "qty_column_count": len(qty_columns),
            "promo_column_count": len(promo_columns),
            "date_alignment": "Arrow start plus complete daily target index",
            "source_date_count": len(normalized_dates),
            "aligned_date_count": len(expected_dates),
            "overlap_date_count": overlap_count,
            "qty_values_match_arrow": True,
            "missing_date_count": missing_date_count,
            "missing_value_count": missing_value_count,
            "missing_date_fill_policy": (
                "missing dates and missing PROMO cells are filled with 0"
            ),
        },
    )


@register_real_data_adapter("gift_hierarchical_sales")
def load_gift_hierarchical_sales(
    asset_path: Path,
    record_limit: int | None,
) -> RealDatasetBundle:
    """Restore deterministic additive sibling pairs from Hierarchical Sales."""

    table, arrow_path = _hierarchical_sales_arrow_table(asset_path)
    item_ids = [str(value) for value in table.column("item_id").to_pylist()]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Hierarchical Sales contains duplicate item IDs")

    expected_ids = {
        f"QTY_{brand}_{index}"
        for brand, count in _HIERARCHICAL_SALES_EXPECTED_BRAND_COUNTS.items()
        for index in range(1, count + 1)
    }
    actual_ids = set(item_ids)
    if actual_ids != expected_ids:
        missing_ids = sorted(expected_ids - actual_ids)
        unexpected_ids = sorted(actual_ids - expected_ids)
        raise ValueError(
            "Hierarchical Sales item IDs do not match the fail-closed "
            f"118-leaf contract; missing={missing_ids}, "
            f"unexpected={unexpected_ids}"
        )

    frequencies = {
        str(value) for value in table.column("freq").to_pylist()
    }
    if frequencies != {"D"}:
        raise ValueError(
            "Hierarchical Sales must have one daily frequency; "
            f"got {sorted(frequencies)}"
        )
    starts = table.column("start").to_pylist()
    if any(value is None for value in starts) or len(set(starts)) != 1:
        raise ValueError(
            "Hierarchical Sales leaves must have one non-null common start"
        )

    values_by_id: dict[str, np.ndarray] = {}
    lengths: set[int] = set()
    for item_id, target in zip(
        item_ids,
        table.column("target").to_pylist(),
        strict=True,
    ):
        match = _HIERARCHICAL_SALES_ITEM_PATTERN.fullmatch(item_id)
        if match is None:
            raise ValueError(
                f"unsupported Hierarchical Sales item ID: {item_id!r}"
            )
        values = np.asarray(target, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(
                f"Hierarchical Sales item {item_id!r} has unsupported "
                f"target shape {values.shape}"
            )
        values_by_id[item_id] = values
        lengths.add(int(values.size))
    if len(lengths) != 1:
        raise ValueError(
            "Hierarchical Sales leaves must have one common target length; "
            f"got {sorted(lengths)}"
        )
    target_length = next(iter(lengths))
    common_start = pd.Timestamp(starts[0])
    (
        promotion_by_item_id,
        promotion_path,
        promotion_metadata,
    ) = _hierarchical_sales_promotion_covariates(
        asset_path,
        expected_targets=values_by_id,
        start=common_start,
        target_length=target_length,
    )

    records: list[RealSeriesRecord] = []
    unpaired_child_ids: list[str] = []
    for brand, count in _HIERARCHICAL_SALES_EXPECTED_BRAND_COUNTS.items():
        brand_ids = [f"QTY_{brand}_{index}" for index in range(1, count + 1)]
        for offset in range(0, count - 1, 2):
            channel_ids = tuple(brand_ids[offset : offset + 2])
            children = np.vstack(
                [values_by_id[channel_id] for channel_id in channel_ids]
            )
            covariate_names = tuple(
                channel_id.replace("QTY_", "PROMO_", 1)
                for channel_id in channel_ids
            )
            covariates = (
                None
                if promotion_by_item_id is None
                else np.column_stack(
                    [
                        promotion_by_item_id[channel_id]
                        for channel_id in channel_ids
                    ]
                )
            )
            records.append(
                RealSeriesRecord(
                    item_id=(
                        f"hierarchical_sales:{brand}:"
                        f"{offset + 1}-{offset + 2}"
                    ),
                    values=np.asarray(children, dtype=float),
                    channel_ids=channel_ids,
                    covariates=covariates,
                    covariate_names=(
                        covariate_names if covariates is not None else ()
                    ),
                    covariate_kind=(
                        "known_future" if covariates is not None else None
                    ),
                    hierarchy_values=np.asarray(children, dtype=float),
                    hierarchy_kind="children_only_additive",
                    structural_group_id=f"hierarchical_sales:{brand}",
                )
            )
        if count % 2:
            unpaired_child_ids.append(brand_ids[-1])

    eligible_pair_count = len(records)
    selected = (
        records if record_limit is None else records[: int(record_limit)]
    )
    asset_files = (
        (arrow_path,)
        if promotion_path is None
        else (arrow_path, promotion_path)
    )
    return RealDatasetBundle(
        frequency="D",
        records=tuple(selected),
        asset_files=asset_files,
        adapter_id="gift_hierarchical_sales",
        metadata={
            "record_selection": (
                "natural_sibling_pair_prefix"
                if record_limit is not None
                else "all_natural_sibling_pairs"
            ),
            "validated_leaf_count": len(values_by_id),
            "validated_brand_counts": dict(
                _HIERARCHICAL_SALES_EXPECTED_BRAND_COUNTS
            ),
            "target_length": target_length,
            "common_start": common_start.isoformat(),
            "eligible_pair_count": eligible_pair_count,
            "selected_pair_count": len(selected),
            "unpaired_child_ids": unpaired_child_ids,
            "panel_contract": (
                "two consecutive leaves from one validated brand, paired in "
                "natural item-number order without overlap"
            ),
            "hierarchy_contract": (
                "children_only_additive: the synthetic hierarchy parent is "
                "constructed as the exact sum of the two declared children"
            ),
            "hierarchy_provenance": {
                "source": "GIFT-Eval hierarchical_sales/D Arrow",
                "item_id_pattern": _HIERARCHICAL_SALES_ITEM_PATTERN.pattern,
                "grouping_key": "B1-B4 brand token in item_id",
                "pairing": "consecutive natural-order leaves within brand",
                "validation": (
                    "exact IDs, brand counts, daily frequency, common start, "
                    "common non-empty target length"
                ),
            },
            "promotion_covariates": promotion_metadata,
            "known_future_covariate_provenance": (
                promotion_metadata.get("source")
                if promotion_metadata["available"]
                else None
            ),
            "asset_sha256": {
                path.name: file_sha256(path)
                for path in asset_files
            },
        },
    )


def _m5_paths(source_path: Path) -> tuple[Path, Path]:
    if not source_path.is_dir():
        raise FileNotFoundError(f"M5 directory not found: {source_path}")
    calendar_path = source_path / "calendar.csv"
    evaluation_path = source_path / "sales_train_evaluation.csv"
    validation_path = source_path / "sales_train_validation.csv"
    sales_path = (
        evaluation_path if evaluation_path.is_file() else validation_path
    )
    for path in (calendar_path, sales_path):
        if not path.is_file():
            raise FileNotFoundError(f"required M5 asset not found: {path}")
    return calendar_path, sales_path


def _m5_read_frames(
    source_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], Path, Path]:
    calendar_path, sales_path = _m5_paths(source_path)
    columns = list(pd.read_csv(sales_path, nrows=0).columns)
    day_columns = sorted(
        (column for column in columns if column.startswith("d_")),
        key=lambda value: int(value.split("_", 1)[1]),
    )
    if len(day_columns) < 216:
        raise ValueError(
            f"M5 sales history is too short for CaFE: {len(day_columns)}"
        )
    identifier_columns = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ]
    dtype = {column: "string" for column in identifier_columns}
    dtype.update({column: np.int16 for column in day_columns})
    sales = pd.read_csv(
        sales_path,
        usecols=[*identifier_columns, *day_columns],
        dtype=dtype,
    )
    calendar = pd.read_csv(calendar_path)
    calendar = calendar.loc[calendar["d"].isin(day_columns)].copy()
    day_order = {day: index for index, day in enumerate(day_columns)}
    calendar["_day_order"] = calendar["d"].map(day_order)
    calendar = (
        calendar.sort_values("_day_order")
        .drop(columns=["_day_order"])
        .reset_index(drop=True)
    )
    if len(calendar) != len(day_columns):
        raise ValueError(
            "M5 calendar does not align one-to-one with sales day columns"
        )
    return calendar, sales, day_columns, calendar_path, sales_path


def _evenly_select(
    values: list[tuple[str, tuple[int, ...]]],
    limit: int | None,
) -> list[tuple[str, tuple[int, ...]]]:
    if limit is None or len(values) <= limit:
        return values
    positions = np.linspace(0, len(values) - 1, num=int(limit), dtype=int)
    return [values[int(position)] for position in positions]


@register_real_data_adapter("m5_csv")
def load_m5_csv(
    source_path: Path,
    record_limit: int | None,
) -> RealDatasetBundle:
    (
        calendar,
        sales,
        day_columns,
        calendar_path,
        sales_path,
    ) = _m5_read_frames(source_path)
    sales_values = sales[day_columns].to_numpy(dtype=np.float32, copy=False)
    descriptors: list[tuple[str, tuple[int, ...]]] = []
    hierarchy_by_group: dict[str, np.ndarray] = {}
    state_by_group: dict[str, str] = {}

    grouped = sales.groupby(["store_id", "cat_id"], sort=True, observed=True)
    for (store_id_value, cat_id_value), group in grouped:
        store_id = str(store_id_value)
        cat_id = str(cat_id_value)
        departments = sorted(str(value) for value in group["dept_id"].unique())
        # The cafe hierarchy mechanism is a parent plus exactly two children.
        # M5 HOBBIES and HOUSEHOLD expose this mapping directly; FOODS has
        # three departments and is therefore not silently projected.
        if len(departments) != 2:
            continue
        group_id = f"{store_id}:{cat_id}"
        group_indexes = group.index.to_numpy(dtype=int)
        child_values = [
            np.sum(
                sales_values[
                    group.loc[
                        group["dept_id"].astype(str) == department
                    ].index.to_numpy(dtype=int)
                ],
                axis=0,
                dtype=np.float64,
            )
            for department in departments
        ]
        hierarchy_by_group[group_id] = np.vstack(
            [child_values[0] + child_values[1], *child_values]
        )
        state_by_group[group_id] = str(group.iloc[0]["state_id"])

        active_indexes = [
            int(index)
            for index in group_indexes
            if float(np.sum(sales_values[int(index)], dtype=np.float64)) > 0.0
        ]
        active_indexes.sort(
            key=lambda index: (
                str(sales.iloc[index]["dept_id"]),
                str(sales.iloc[index]["item_id"]),
                str(sales.iloc[index]["id"]),
            )
        )
        for start in range(0, len(active_indexes) - 4, 5):
            indexes = tuple(active_indexes[start : start + 5])
            item_ids = tuple(str(sales.iloc[index]["item_id"]) for index in indexes)
            if len(set(item_ids)) != 5:
                continue
            descriptors.append((group_id, indexes))

    descriptors.sort(key=lambda value: (value[0], value[1]))
    selected = _evenly_select(descriptors, record_limit)
    records: list[RealSeriesRecord] = []
    for group_id, indexes in selected:
        channel_ids = tuple(str(sales.iloc[index]["id"]) for index in indexes)
        covariates = m5_covariate_matrix(
            calendar,
            state_id=state_by_group[group_id],
        )
        records.append(
            RealSeriesRecord(
                item_id=f"{group_id}:leaf-panel:{channel_ids[0]}",
                values=np.asarray(sales_values[list(indexes)], dtype=float),
                channel_ids=channel_ids,
                covariates=np.asarray(covariates, dtype=float),
                covariate_names=tuple(M5_KNOWN_FUTURE_COVARIATES),
                covariate_kind="known_future",
                hierarchy_values=np.asarray(
                    hierarchy_by_group[group_id],
                    dtype=float,
                ),
                hierarchy_kind="additive_first",
                structural_group_id=group_id,
            )
        )
    if not records:
        raise ValueError("M5 produced no eligible five-leaf structural panels")
    return RealDatasetBundle(
        frequency="D",
        records=tuple(records),
        asset_files=(calendar_path, sales_path),
        adapter_id="m5_csv",
        metadata={
            "sales_asset": sales_path.name,
            "day_count": len(day_columns),
            "eligible_panel_count": len(descriptors),
            "selected_panel_count": len(records),
            "panel_contract": (
                "five distinct active M5 leaves from one store/category"
            ),
            "hierarchy_contract": (
                "official category parent equals the sum of exactly two "
                "official department children; three-child FOODS groups "
                "are excluded without projection"
            ),
            "known_future_covariate_columns": list(
                M5_KNOWN_FUTURE_COVARIATES
            ),
            "known_future_covariate_provenance": M5_COVARIATE_PROVENANCE,
            "excluded_assets": {
                "sell_prices.csv": (
                    "sell price is not guaranteed known at forecast issue time"
                ),
            },
            "asset_sha256": {
                path.name: file_sha256(path)
                for path in (calendar_path, sales_path)
            },
        },
    )
