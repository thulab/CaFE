from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from synthetic_feature_profile import (
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
            raise ValueError(f"duplicate Paper v8 real-data adapter {adapter_id!r}")
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
            f"unknown Paper v8 real-data adapter {adapter_id!r}; "
            f"registered={sorted(_LOADERS)}"
        ) from error
    bundle = loader(source_path.resolve(), record_limit)
    if not bundle.records:
        raise ValueError(
            f"Paper v8 real-data adapter {adapter_id!r} returned no records"
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
            f"M5 sales history is too short for Paper v8: {len(day_columns)}"
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
        # The v8 hierarchy mechanism is a parent plus exactly two children.
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
