from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pa_parquet
import pytest

from cafe import protocol
from cafe.data import fev_bench
from cafe.data import real


def load_real_data_module():
    return real


BRAND_COUNTS = {
    "B1": 42,
    "B2": 45,
    "B3": 21,
    "B4": 10,
}


def hierarchical_sales_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for brand, count in BRAND_COUNTS.items():
        for index in range(1, count + 1):
            rows.append(
                {
                    "item_id": f"QTY_{brand}_{index}",
                    "start": datetime(2014, 1, 2),
                    "target": [
                        float(index),
                        float(index + 1),
                        float(index + 2),
                    ],
                    "freq": "D",
                }
            )
    return rows


def write_arrow(asset_path: Path, rows: list[dict[str, object]]) -> Path:
    asset_path.mkdir(parents=True)
    arrow_path = asset_path / "data-00000-of-00001.arrow"
    table = pa.table(
        {
            "item_id": pa.array(
                [row["item_id"] for row in rows],
                type=pa.string(),
            ),
            "start": pa.array(
                [row["start"] for row in rows],
                type=pa.timestamp("s"),
            ),
            "target": pa.array(
                [row["target"] for row in rows],
                type=pa.list_(pa.float32()),
            ),
            "freq": pa.array(
                [row["freq"] for row in rows],
                type=pa.string(),
            ),
        }
    )
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with pa_ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    return arrow_path


def write_fev_parquet(
    asset_path: Path,
    *,
    timestamps: list[datetime] | None = None,
) -> tuple[Path, fev_bench.FevBenchConfig]:
    asset_path.mkdir(parents=True)
    resolved_timestamps = timestamps or [
        datetime(2024, 1, 1) + timedelta(hours=index) for index in range(4)
    ]
    parquet_path = asset_path / "train-00000-of-00001.parquet"
    table = pa.table(
        {
            "id": pa.array(["item-1"]),
            "timestamp": pa.array(
                [resolved_timestamps],
                type=pa.list_(pa.timestamp("ms")),
            ),
            "target_a": pa.array([[1.0, 2.0, 3.0, 4.0]]),
            "target_b": pa.array([[4.0, 3.0, 2.0, 1.0]]),
            "known": pa.array([[0.0, 1.0, 0.0, 1.0]]),
            "known_cat": pa.array([["0", "a", "0", "a"]]),
            "past": pa.array([[5.0, 6.0, 7.0, 8.0]]),
            "static": pa.array(["group-a"]),
        }
    )
    pa_parquet.write_table(table, parquet_path)
    digest = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    config = fev_bench.FevBenchConfig(
        config_id=asset_path.name,
        source_path=f"test/{parquet_path.name}",
        frequency="h",
        target_columns=("target_a", "target_b"),
        known_dynamic_columns=("known", "known_cat"),
        past_dynamic_columns=("past",),
        static_columns=("static",),
        categorical_dynamic_levels=(("known_cat", ("0", "a")),),
        sha256=digest,
        size_bytes=parquet_path.stat().st_size,
    )
    return parquet_path, config


def test_fev_parquet_adapter_preserves_native_targets_and_known_covariates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    asset_path = tmp_path / "test_fev"
    parquet_path, config = write_fev_parquet(asset_path)
    monkeypatch.setitem(real.FEV_BENCH_CONFIGS, config.config_id, config)

    bundle = real.load_real_dataset("fev_parquet", asset_path)

    assert bundle.frequency == "h"
    assert bundle.asset_files == (parquet_path,)
    assert bundle.adapter_id == "fev_parquet"
    assert len(bundle.records) == 1
    record = bundle.records[0]
    assert record.channel_ids == ("target_a", "target_b")
    np.testing.assert_array_equal(
        record.values,
        np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]),
    )
    assert record.covariate_names == (
        "known",
        "known_cat=0",
        "known_cat=a",
    )
    assert record.covariate_kind == "known_future"
    np.testing.assert_array_equal(
        record.covariates,
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ]
        ),
    )
    assert bundle.metadata["past_dynamic_columns"] == ["past"]
    assert bundle.metadata["static_columns"] == ["static"]
    assert bundle.metadata["dataset_revision"] == (
        fev_bench.FEV_DATASET_REVISION
    )


def test_fev_parquet_adapter_rejects_irregular_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    asset_path = tmp_path / "test_irregular_fev"
    timestamps = [
        datetime(2024, 1, 1),
        datetime(2024, 1, 1, 1),
        datetime(2024, 1, 1, 3),
        datetime(2024, 1, 1, 4),
    ]
    _parquet_path, config = write_fev_parquet(
        asset_path,
        timestamps=timestamps,
    )
    monkeypatch.setitem(real.FEV_BENCH_CONFIGS, config.config_id, config)

    with pytest.raises(ValueError, match="not regular"):
        real.load_real_dataset("fev_parquet", asset_path)


def test_fev_parquet_adapter_accepts_anchored_calendar_frequency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    asset_path = tmp_path / "test_weekly_fev"
    timestamps = [
        datetime(2024, 1, 5) + timedelta(weeks=index) for index in range(4)
    ]
    _parquet_path, config = write_fev_parquet(
        asset_path,
        timestamps=timestamps,
    )
    config = replace(config, frequency="W-FRI")
    monkeypatch.setitem(real.FEV_BENCH_CONFIGS, config.config_id, config)

    bundle = real.load_real_dataset("fev_parquet", asset_path)

    assert bundle.frequency == "W-FRI"


def test_fev_categorical_covariates_preserve_missingness_as_nan():
    table = pa.table(
        {
            "known_cat": pa.array(
                [["none", None, "event"]],
                type=pa.list_(pa.string()),
            )
        }
    )
    config = fev_bench.FevBenchConfig(
        config_id="category-test",
        source_path="unused.parquet",
        frequency="D",
        target_columns=("target",),
        known_dynamic_columns=("known_cat",),
        categorical_dynamic_levels=(
            ("known_cat", ("event", "none")),
        ),
    )

    values, names = real._fev_known_covariates(
        table,
        row_index=0,
        config=config,
        expected_length=3,
    )

    assert names == ("known_cat=event", "known_cat=none")
    assert values is not None
    np.testing.assert_array_equal(
        values,
        np.array([[0.0, 1.0], [np.nan, np.nan], [1.0, 0.0]]),
    )


def test_fev_pilot_registry_uses_local_parquet_adapter():
    dataset_ids = {
        "fev_ett_1h",
        "fev_jena_weather_1h",
        "fev_boomlet_1282",
        "fev_uci_air_quality_1h",
        "fev_solar_with_weather_1h",
        "fev_proenfo_gfc14",
        "fev_rohlik_orders_1d",
        "fev_rossmann_1d",
        "fev_hospital_admissions_1d",
    }
    for dataset_id in dataset_ids:
        dataset = protocol.resolve_dataset(dataset_id)
        assert dataset.real_data_adapter == "fev_parquet"
        assert dataset.asset_name in fev_bench.FEV_BENCH_CONFIGS


def write_promotion_csv(
    asset_path: Path,
    *,
    dates: list[datetime] | None = None,
) -> Path:
    resolved_dates = dates or [
        datetime(2014, 1, 2),
        datetime(2014, 1, 3),
        datetime(2014, 1, 4),
    ]
    columns: dict[str, list[object]] = {"DATE": resolved_dates}
    first_date = min(resolved_dates)
    for brand, count in BRAND_COUNTS.items():
        for index in range(1, count + 1):
            columns[f"QTY_{brand}_{index}"] = [
                float(index + (date - first_date).days)
                for date in resolved_dates
            ]
            columns[f"PROMO_{brand}_{index}"] = [
                float((index + date_index) % 2)
                for date_index in range(len(resolved_dates))
            ]
    promotion_path = asset_path / "hierarchical_sales_data.csv"
    pd.DataFrame(columns).to_csv(promotion_path, index=False)
    return promotion_path


def test_hierarchical_sales_adapter_validates_then_pairs_in_natural_order(
    tmp_path: Path,
):
    module = load_real_data_module()
    asset_path = tmp_path / "hierarchical_sales" / "D"
    arrow_path = write_arrow(
        asset_path,
        list(reversed(hierarchical_sales_rows())),
    )

    bundle = module.load_real_dataset(
        "gift_hierarchical_sales",
        asset_path,
        record_limit=2,
    )

    assert bundle.frequency == "D"
    assert bundle.adapter_id == "gift_hierarchical_sales"
    assert bundle.asset_files == (arrow_path,)
    assert len(bundle.records) == 2
    first = bundle.records[0]
    assert first.item_id == "hierarchical_sales:B1:1-2"
    assert first.channel_ids == ("QTY_B1_1", "QTY_B1_2")
    assert first.structural_group_id == "hierarchical_sales:B1"
    assert first.hierarchy_kind == "children_only_additive"
    assert first.covariates is None
    assert first.covariate_names == ()
    assert first.covariate_kind is None
    np.testing.assert_array_equal(
        first.values,
        np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
    )
    np.testing.assert_array_equal(first.hierarchy_values, first.values)

    metadata = bundle.metadata
    assert metadata["validated_leaf_count"] == 118
    assert metadata["validated_brand_counts"] == BRAND_COUNTS
    assert metadata["eligible_pair_count"] == 58
    assert metadata["selected_pair_count"] == 2
    assert metadata["unpaired_child_ids"] == ["QTY_B2_45", "QTY_B3_21"]
    assert metadata["hierarchy_provenance"]["grouping_key"]
    assert metadata["promotion_covariates"]["available"] is False
    checksum = metadata["asset_sha256"][arrow_path.name]
    assert len(checksum) == 64


def test_hierarchical_sales_adapter_attaches_aligned_promotion_covariates(
    tmp_path: Path,
):
    module = load_real_data_module()
    asset_path = tmp_path / "hierarchical_sales" / "D"
    rows = hierarchical_sales_rows()
    for row in rows:
        row["target"][1] = float("nan")
    arrow_path = write_arrow(asset_path, rows)
    promotion_path = write_promotion_csv(
        asset_path,
        dates=[datetime(2014, 1, 2), datetime(2014, 1, 4)],
    )

    bundle = module.load_real_dataset(
        "gift_hierarchical_sales",
        asset_path,
        record_limit=1,
    )

    assert bundle.asset_files == (arrow_path, promotion_path)
    record = bundle.records[0]
    assert record.covariate_names == ("PROMO_B1_1", "PROMO_B1_2")
    assert record.covariate_kind == "known_future"
    np.testing.assert_array_equal(
        record.covariates,
        np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
    )
    promotion_metadata = bundle.metadata["promotion_covariates"]
    assert promotion_metadata["available"] is True
    assert promotion_metadata["missing_date_count"] == 1
    assert promotion_metadata["missing_value_count"] == 118
    assert "filled with 0" in promotion_metadata["missing_date_fill_policy"]
    assert bundle.metadata["known_future_covariate_provenance"]
    assert len(
        bundle.metadata["asset_sha256"][promotion_path.name]
    ) == 64


def test_hierarchical_sales_adapter_validates_all_leaves_before_limit(
    tmp_path: Path,
):
    module = load_real_data_module()
    asset_path = tmp_path / "hierarchical_sales" / "D"
    rows = hierarchical_sales_rows()
    rows[-1]["item_id"] = "QTY_B4_11"
    write_arrow(asset_path, rows)

    with pytest.raises(ValueError, match="118-leaf contract"):
        module.load_real_dataset(
            "gift_hierarchical_sales",
            asset_path,
            record_limit=1,
        )


def test_hierarchical_sales_adapter_rejects_incomplete_promotion_mapping(
    tmp_path: Path,
):
    module = load_real_data_module()
    asset_path = tmp_path / "hierarchical_sales" / "D"
    write_arrow(asset_path, hierarchical_sales_rows())
    promotion_path = write_promotion_csv(asset_path)
    frame = pd.read_csv(promotion_path).drop(columns=["PROMO_B4_10"])
    frame.to_csv(promotion_path, index=False)

    with pytest.raises(ValueError, match="one-to-one"):
        module.load_real_dataset(
            "gift_hierarchical_sales",
            asset_path,
            record_limit=1,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda rows: rows[-1].__setitem__(
                "start",
                datetime(2014, 1, 3),
            ),
            "common start",
        ),
        (
            lambda rows: rows[-1].__setitem__(
                "target",
                [1.0, 2.0],
            ),
            "common target length",
        ),
        (
            lambda rows: rows[-1].__setitem__("freq", "H"),
            "daily frequency",
        ),
    ],
)
def test_hierarchical_sales_adapter_rejects_misaligned_leaves(
    tmp_path: Path,
    mutation,
    message: str,
):
    module = load_real_data_module()
    asset_path = tmp_path / "hierarchical_sales" / "D"
    rows = hierarchical_sales_rows()
    mutation(rows)
    write_arrow(asset_path, rows)

    with pytest.raises(ValueError, match=message):
        module.load_real_dataset("gift_hierarchical_sales", asset_path)
