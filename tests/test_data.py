from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest

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
