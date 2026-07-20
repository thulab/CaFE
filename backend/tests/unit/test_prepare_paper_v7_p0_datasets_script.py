from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "prepare_paper_v7_p0_datasets.py"
)


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location(
        "prepare_paper_v7_p0_datasets",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def swiss_power_fixture(module, index: pd.DatetimeIndex) -> pd.DataFrame:
    meter_columns = [f"meter_{index}" for index in range(24)]
    leaves = np.column_stack(
        [
            np.arange(len(index), dtype=float) + meter_index
            for meter_index in range(24)
        ]
    )
    frame = pd.DataFrame(leaves, index=index, columns=meter_columns)
    frame["S11"] = frame[meter_columns[0:6]].sum(axis=1)
    frame["S12"] = frame[meter_columns[6:12]].sum(axis=1)
    frame["S21"] = frame[meter_columns[12:18]].sum(axis=1)
    frame["S22"] = frame[meter_columns[18:24]].sum(axis=1)
    frame["S1"] = frame["S11"] + frame["S12"]
    frame["S2"] = frame["S21"] + frame["S22"]
    frame["all"] = frame["S1"] + frame["S2"]
    return frame[
        [
            *meter_columns,
            *module.SWISS_AGGREGATE_COLUMNS,
        ]
    ]


def test_swiss_aggregation_keeps_only_complete_trailing_bins(module) -> None:
    index = pd.date_range(
        "2020-01-01 00:10",
        periods=8,
        freq="10min",
        tz="UTC",
    )
    frame = swiss_power_fixture(module, index)

    aggregated, audit = module.complete_trailing_30_minute_means(frame)

    assert list(aggregated.index) == [
        pd.Timestamp("2020-01-01 00:00", tz="UTC"),
        pd.Timestamp("2020-01-01 00:30", tz="UTC"),
    ]
    assert aggregated["meter_0"].tolist() == pytest.approx([1.0, 4.0])
    assert audit == {
        "source_rows": 8,
        "complete_bin_rows": 2,
        "discarded_source_rows": 2,
        "discarded_bins": 1,
    }
    assert module.swiss_hierarchy_audit(aggregated)[
        "max_absolute_error"
    ] == pytest.approx(0.0)


def test_swiss_nwp_alignment_is_latest_asof_and_numeric(module) -> None:
    variables = module.SWISS_NWP_CANONICAL_COLUMNS
    nwp_index = pd.DatetimeIndex(
        [
            "2019-12-31 12:00",
            "2020-01-01 00:00",
            "2020-01-01 00:20",
        ],
        tz="UTC",
    )
    nwp = pd.DataFrame(
        {
            variable: [
                np.full(24, variable_index * 10 + row_index, dtype=float)
                for row_index in range(3)
            ]
            for variable_index, variable in enumerate(variables)
        },
        index=nwp_index,
    )
    targets = pd.DatetimeIndex(
        ["2020-01-01 00:00", "2020-01-01 00:30"],
        tz="UTC",
    )

    available, asof, valid, cube = module.align_latest_swiss_nwp(targets, nwp)

    assert available.tolist() == [True, True]
    assert np.array_equal(
        asof,
        np.asarray(
            [
                "2020-01-01T00:00:00.000000000",
                "2020-01-01T00:20:00.000000000",
            ],
            dtype="datetime64[ns]",
        ),
    )
    assert cube.shape == (2, 6, 24)
    assert cube[0, 0, 0] == pytest.approx(1.0)
    assert cube[1, 5, -1] == pytest.approx(52.0)
    assert not cube.dtype.hasobject
    assert valid.shape == (2, 24)
    assert np.array_equal(valid[:, 0], asof)
    assert valid[1, 1] == np.datetime64("2020-01-01T01:20:00")
    assert np.all(np.diff(valid, axis=1) == np.timedelta64(1, "h"))


def gefcom_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hours = [f"h{hour}" for hour in range(1, 25)]
    history_rows = []
    for zone in range(1, 21):
        for day in range(1, 4):
            values = [float(zone * 100 + hour + day) for hour in range(1, 25)]
            if day == 2:
                values = [float("nan")] * 24
            if day == 3:
                values[6:] = [float("nan")] * 18
            history_rows.append(
                {
                    "zone_id": zone,
                    "year": 2004,
                    "month": 1,
                    "day": day,
                    **dict(zip(hours, values, strict=True)),
                }
            )

    solution_rows = []
    leaf_values: dict[int, list[float]] = {}
    for zone in range(1, 21):
        leaf_values[zone] = [
            float(zone * 100 + hour + 2) for hour in range(1, 25)
        ]
        solution_rows.append(
            {
                "zone_id": zone,
                "year": 2004,
                "month": 1,
                "day": 2,
                **dict(zip(hours, leaf_values[zone], strict=True)),
            }
        )
    total_values = [
        sum(leaf_values[zone][hour] for zone in range(1, 21))
        for hour in range(24)
    ]
    solution_rows.append(
        {
            "zone_id": 21,
            "year": 2004,
            "month": 1,
            "day": 2,
            **dict(zip(hours, total_values, strict=True)),
        }
    )
    holidays = pd.DataFrame(
        {
            "Unnamed: 0": ["New Year's Day"],
            "2004": ["Thursday, January 1"],
        }
    )
    return (
        pd.DataFrame(history_rows),
        pd.DataFrame(solution_rows),
        holidays,
    )


def test_gefcom_excludes_missing_targets_and_freezes_hourly_segments(module) -> None:
    history, solution, holidays = gefcom_fixture()

    arrays, audit = module.prepare_gefcom2012_arrays(
        history,
        solution,
        holidays,
    )

    assert arrays["timestamps"].shape == (30,)
    assert arrays["timestamps"][0] == np.datetime64("2004-01-01T01:00:00")
    assert arrays["timestamps"][-1] == np.datetime64("2004-01-03T06:00:00")
    assert arrays["zones"].shape == (30, 20)
    assert arrays["canonical_hierarchy"].shape == (30, 3)
    assert arrays["calendar_covariates"].shape == (30, 6)
    assert arrays["segment_ids"].dtype.kind in "iu"
    assert arrays["segment_ids"].tolist() == [0] * 24 + [1] * 6
    assert arrays["total"] == pytest.approx(arrays["zones"].sum(axis=1))
    assert arrays["canonical_hierarchy"][:, 0] == pytest.approx(
        arrays["canonical_hierarchy"][:, 1:].sum(axis=1)
    )
    assert audit["retained_history_hours"] == 30
    assert audit["excluded_target_missing_hours"] == 42
    assert audit["excluded_target_missing_cells"] == 840
    assert audit["excluded_official_hidden_hours"] == 24
    assert audit["excluded_official_hidden_cells"] == 480
    assert audit["excluded_internal_hidden_hours"] == 24
    assert audit["excluded_post_tail_evaluation_hours"] == 0
    assert audit["excluded_incomplete_tail_hours"] == 18
    assert audit["excluded_incomplete_tail_cells"] == 360
    assert audit["segment_count"] == 2
    assert audit["segment_lengths"] == [24, 6]
    assert audit["solution_zone21_max_absolute_error"] == pytest.approx(0.0)


def test_written_npz_is_loadable_without_pickle(module, tmp_path) -> None:
    path = tmp_path / "numeric.npz"
    arrays = {
        "timestamps": np.asarray(
            ["2020-01-01T00:00:00"],
            dtype="datetime64[ns]",
        ),
        "values": np.asarray([[1.0, 2.0]], dtype=np.float64),
    }

    module.write_npz_atomic(path, arrays)

    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == {"timestamps", "values"}
        assert loaded["values"].tolist() == [[1.0, 2.0]]


def test_npz_writer_rejects_object_arrays(module, tmp_path) -> None:
    with pytest.raises(ValueError, match="object dtype"):
        module.write_npz_atomic(
            tmp_path / "bad.npz",
            {"bad": np.asarray([{"not": "numeric"}], dtype=object)},
        )
