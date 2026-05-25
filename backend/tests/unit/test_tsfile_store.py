"""TDD tests for TsFileStore and TsFileSlicer (plan Task B1.x).

Covers write → slice roundtrip via the tsfile table model (§2 spec).
"""
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

tsfile = pytest.importorskip("tsfile")

from app.services.tsfile_store import TsFileStore, TsFileSlicer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 8
BASE_DT = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

TIMESTAMPS = [
    datetime(2024, 1, 1, i, 0, 0, tzinfo=timezone.utc) for i in range(N)
]
VALUE_COLUMNS = ["target", "extra"]
# row-major: shape [N, 2]
VALUES = [[float(i * 10), float(i * 10 + 1)] for i in range(N)]
DATASET_ID = "ds42"
SHARD_ID = "shard-001"


@pytest.fixture()
def store_and_path(tmp_path: Path):
    store = TsFileStore(tmp_path / "tsfiles")
    path = store.write(SHARD_ID, DATASET_ID, TIMESTAMPS, VALUE_COLUMNS, VALUES)
    return store, path


# ---------------------------------------------------------------------------
# 1. write() returns correct path and the file exists
# ---------------------------------------------------------------------------


def test_write_returns_path_and_file_exists(tmp_path: Path):
    store = TsFileStore(tmp_path / "tsfiles")
    path = store.write(SHARD_ID, DATASET_ID, TIMESTAMPS, VALUE_COLUMNS, VALUES)

    assert path == store.tsfile_uri(SHARD_ID)
    assert path.exists()
    assert path.suffix == ".tsfile"


# ---------------------------------------------------------------------------
# 2. slice() multi-column returns correct row-major floats
# ---------------------------------------------------------------------------


def test_slice_multi_column(store_and_path):
    _, path = store_and_path
    slicer = TsFileSlicer(path)
    rows = slicer.slice(DATASET_ID, VALUE_COLUMNS, 2, 5)
    slicer.close()

    assert len(rows) == 3  # rows [2, 5)
    expected = VALUES[2:5]
    for got_row, exp_row in zip(rows, expected):
        assert len(got_row) == len(exp_row)
        for g, e in zip(got_row, exp_row):
            assert math.isclose(g, e, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 3. slice_timestamps() returns ms-epoch ints for [row_start, row_end)
# ---------------------------------------------------------------------------


def test_slice_timestamps(store_and_path):
    _, path = store_and_path
    slicer = TsFileSlicer(path)
    ts = slicer.slice_timestamps(DATASET_ID, 2, 5)
    slicer.close()

    assert len(ts) == 3
    expected_ms = [int(dt.timestamp() * 1000) for dt in TIMESTAMPS[2:5]]
    assert ts == expected_ms


# ---------------------------------------------------------------------------
# 4. Single-column slice works
# ---------------------------------------------------------------------------


def test_slice_single_column(store_and_path):
    _, path = store_and_path
    slicer = TsFileSlicer(path)
    rows = slicer.slice(DATASET_ID, ["target"], 0, N)
    slicer.close()

    assert len(rows) == N
    for i, row in enumerate(rows):
        assert len(row) == 1
        assert math.isclose(row[0], VALUES[i][0], rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 5. Context-manager protocol works
# ---------------------------------------------------------------------------


def test_slicer_context_manager(store_and_path):
    _, path = store_and_path
    with TsFileSlicer(path) as slicer:
        rows = slicer.slice(DATASET_ID, VALUE_COLUMNS, 0, 2)

    assert len(rows) == 2
