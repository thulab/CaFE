"""TsFile-backed storage for time-series shards (table model).

Two public classes:
- TsFileStore  — writes a per-shard .tsfile from raw timestamps + value matrix.
- TsFileSlicer — opens a .tsfile and slices it by row number range.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import tsfile

_TABLE = "tsbench"


class TsFileStore:
    """Manages a directory of per-shard .tsfile files."""

    def __init__(self, tsfiles_dir: Path) -> None:
        self.tsfiles_dir = Path(tsfiles_dir)
        self.tsfiles_dir.mkdir(parents=True, exist_ok=True)

    def tsfile_uri(self, shard_id: str) -> Path:
        return self.tsfiles_dir / f"{shard_id}.tsfile"

    def write(
        self,
        shard_id: str,
        dataset_id: str,
        timestamps: list[datetime],
        value_columns: list[str],
        values: list[list[float]],
    ) -> Path:
        """Write a shard to disk and return its path.

        Parameters
        ----------
        shard_id:       unique key for the shard file name.
        dataset_id:     tag value used as the tsfile device/table identifier.
        timestamps:     N UTC datetimes, one per row.
        value_columns:  column names, length C.
        values:         row-major float matrix, shape [N, C].
        """
        n = len(timestamps)
        time_ms = [int(dt.timestamp() * 1000) for dt in timestamps]
        data: dict[str, object] = {"time": time_ms}
        for col_idx, col in enumerate(value_columns):
            data[col] = [values[row][col_idx] for row in range(n)]
        data["dataset_id"] = [dataset_id] * n

        df = pd.DataFrame(data)
        path = self.tsfile_uri(shard_id)
        tsfile.dataframe_to_tsfile(
            df,
            str(path),
            table_name=_TABLE,
            time_column="time",
            tag_column=["dataset_id"],
        )
        return path


class TsFileSlicer:
    """Reads row slices from a .tsfile written by TsFileStore."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._df: tsfile.TsFileDataFrame | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self) -> tsfile.TsFileDataFrame:
        if self._df is None:
            self._df = tsfile.TsFileDataFrame(str(self._path))
        return self._df

    def _series_key(self, dataset_id: str, column: str) -> str:
        return f"{_TABLE}.{dataset_id}.{column}"

    def _first_column(self, dataset_id: str) -> str:
        """Return the first column name available for the given dataset."""
        prefix = f"{_TABLE}.{dataset_id}"
        series_list: list[str] = self._open().list_timeseries(prefix)
        # series_list entries look like "tsbench.ds42.target"
        return series_list[0].split(".")[-1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def slice(
        self,
        dataset_id: str,
        columns: list[str],
        row_start: int,
        row_end: int,
    ) -> list[list[float]]:
        """Return rows [row_start, row_end) as a row-major list[list[float]]."""
        df = self._open()
        arrays = [
            df[self._series_key(dataset_id, col)][row_start:row_end]
            for col in columns
        ]
        # arrays is [col0_ndarray, col1_ndarray, …]; transpose to row-major
        n_rows = row_end - row_start
        return [
            [float(arrays[c][r]) for c in range(len(columns))]
            for r in range(n_rows)
        ]

    def slice_timestamps(
        self,
        dataset_id: str,
        row_start: int,
        row_end: int,
    ) -> list[int]:
        """Return ms-epoch ints for rows [row_start, row_end).

        All columns in the same device share identical timestamps,
        so we pick the first available column internally.
        """
        df = self._open()
        col = self._first_column(dataset_id)
        ts_array = df[self._series_key(dataset_id, col)].timestamps[row_start:row_end]
        return [int(t) for t in ts_array]

    def close(self) -> None:
        if self._df is not None:
            self._df.__exit__(None, None, None)
            self._df = None

    def __enter__(self) -> "TsFileSlicer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
