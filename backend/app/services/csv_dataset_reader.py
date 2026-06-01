import csv
import math
from datetime import datetime
from pathlib import Path

from app.core.errors import ApiError
from app.services.dataset_reader import DatasetReadResult
from app.services.time_axis import validate_time_axis


SUPPORTED_DELIMITERS = [",", "\t", ";"]


class CsvDatasetReader:
    def read(
        self,
        path: Path,
        time_column: str,
        target_columns: list[str] | None = None,
        frequency: str | None = None,
    ) -> DatasetReadResult:
        text, encoding = self._read_text(path)
        delimiter = self._detect_delimiter(text)
        rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
        if not rows:
            raise ApiError("csv_missing_header", "CSV must include a header row")

        columns = [name.strip().removeprefix("﻿") for name in rows[0]]
        if self._looks_like_data_row(columns):
            raise ApiError("csv_missing_header", "CSV must include a header row")
        if len(set(columns)) != len(columns):
            raise ApiError("csv_duplicate_columns", "CSV column names must be unique")
        if time_column not in columns:
            raise ApiError("csv_time_column_missing", "time_column was not found", {"time_column": time_column})

        selected = list(target_columns or [])
        if len(selected) != 1:
            raise ApiError(
                "csv_target_columns_invalid",
                "exactly one target column must be selected",
                {"target_columns": selected},
            )
        for target_column in selected:
            if target_column not in columns:
                raise ApiError(
                    "csv_target_column_missing",
                    "target column was not found",
                    {"target_column": target_column},
                )

        dict_rows: list[dict[str, str]] = []
        timestamps: list[datetime] = []
        value_matrix: list[list[float]] = []

        for row_index, row in enumerate(rows[1:], start=2):
            if not row or all(not value.strip() for value in row):
                continue
            values = {column: row[position].strip() if position < len(row) else "" for position, column in enumerate(columns)}
            parsed_time = self._parse_time(values.get(time_column, ""), row_index)
            timestamps.append(parsed_time)

            row_values: list[float] = []
            for value_column in selected:
                raw_value = values.get(value_column, "")
                if raw_value == "":
                    raise ApiError("csv_value_missing", "value is missing", {"row_index": row_index, "column": value_column})
                try:
                    parsed_value = float(raw_value)
                except ValueError as exc:
                    raise ApiError("csv_value_not_float", "value must be numeric", {"row_index": row_index, "column": value_column}) from exc
                if not math.isfinite(parsed_value):
                    raise ApiError("csv_value_not_finite", "value must be finite", {"row_index": row_index, "column": value_column})
                row_values.append(parsed_value)

            dict_rows.append(values)
            value_matrix.append(row_values)

        inferred_frequency = validate_time_axis(timestamps, frequency)

        return DatasetReadResult(
            columns=columns,
            rows=dict_rows,
            timestamps=timestamps,
            target_columns=selected,
            values=value_matrix,
            frequency=inferred_frequency,
            encoding=encoding,
            delimiter=delimiter,
        )

    def _read_text(self, path: Path) -> tuple[str, str]:
        raw = path.read_bytes()
        encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            raise ApiError("csv_encoding_unsupported", "CSV encoding must be UTF-8") from exc

    def _detect_delimiter(self, text: str) -> str:
        first_line = next((line for line in text.splitlines() if line.strip()), "")
        if not first_line:
            raise ApiError("csv_missing_header", "CSV must include a header row")
        return max(SUPPORTED_DELIMITERS, key=first_line.count)

    def _parse_time(self, value: str, row_index: int) -> datetime:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ApiError("csv_time_parse_failed", "time_column contains an unsupported datetime", {"row_index": row_index}) from exc

    def _looks_like_data_row(self, columns: list[str]) -> bool:
        if len(columns) < 2:
            return False
        try:
            datetime.fromisoformat(columns[0])
            float(columns[1])
            return True
        except ValueError:
            return False
