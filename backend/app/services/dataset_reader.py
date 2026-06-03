from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class DatasetReadResult:
    columns: list[str]
    rows: list[dict[str, str]]
    timestamps: list[datetime]
    target_columns: list[str]
    covariate_columns: list[str]
    values: list[list[float]]
    frequency: str
    encoding: str
    delimiter: str

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def column_matrix(self, columns: list[str]) -> list[list[float]]:
        """Return a row-major matrix for the given selected columns, shape [row_count, len(columns)]."""
        indexes = [self.value_columns.index(column) for column in columns]
        return [[row[index] for index in indexes] for row in self.values]

    @property
    def value_columns(self) -> list[str]:
        return [*self.target_columns, *self.covariate_columns]


class DatasetReader(Protocol):
    def read(
        self,
        path: Path,
        time_column: str,
        target_columns: list[str] | None = None,
        covariate_columns: list[str] | None = None,
        frequency: str | None = None,
    ) -> DatasetReadResult:
        ...


def get_dataset_reader(file_format: str) -> DatasetReader:
    """按 manifest.file_format 选输入读取器：csv（默认）或 tsfile。"""
    if file_format == "tsfile":
        from app.services.tsfile_dataset_reader import TsFileDatasetReader

        return TsFileDatasetReader()
    from app.services.csv_dataset_reader import CsvDatasetReader

    return CsvDatasetReader()
