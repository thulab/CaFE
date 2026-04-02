from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from ...config import future_known_covariates, infer_difficulty, infer_periods_for_track, infer_trend_type
from ...domain import CsvBatchLoadRequest, DatasetLoadRequest, DatasetSourceType, SeriesSample, SeriesTruth, TrackKind
from .base import DataLoaderError, DatasetLoader


class CsvLoaderError(DataLoaderError):
    pass


class CsvDatasetLoader(DatasetLoader):
    source_type = DatasetSourceType.CSV

    def load_samples(self, request: DatasetLoadRequest) -> list[SeriesSample]:
        if not isinstance(request, CsvBatchLoadRequest):
            raise CsvLoaderError(f"csv loader received unsupported request type: {type(request).__name__}")

        csv_path = Path(request.csv_path).expanduser().resolve()
        if not csv_path.exists():
            raise CsvLoaderError(f"csv file not found: {csv_path}")

        grouped_rows = self._read_rows(csv_path=csv_path, request=request)
        samples: list[SeriesSample] = []
        for index, (sample_key, rows) in enumerate(sorted(grouped_rows.items())):
            if request.max_samples is not None and index >= request.max_samples:
                break
            rows.sort(key=lambda item: item["step"])
            expected = request.context_length + request.horizon
            if len(rows) < expected:
                raise CsvLoaderError(f"sample {sample_key} has only {len(rows)} rows, requires at least {expected}")

            trimmed_rows = rows[:expected]
            history = [row["target"] for row in trimmed_rows[: request.context_length]]
            target = [row["target"] for row in trimmed_rows[request.context_length : expected]]
            covariates = {
                covariate: [row["covariates"][covariate] for row in trimmed_rows]
                for covariate in request.covariate_columns
            }
            truth = self._build_truth(series=history + target, track=request.track)
            notes: dict[str, object] = {
                "source_type": request.source_type.value,
                "source_path": str(csv_path),
                "row_count": len(trimmed_rows),
                "future_known_covariates": self._future_known_covariates(request.covariate_columns),
            }
            samples.append(
                SeriesSample(
                    sample_id=str(sample_key),
                    history=history,
                    target=target,
                    covariates=covariates,
                    track_tags=[request.track.value, "csv_loaded"],
                    truth=truth,
                    notes=notes,
                )
            )

        if not samples:
            raise CsvLoaderError(f"csv file {csv_path} did not produce any samples")
        return samples

    def _read_rows(self, csv_path: Path, request: CsvBatchLoadRequest) -> dict[str, list[dict[str, object]]]:
        grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=request.delimiter)
            missing_columns = {
                column
                for column in [request.sample_id_column, request.step_column, request.target_column, *request.covariate_columns]
                if reader.fieldnames is None or column not in reader.fieldnames
            }
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise CsvLoaderError(f"csv file missing required columns: {missing}")

            for row_number, row in enumerate(reader, start=2):
                sample_id = row[request.sample_id_column]
                if sample_id is None or sample_id == "":
                    raise CsvLoaderError(f"row {row_number} missing sample id")
                try:
                    step = int(row[request.step_column])
                    target = float(row[request.target_column])
                    covariates = {
                        column: float(row[column]) if row[column] not in (None, "") else 0.0
                        for column in request.covariate_columns
                    }
                except ValueError as exc:
                    raise CsvLoaderError(f"row {row_number} contains invalid numeric value: {exc}") from exc
                grouped_rows[sample_id].append({"step": step, "target": target, "covariates": covariates})
        return grouped_rows

    def _build_truth(self, series: list[float], track: TrackKind) -> SeriesTruth:
        trend_type = infer_trend_type(series)
        difficulty = infer_difficulty(series)
        noise_level = self._estimate_noise(series)
        periods = self._infer_periods(track=track, length=len(series))
        return SeriesTruth(
            trend_type=trend_type,
            periods=periods,
            dominant_period=periods[-1],
            amplitude_mode="stable",
            phase_shift=False,
            noise_level=noise_level,
            difficulty=difficulty,
        )

    def _infer_periods(self, track: TrackKind, length: int) -> list[int]:
        return infer_periods_for_track(track, length)

    def _estimate_noise(self, series: list[float]) -> float:
        if len(series) < 3:
            return 0.0
        second_diffs = [series[index] - 2 * series[index - 1] + series[index - 2] for index in range(2, len(series))]
        variance = sum(value * value for value in second_diffs) / max(len(second_diffs), 1)
        return round(math.sqrt(max(variance, 0.0)), 4)

    def _future_known_covariates(self, covariate_columns: list[str]) -> list[str]:
        return future_known_covariates(covariate_columns)
