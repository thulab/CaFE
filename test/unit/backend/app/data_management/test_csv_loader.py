from __future__ import annotations

import unittest

from backend.app.data_management.data_loader.csv_loader import CsvDatasetLoader, CsvLoaderError
from backend.app.data_management.domain import CsvBatchLoadRequest, TrackKind
from test.support.helpers import temporary_runtime_dir, write_demo_csv


class CsvDatasetLoaderTest(unittest.TestCase):
    def test_load_samples_reads_history_target_and_future_known_covariates(self) -> None:
        with temporary_runtime_dir(prefix="csv-loader-") as runtime_root:
            csv_path = write_demo_csv(runtime_root)
            loader = CsvDatasetLoader()

            request = CsvBatchLoadRequest(
                csv_path=str(csv_path),
                track=TrackKind.FORECAST_ACCURACY,
                context_length=8,
                horizon=4,
                covariate_columns=["calendar_signal", "noise_signal"],
                max_samples=1,
            )

            samples = loader.load_samples(request)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.sample_id, "series_a")
        self.assertEqual(sample.history, [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7])
        self.assertEqual(sample.target, [10.8, 10.9, 11.0, 11.1])
        self.assertEqual(sample.notes["future_known_covariates"], ["calendar_signal"])
        self.assertEqual(sample.track_tags, ["forecast_accuracy", "csv_loaded"])

    def test_load_samples_rejects_missing_columns(self) -> None:
        with temporary_runtime_dir(prefix="csv-loader-") as runtime_root:
            csv_path = runtime_root / "broken.csv"
            csv_path.write_text("sample_id,step,target\ns1,0,1.0\n", encoding="utf-8")
            loader = CsvDatasetLoader()

            request = CsvBatchLoadRequest(
                csv_path=str(csv_path),
                context_length=1,
                horizon=1,
                covariate_columns=["calendar_signal"],
            )

            with self.assertRaises(CsvLoaderError) as ctx:
                loader.load_samples(request)

        self.assertIn("missing required columns", str(ctx.exception))
