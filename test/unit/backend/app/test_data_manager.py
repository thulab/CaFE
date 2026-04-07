from __future__ import annotations

import unittest

from backend.app.data_management.domain import BatchGenerationRequest, CsvBatchLoadRequest, DataProcessorConfig, DataProcessorType, TrackKind
from backend.app.data_management.manager import DataManager
from backend.app.errors import BenchmarkError
from test.support.helpers import temporary_runtime_dir, write_demo_csv


class DataManagerTest(unittest.TestCase):
    def test_generate_batch_persists_and_lists_batches(self) -> None:
        with temporary_runtime_dir(prefix="data-manager-") as runtime_root:
            manager = DataManager(runtime_root)
            batch = manager.generate_batch(
                BatchGenerationRequest(
                    track=TrackKind.FORECAST_ACCURACY,
                    sample_count=2,
                    context_length=24,
                    horizon=8,
                    seed=5,
                )
            )
            listed = manager.list_batches()
            loaded = manager.get_batch(batch.batch_id)

        self.assertEqual(batch.sample_count, 2)
        self.assertTrue(batch.validation.passed)
        self.assertEqual(len(listed), 1)
        self.assertEqual(loaded.batch_id, batch.batch_id)

    def test_load_batch_applies_processors_and_persists(self) -> None:
        with temporary_runtime_dir(prefix="data-manager-") as runtime_root:
            csv_path = write_demo_csv(runtime_root)
            manager = DataManager(runtime_root)
            batch = manager.load_batch(
                CsvBatchLoadRequest(
                    csv_path=str(csv_path),
                    track=TrackKind.FORECAST_ACCURACY,
                    context_length=8,
                    horizon=4,
                    max_samples=1,
                    processors=[
                        DataProcessorConfig(
                            processor_type=DataProcessorType.SCALE,
                            params={"factor": 10, "include_covariates": False},
                        )
                    ],
                )
            )

        self.assertEqual(batch.sample_count, 1)
        self.assertTrue(batch.validation.passed)
        self.assertEqual(batch.samples[0].history[:2], [100.0, 101.0])
        self.assertEqual(batch.samples[0].notes["processors_applied"], ["scale"])

    def test_load_batch_rejects_non_positive_max_samples(self) -> None:
        with temporary_runtime_dir(prefix="data-manager-") as runtime_root:
            csv_path = write_demo_csv(runtime_root)
            manager = DataManager(runtime_root)

            with self.assertRaises(BenchmarkError) as ctx:
                manager.load_batch(
                    CsvBatchLoadRequest(
                        csv_path=str(csv_path),
                        context_length=8,
                        horizon=4,
                        max_samples=0,
                    )
                )

        self.assertIn("max_samples must be positive", str(ctx.exception))
