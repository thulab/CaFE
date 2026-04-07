from __future__ import annotations

import unittest

from backend.app.data_management.domain import CsvBatchLoadRequest, DataProcessorConfig, DataProcessorType
from backend.app.data_management.processors import DataProcessorError, build_default_dataset_processor_pipeline
from test.support.helpers import build_sample


class DatasetProcessorPipelineTest(unittest.TestCase):
    def test_process_applies_scale_then_covariate_filter(self) -> None:
        pipeline = build_default_dataset_processor_pipeline()
        request = CsvBatchLoadRequest(
            csv_path="unused.csv",
            context_length=4,
            horizon=2,
            processors=[
                DataProcessorConfig(
                    processor_type=DataProcessorType.SCALE,
                    params={"factor": 2, "include_covariates": False},
                ),
                DataProcessorConfig(
                    processor_type=DataProcessorType.COVARIATE_FILTER,
                    params={"keep": ["calendar_signal"]},
                ),
            ],
        )
        sample = build_sample(
            covariates={
                "calendar_signal": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "noise_signal": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            }
        )

        processed = pipeline.process([sample], request)

        self.assertEqual(processed[0].history, [2.0, 4.0, 6.0, 8.0])
        self.assertEqual(processed[0].target, [10.0, 12.0])
        self.assertEqual(list(processed[0].covariates), ["calendar_signal"])
        self.assertEqual(processed[0].covariates["calendar_signal"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertEqual(processed[0].notes["processors_applied"], ["scale", "covariate_filter"])

    def test_process_rejects_invalid_clip_range(self) -> None:
        pipeline = build_default_dataset_processor_pipeline()
        request = CsvBatchLoadRequest(
            csv_path="unused.csv",
            context_length=4,
            horizon=2,
            processors=[
                DataProcessorConfig(
                    processor_type=DataProcessorType.CLIP,
                    params={"min_value": 5, "max_value": 1},
                )
            ],
        )

        with self.assertRaises(DataProcessorError) as ctx:
            pipeline.process([build_sample()], request)

        self.assertIn("min_value <= max_value", str(ctx.exception))
