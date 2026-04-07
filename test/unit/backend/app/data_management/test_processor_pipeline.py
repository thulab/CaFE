from __future__ import annotations

import unittest

from backend.app.data_management.domain import CsvBatchLoadRequest, DataProcessorConfig, DataProcessorType, TrackKind, TrackSpec, TrackTemplateKind, NoiseMode, ExecutionConstraint
from backend.app.data_management.processors import DataProcessorError, build_default_dataset_processor_pipeline
from test.support.helpers import build_sample


class DatasetProcessorPipelineTest(unittest.TestCase):
    def _track_spec(self) -> TrackSpec:
        return TrackSpec(
            track=TrackKind.FORECAST_ACCURACY,
            track_variant_id="univariate_forecast.clean",
            track_template_kind=TrackTemplateKind.UNIVARIATE_FORECAST,
            noise_mode=NoiseMode.CLEAN,
            execution_constraint=ExecutionConstraint.JOINT_MULTIVARIATE,
            name="univariate_forecast.clean",
            description="test",
            fairness_policy="monthly_replay",
            default_context_length=4,
            default_horizon=2,
            suggested_sample_count=1,
            input_channels=["target"],
            target_channels=["target"],
        )

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

        processed = pipeline.process([sample], request, self._track_spec())

        self.assertEqual(processed[0].history, [2.0, 4.0, 6.0, 8.0])
        self.assertEqual(processed[0].target, [10.0, 12.0])
        self.assertEqual(list(processed[0].covariates), ["calendar_signal"])
        self.assertEqual(processed[0].covariates["calendar_signal"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertEqual(processed[0].notes["processors_applied"], ["scale", "covariate_filter"])
        self.assertEqual(processed[0].input_channel_values["target"], [2.0, 4.0, 6.0, 8.0])
        self.assertEqual(processed[0].target_channel_values["target"], [10.0, 12.0])

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
            pipeline.process([build_sample()], request, self._track_spec())

        self.assertIn("min_value <= max_value", str(ctx.exception))
