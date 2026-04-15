from __future__ import annotations

import math
import unittest

from backend.app.datasets.validators import DataValidationContext, build_default_dataset_validation_pipeline
from test.support.helpers import build_sample


class DatasetValidationPipelineTest(unittest.TestCase):
    def test_validate_collects_multiple_issues(self) -> None:
        pipeline = build_default_dataset_validation_pipeline()
        sample = build_sample(
            sample_id="problem-sample",
            history=[1.0, 1.0, 1.0],
            target=[math.inf],
            covariates={"known_future": [0.0, math.nan, 0.2, 0.3]},
        )

        report = pipeline.validate([sample], DataValidationContext(context_length=4, horizon=2))

        self.assertFalse(report.passed)
        self.assertTrue(any("context length mismatch" in issue for issue in report.issues))
        self.assertTrue(any("horizon length mismatch" in issue for issue in report.issues))
        self.assertTrue(any("non-finite value found" in issue for issue in report.issues))
        self.assertTrue(any("low variance sequence" in issue for issue in report.issues))
