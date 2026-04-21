from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from backend.app.datasets.benchmark_v1.anchor import build_anchor_stats_artifacts
from backend.app.datasets.benchmark_v1.calibration import calibrate_family
from backend.app.datasets.benchmark_v1.families import FAMILY_GENERATORS
from backend.app.datasets.benchmark_v1.features import extract_features
from backend.app.datasets.benchmark_v1.generate import build_benchmark_artifacts
from backend.app.datasets.benchmark_v1.metrics import mase, relative_skill, smape
from backend.app.datasets.benchmark_v1.runner import run_model_eval
from backend.app.datasets.benchmark_v1.aggregate import make_report_artifacts
from backend.app.datasets.benchmark_v1.utils import adjacent_meta_path, read_json
from test.support.helpers import temporary_runtime_dir


class BenchmarkV1FeatureTest(unittest.TestCase):
    def test_feature_translation_and_scale_invariance_for_shape_features(self) -> None:
        t = np.arange(240, dtype=float)
        base = 0.03 * t + 2.5 * np.sin(2 * np.pi * t / 24) + 0.1 * np.cos(2 * np.pi * t / 12)
        shifted = 10.0 + 3.0 * base

        features_base = extract_features(base)
        features_shifted = extract_features(shifted)

        for name in ["trend_strength", "seasonal_strength", "spectral_entropy", "acf_half_life"]:
            self.assertLess(abs(features_base[name] - features_shifted[name]), 0.05)


class BenchmarkV1GeneratorTest(unittest.TestCase):
    def test_generators_emit_finite_nonempty_series(self) -> None:
        rng = np.random.default_rng(42)
        for family, generator in FAMILY_GENERATORS.items():
            output = generator(
                length=320,
                season_length=24,
                control_lambda=0.6,
                rng=rng,
                anchor_features={"spectral_entropy": 0.5},
            )
            self.assertEqual(len(output.values), 320, family)
            self.assertTrue(np.isfinite(output.values).all(), family)
            self.assertGreater(float(np.std(output.values)), 0.0, family)


class BenchmarkV1CalibrationTest(unittest.TestCase):
    def test_calibration_is_monotone(self) -> None:
        calibration, frame = calibrate_family(
            family="trend",
            n_candidates=64,
            horizon_ratio=0.5,
            seed=123,
        )
        scores = [calibration.score(value) for value in frame["lambda"].tolist()]
        self.assertTrue(all(left <= right + 1e-8 for left, right in zip(scores, scores[1:])))


class BenchmarkV1MetricTest(unittest.TestCase):
    def test_metrics_are_computed(self) -> None:
        history = np.array([1.0, 2.0, 1.0, 2.0])
        target = np.array([1.0, 2.0])
        forecast = np.array([1.5, 1.5])

        self.assertGreater(mase(history, target, forecast, season_length=2), 0.0)
        self.assertGreater(smape(target, forecast), 0.0)
        self.assertAlmostEqual(relative_skill(0.5, 1.0), 0.5)


class BenchmarkV1PipelineTest(unittest.TestCase):
    def test_pipeline_smoke_uses_bootstrap_anchor_metadata(self) -> None:
        with temporary_runtime_dir(prefix="benchmark-v1-") as runtime_root:
            anchor_path = Path(runtime_root) / "anchor_stats.parquet"
            benchmark_path = Path(runtime_root) / "benchmark_v1.parquet"
            eval_dir = Path(runtime_root) / "eval"
            report_dir = Path(runtime_root) / "reports"

            build_anchor_stats_artifacts(
                output_path=anchor_path,
                gift_root=None,
                tfb_root=None,
                n_clusters=6,
                bootstrap_size=48,
                seed=7,
            )
            build_benchmark_artifacts(
                anchor_stats_path=anchor_path,
                output_path=benchmark_path,
                anchor_track_size=5,
                diagnostic_per_cell=1,
                seed=7,
            )
            result_path = run_model_eval(
                model_name="last_value",
                benchmark_path=benchmark_path,
                output_dir=eval_dir,
                seeds=[0],
            )
            make_report_artifacts(
                benchmark_path=benchmark_path,
                eval_dir=eval_dir,
                output_dir=report_dir,
                real_eval_path=None,
            )

            self.assertTrue(anchor_path.exists())
            self.assertTrue(benchmark_path.exists())
            self.assertTrue(result_path.exists())
            self.assertTrue((report_dir / "summary.json").exists())
            benchmark = pd.read_parquet(benchmark_path)
            self.assertIn("baseline_mase", benchmark.columns)
            meta = read_json(adjacent_meta_path(benchmark_path))
            self.assertEqual(meta["benchmark_version"], "v1-s7")
            self.assertEqual(meta["anchor_mode"], "bootstrap")
            self.assertIn("validation_summary", meta)


if __name__ == "__main__":
    unittest.main()
