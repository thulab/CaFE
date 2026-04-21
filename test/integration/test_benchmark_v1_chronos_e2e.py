from __future__ import annotations

import os
import unittest
from pathlib import Path

import pandas as pd

from backend.app.datasets.benchmark_v1.runner import run_model_eval
from backend.app.datasets.benchmark_v1.utils import adjacent_meta_path, write_json
from test.support.helpers import temporary_runtime_dir


def run_real_v1_chronos_bolt_e2e() -> dict[str, object]:
    """Run one real Chronos-Bolt prediction through the v1 adapter path.

    This intentionally uses the external Python backend instead of a fake runner.
    Required environment:
    - TSBENCHMARK_RUN_V1_CHRONOS_E2E=1
    - TSBENCHMARK_MODEL_PYTHON points to an env with chronos/torch installed, or
      .venv-models/bin/python exists.
    - TSBENCHMARK_CHRONOS_DIR points to local Chronos-Bolt weights, or
      models/amazon_chronos_bolt_base exists.
    """
    with temporary_runtime_dir(prefix="benchmark-v1-chronos-") as runtime_root:
        root = Path(runtime_root)
        benchmark_path = root / "benchmark_v1_chronos_smoke.parquet"
        eval_dir = root / "eval"

        context = [float(value) for value in range(1, 65)]
        target = [65.0, 66.0, 67.0, 68.0]
        frame = pd.DataFrame(
            [
                {
                    "id": "chronos-bolt-smoke-001",
                    "context": context,
                    "target": target,
                    "horizon": len(target),
                    "season_length": 1,
                    "benchmark_version": "v1-chronos-e2e",
                }
            ]
        )
        frame.to_parquet(benchmark_path, index=False)
        write_json({"benchmark_version": "v1-chronos-e2e"}, adjacent_meta_path(benchmark_path))

        result_path = run_model_eval(
            model_name="chronos_bolt_base",
            benchmark_path=benchmark_path,
            output_dir=eval_dir,
            seeds=[0],
        )
        result = pd.read_parquet(result_path)
        if result.empty:
            raise AssertionError("Chronos-Bolt v1 eval produced no rows")
        row = result.iloc[0].to_dict()
        payload = {
            "model": row["model"],
            "series_id": row["series_id"],
            "benchmark_version": row["benchmark_version"],
            "mase": float(row["mase"]),
            "smape": float(row["smape"]),
            "runtime_ms": float(row["runtime_ms"]),
        }
        print("VERIFY_V1_CHRONOS_BOLT_OK")
        print(payload)
        return payload


@unittest.skipUnless(
    os.environ.get("TSBENCHMARK_RUN_V1_CHRONOS_E2E") == "1",
    "set TSBENCHMARK_RUN_V1_CHRONOS_E2E=1 to run the real v1 Chronos-Bolt integration test",
)
class BenchmarkV1ChronosBoltE2EIntegrationTest(unittest.TestCase):
    def test_real_v1_chronos_bolt_e2e(self) -> None:
        result = run_real_v1_chronos_bolt_e2e()
        self.assertEqual(result["model"], "chronos_bolt_base")
        self.assertEqual(result["benchmark_version"], "v1-chronos-e2e")
        self.assertGreaterEqual(result["runtime_ms"], 0.0)


if __name__ == "__main__":
    run_real_v1_chronos_bolt_e2e()
