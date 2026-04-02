from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.main import create_backend_app


def main() -> None:
    repo_id = os.environ.get("TSBENCHMARK_CHRONOS2_REPO_ID", "amazon/chronos-2")
    with TemporaryDirectory() as tmpdir:
        app = create_backend_app(Path(tmpdir))
        client = TestClient(app)

        register = client.post(
            "/api/v1/models/register/huggingface",
            json={
                "repo_id": repo_id,
                "name": "Amazon Chronos-2",
                "manual": "Real Chronos-2 end-to-end validation run.",
                "task": "chronos-2",
                "revision": "main",
                "trust_remote_code": True,
                "device_map": "cpu",
                "torch_dtype": "float32",
                "batch_size": 2,
                "context_length": 128,
                "use_covariates": True,
                "cross_learning": False,
                "load_retries": 3,
                "load_retry_backoff_seconds": 1.0,
            },
        )
        register.raise_for_status()
        model = register.json()

        load = client.post(f"/api/v1/models/{model['model_id']}/load")
        load.raise_for_status()

        batch = client.post(
            "/api/v1/datasets/generate",
            json={
                "track": "forecast_accuracy",
                "sample_count": 1,
                "context_length": 96,
                "horizon": 16,
                "seed": 23,
            },
        )
        batch.raise_for_status()
        batch_id = batch.json()["batch_id"]

        task = client.post("/api/v1/tasks/run", json={"model_id": model["model_id"], "batch_id": batch_id})
        task.raise_for_status()

        payload = task.json()
        print("VERIFY_CHRONOS2_OK")
        print(
            {
                "model_id": model["model_id"],
                "batch_id": batch_id,
                "task_id": payload["task_id"],
                "score": payload["metrics"]["composite_score"],
            }
        )


if __name__ == "__main__":
    main()
