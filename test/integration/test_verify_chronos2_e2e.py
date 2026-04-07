from __future__ import annotations

import asyncio
import os
import unittest

import httpx

from backend.app.config import get_settings
from backend.app.main import create_backend_app
from test.support.helpers import temporary_runtime_dir


def run_real_chronos2_e2e() -> dict[str, object]:
    settings = get_settings()
    user_defaults = settings.ui.user_model_submission
    admin_defaults = settings.ui.admin_batch_generation
    repo_id = os.environ.get("TSBENCHMARK_CHRONOS2_REPO_ID", user_defaults.repo_id)

    async def run() -> dict[str, object]:
        with temporary_runtime_dir(prefix="chronos2-") as runtime_root:
            app = create_backend_app(runtime_root)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                register = await client.post(
                    "/api/v1/models/register/huggingface",
                    json={
                        "repo_id": repo_id,
                        "name": user_defaults.name,
                        "manual": "Real Chronos-2 end-to-end validation run.",
                        "task": user_defaults.task,
                        "revision": user_defaults.revision,
                        "trust_remote_code": user_defaults.trust_remote_code,
                        "device_map": user_defaults.device_map,
                        "torch_dtype": user_defaults.torch_dtype,
                        "batch_size": min(user_defaults.batch_size, 2),
                        "context_length": 128,
                        "use_covariates": user_defaults.use_covariates,
                        "cross_learning": user_defaults.cross_learning,
                        "load_retries": user_defaults.load_retries,
                        "load_retry_backoff_seconds": user_defaults.load_retry_backoff_seconds,
                    },
                )
                register.raise_for_status()
                model = register.json()

                load = await client.post(f"/api/v1/models/{model['model_id']}/load")
                load.raise_for_status()

                batch = await client.post(
                    "/api/v1/datasets/generate",
                    json={
                        "track": admin_defaults.track,
                        "sample_count": 1,
                        "context_length": admin_defaults.context_length,
                        "horizon": 16,
                        "seed": admin_defaults.seed + 6,
                    },
                )
                batch.raise_for_status()
                batch_id = batch.json()["batch_id"]

                task = await client.post(
                    "/api/v1/tasks/run",
                    json={"model_id": model["model_id"], "batch_id": batch_id},
                )
                task.raise_for_status()

                payload = task.json()
                return {
                    "model_id": model["model_id"],
                    "batch_id": batch_id,
                    "task_id": payload["task_id"],
                    "score": payload["metrics"]["composite_score"],
                }

    result = asyncio.run(run())
    print("VERIFY_CHRONOS2_OK")
    print(result)
    return result


@unittest.skipUnless(
    os.environ.get("TSBENCHMARK_RUN_CHRONOS2_E2E") == "1",
    "set TSBENCHMARK_RUN_CHRONOS2_E2E=1 to run the real Chronos-2 integration test",
)
class Chronos2E2EIntegrationTest(unittest.TestCase):
    def test_real_chronos2_e2e(self) -> None:
        result = run_real_chronos2_e2e()
        self.assertIn("score", result)


if __name__ == "__main__":
    run_real_chronos2_e2e()
