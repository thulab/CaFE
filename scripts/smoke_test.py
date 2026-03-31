from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.main import create_backend_app
from frontend.app import BackendProvider, create_app


class TestClientProvider(BackendProvider):
    def __init__(self, client: TestClient) -> None:
        self.client = client

    def fetch_overview(self):
        response = self.client.get("/api/v1/overview")
        response.raise_for_status()
        return response.json()

    def generate_batch(self, payload):
        response = self.client.post("/api/v1/datasets/generate", json=payload)
        response.raise_for_status()
        return response.json()

    def run_task(self, payload):
        response = self.client.post("/api/v1/tasks/run", json=payload)
        response.raise_for_status()
        return response.json()


def main() -> None:
    with TemporaryDirectory() as tmpdir:
        backend_app = create_backend_app(Path(tmpdir))
        backend_client = TestClient(backend_app)

        health = backend_client.get("/health")
        assert health.status_code == 200

        overview_before = backend_client.get("/api/v1/overview").json()
        assert len(overview_before["models"]) >= 3

        batch = backend_client.post(
            "/api/v1/datasets/generate",
            json={
                "track": "covariate_robustness",
                "sample_count": 6,
                "context_length": 72,
                "horizon": 18,
                "seed": 11,
            },
        ).json()
        assert batch["validation"]["passed"] is True

        task = backend_client.post(
            "/api/v1/tasks/run",
            json={"model_id": "seasonal-naive-stub", "batch_id": batch["batch_id"]},
        ).json()
        assert task["status"] == "succeeded"
        assert task["metrics"]["composite_score"] != 0

        report = backend_client.get(f"/api/v1/reports/{task['report_id']}").json()
        assert report["bad_cases"]

        leaderboard = backend_client.get("/api/v1/leaderboard").json()
        assert leaderboard

        frontend_app = create_app(provider=TestClientProvider(backend_client))
        frontend_client = frontend_app.test_client()

        page = frontend_client.get("/")
        assert page.status_code == 200
        assert "动态评测集与在线评测系统" in page.get_data(as_text=True)

        console = frontend_client.get("/console")
        assert console.status_code == 200
        assert "动态评测控制台" in console.get_data(as_text=True)

        page = frontend_client.post(
            "/actions/generate",
            data={
                "track": "noise_robustness",
                "sample_count": "4",
                "context_length": "64",
                "horizon": "16",
                "seed": "13",
                "view": "gallery",
            },
            follow_redirects=True,
        )
        assert page.status_code == 200
        assert "生成批次" in page.get_data(as_text=True)

        refreshed = backend_client.get("/api/v1/overview").json()
        latest_batch = refreshed["batches"][0]["batch_id"]
        page = frontend_client.post(
            "/actions/run",
            data={"model_id": "recent-mean-stub", "batch_id": latest_batch, "view": "console"},
            follow_redirects=True,
        )
        assert page.status_code == 200
        assert "评测任务" in page.get_data(as_text=True)

        print("SMOKE_OK")
        print(
            {
                "models": len(refreshed["models"]),
                "batches": len(refreshed["batches"]),
                "leaderboard_entries": len(backend_client.get("/api/v1/leaderboard").json()),
            }
        )


if __name__ == "__main__":
    main()
