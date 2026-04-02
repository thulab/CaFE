from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.huggingface import HuggingFaceForecast
from backend.app.main import create_backend_app
from frontend.app import BackendProvider, create_app


class TestClientProvider(BackendProvider):
    def __init__(self, client: TestClient) -> None:
        self.client = client

    def fetch_user_overview(self):
        response = self.client.get("/api/v1/overview/user")
        response.raise_for_status()
        return response.json()

    def fetch_admin_overview(self):
        response = self.client.get("/api/v1/overview/admin")
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

    def submit_huggingface_model(self, payload):
        response = self.client.post("/api/v1/models/register/huggingface", json=payload)
        response.raise_for_status()
        return response.json()

    def load_model(self, model_id):
        response = self.client.post(f"/api/v1/models/{model_id}/load")
        response.raise_for_status()
        return response.json()


class FakeHuggingFaceRunner:
    def __init__(self, config) -> None:
        self.config = config

    def load(self) -> None:
        return None

    def forecast(self, sample, track) -> HuggingFaceForecast:
        return HuggingFaceForecast(
            prediction=[round(sample.history[-1], 4)] * len(sample.target),
            latency_ms=7.5,
            token_count=160,
            notes={"decision": "fake_huggingface", "repo_id": self.config.repo_id, "track": track.value},
        )

    def forecast_batch(self, samples, track):
        return [self.forecast(sample, track) for sample in samples]


def main() -> None:
    with TemporaryDirectory() as tmpdir:
        backend_app = create_backend_app(Path(tmpdir))
        backend_app.state.engine.huggingface_runner_factory = FakeHuggingFaceRunner
        backend_client = TestClient(backend_app)

        assert backend_client.get("/health").status_code == 200

        user_overview_before = backend_client.get("/api/v1/overview/user").json()
        assert len(user_overview_before["models"]) >= 3
        assert len(user_overview_before["track_leaderboards"]) == 4

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

        builtin_task = backend_client.post(
            "/api/v1/tasks/run",
            json={"model_id": "seasonal-naive-stub", "batch_id": batch["batch_id"]},
        ).json()
        assert builtin_task["status"] == "succeeded"

        frontend_app = create_app(provider=TestClientProvider(backend_client))
        frontend_client = frontend_app.test_client()

        user_page = frontend_client.get("/")
        assert user_page.status_code == 200
        assert "提交 Hugging Face 模型" in user_page.get_data(as_text=True)

        admin_page = frontend_client.get("/admin")
        assert admin_page.status_code == 200
        assert "数据集与任务管理页" in admin_page.get_data(as_text=True)

        submitted_page = frontend_client.post(
            "/actions/models/submit",
            data={
                "repo_id": "org/demo-forecast-model",
                "name": "Demo Forecast HF",
                "model_id": "",
                "manual": "用于验证 Hugging Face 提交流程。",
                "task": "chronos-2",
                "revision": "",
                "max_new_tokens": "64",
                "temperature": "0.0",
                "top_p": "1.0",
                "device_map": "",
                "torch_dtype": "",
                "attn_implementation": "",
                "batch_size": "1",
                "context_length": "",
                "use_covariates": "on",
                "load_retries": "3",
                "load_retry_backoff_seconds": "1.0",
            },
            follow_redirects=True,
        )
        assert submitted_page.status_code == 200
        assert "已提交" in submitted_page.get_data(as_text=True)

        user_overview_after_submit = backend_client.get("/api/v1/overview/user").json()
        huggingface_model = next(
            model
            for model in user_overview_after_submit["models"]
            if model["source_type"] == "huggingface_hub" and model["huggingface"]["repo_id"] == "org/demo-forecast-model"
        )

        load_page = frontend_client.post(
            "/actions/models/load",
            data={"model_id": huggingface_model["model_id"]},
            follow_redirects=True,
        )
        assert load_page.status_code == 200
        assert "已加载" in load_page.get_data(as_text=True)

        admin_generate_page = frontend_client.post(
            "/actions/generate",
            data={
                "track": "noise_robustness",
                "sample_count": "4",
                "context_length": "64",
                "horizon": "16",
                "seed": "13",
            },
            follow_redirects=True,
        )
        assert admin_generate_page.status_code == 200
        assert "生成批次" in admin_generate_page.get_data(as_text=True)

        refreshed_admin = backend_client.get("/api/v1/overview/admin").json()
        latest_batch = refreshed_admin["batches"][0]["batch_id"]
        hf_task_page = frontend_client.post(
            "/actions/run",
            data={"model_id": huggingface_model["model_id"], "batch_id": latest_batch},
            follow_redirects=True,
        )
        assert hf_task_page.status_code == 200
        assert "已完成" in hf_task_page.get_data(as_text=True)

        refreshed_user = backend_client.get("/api/v1/overview/user").json()
        assert refreshed_user["overall_leaderboard"]
        assert any(board["entries"] for board in refreshed_user["track_leaderboards"])

        print("SMOKE_OK")
        print(
            {
                "models": len(refreshed_user["models"]),
                "overall_leaderboard_entries": len(refreshed_user["overall_leaderboard"]),
                "recent_admin_tasks": len(refreshed_admin["recent_tasks"]),
            }
        )


if __name__ == "__main__":
    main()
