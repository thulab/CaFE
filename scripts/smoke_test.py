from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.huggingface import HuggingFaceForecast
from backend.app.main import create_backend_app
from frontend.app import BackendProvider, create_app


class AsgiBackendProvider(BackendProvider):
    def __init__(self, app) -> None:
        self.app = app

    def _request(self, method: str, path: str, payload=None):
        async def once():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.request(method, path, json=payload)
                response.raise_for_status()
                return response.json()

        return asyncio.run(once())

    def fetch_user_overview(self):
        return self._request("GET", "/api/v1/overview/user")

    def fetch_admin_overview(self):
        return self._request("GET", "/api/v1/overview/admin")

    def fetch_report(self, report_id):
        return self._request("GET", f"/api/v1/reports/{report_id}")

    def generate_batch(self, payload):
        return self._request("POST", "/api/v1/datasets/generate", payload)

    def load_csv_batch(self, payload):
        return self._request("POST", "/api/v1/datasets/load/csv", payload)

    def run_task(self, payload):
        return self._request("POST", "/api/v1/tasks/run", payload)

    def register_model(self, payload):
        return self._request("POST", "/api/v1/models/register", payload)

    def submit_huggingface_model(self, payload):
        return self._request("POST", "/api/v1/models/register/huggingface", payload)

    def load_model(self, model_id):
        return self._request("POST", f"/api/v1/models/{model_id}/load")


class FakeHuggingFaceRunner:
    def __init__(self, config) -> None:
        self.config = config

    def load(self) -> None:
        if self.config.repo_id == "org/broken-forecast-model":
            raise RuntimeError("intentional runner failure")
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


def backend_request(app, method: str, path: str, payload=None):
    async def once():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.request(method, path, json=payload)
            response.raise_for_status()
            return response

    return asyncio.run(once())


def backend_request_raw(app, method: str, path: str, payload=None):
    async def once():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, json=payload)

    return asyncio.run(once())


def main() -> None:
    with TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "demo.csv"
        csv_path.write_text(
            "\n".join(
                ["sample_id,step,target,calendar_signal"]
                + [f"series_a,{idx},{10 + idx * 0.1:.1f},{(idx % 24) / 24:.4f}" for idx in range(12)]
            ),
            encoding="utf-8",
        )

        backend_app = create_backend_app(Path(tmpdir))
        backend_app.state.engine.huggingface_runner_factory = FakeHuggingFaceRunner

        assert backend_request(backend_app, "GET", "/health").status_code == 200

        user_overview_before = backend_request(backend_app, "GET", "/api/v1/overview/user").json()
        assert len(user_overview_before["models"]) >= 3
        assert len(user_overview_before["track_leaderboards"]) == 4

        batch = backend_request(
            backend_app,
            "POST",
            "/api/v1/datasets/generate",
            {
                "track": "covariate_robustness",
                "sample_count": 6,
                "context_length": 72,
                "horizon": 18,
                "seed": 11,
            },
        ).json()
        assert batch["validation"]["passed"] is True

        builtin_task = backend_request(
            backend_app,
            "POST",
            "/api/v1/tasks/run",
            {"model_id": "seasonal-naive-stub", "batch_id": batch["batch_id"]},
        ).json()
        assert builtin_task["status"] == "succeeded"

        frontend_app = create_app(provider=AsgiBackendProvider(backend_app))
        frontend_client = frontend_app.test_client()

        user_page = frontend_client.get("/")
        assert user_page.status_code == 200
        assert "提交 Hugging Face 模型" in user_page.get_data(as_text=True)

        admin_page = frontend_client.get("/admin")
        assert admin_page.status_code == 200
        assert "数据集与任务管理页" in admin_page.get_data(as_text=True)

        manual_model_page = frontend_client.post(
            "/actions/models/register",
            data={
                "model_id": "custom-recent-mean",
                "name": "Custom Recent Mean",
                "adapter": "recent_mean",
                "source_type": "uploaded_stub",
                "capabilities": "forecast,manual",
                "metadata_json": '{"owner":"smoke"}',
                "manual": "Manual registration smoke path.",
            },
            follow_redirects=True,
        )
        assert manual_model_page.status_code == 200
        assert "已注册" in manual_model_page.get_data(as_text=True)

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

        user_overview_after_submit = backend_request(backend_app, "GET", "/api/v1/overview/user").json()
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

        csv_import_page = frontend_client.post(
            "/actions/datasets/load_csv",
            data={
                "csv_path": str(csv_path),
                "track": "forecast_accuracy",
                "context_length": "8",
                "horizon": "4",
                "max_samples": "1",
                "batch_id_prefix": "csv",
                "sample_id_column": "sample_id",
                "step_column": "step",
                "target_column": "target",
                "covariate_columns": "calendar_signal",
                "delimiter": ",",
                "processors_json": "",
            },
            follow_redirects=True,
        )
        assert csv_import_page.status_code == 200
        assert "已导入" in csv_import_page.get_data(as_text=True)

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

        refreshed_admin = backend_request(backend_app, "GET", "/api/v1/overview/admin").json()
        latest_batch = refreshed_admin["batches"][0]["batch_id"]
        hf_task_page = frontend_client.post(
            "/actions/run",
            data={"model_id": huggingface_model["model_id"], "batch_id": latest_batch},
            follow_redirects=True,
        )
        assert hf_task_page.status_code == 200
        assert "已完成" in hf_task_page.get_data(as_text=True)

        refreshed_tasks = backend_request(backend_app, "GET", "/api/v1/tasks").json()
        latest_task = next(task for task in refreshed_tasks if task["model_id"] == huggingface_model["model_id"])
        report_id = latest_task["report_id"]
        report_payload = backend_request(backend_app, "GET", f"/api/v1/reports/{report_id}").json()
        assert report_payload["report_id"] == report_id
        assert report_payload["summary"]

        report_page = frontend_client.get(f"/reports/{report_id}")
        assert report_page.status_code == 200
        assert report_id in report_page.get_data(as_text=True)

        missing_report_page = frontend_client.get("/reports/report-missing", follow_redirects=True)
        assert missing_report_page.status_code == 200
        assert "报告加载失败" in missing_report_page.get_data(as_text=True)

        invalid_csv_path = Path(tmpdir) / "invalid.csv"
        invalid_csv_path.write_text(
            "\n".join(
                ["sample_id,step,target,calendar_signal"]
                + [f"series_bad,{idx},{10 + idx * 0.1:.1f},nan" for idx in range(12)]
            ),
            encoding="utf-8",
        )
        invalid_csv_response = backend_request_raw(
            backend_app,
            "POST",
            "/api/v1/datasets/load/csv",
            {
                "source_type": "csv",
                "csv_path": str(invalid_csv_path),
                "track": "forecast_accuracy",
                "context_length": 8,
                "horizon": 4,
                "sample_id_column": "sample_id",
                "step_column": "step",
                "target_column": "target",
                "covariate_columns": ["calendar_signal"],
            },
        )
        assert invalid_csv_response.status_code == 400
        assert "non-finite covariate value" in invalid_csv_response.text

        broken_model = backend_request(
            backend_app,
            "POST",
            "/api/v1/models/register/huggingface",
            {
                "repo_id": "org/broken-forecast-model",
                "name": "Broken HF",
                "manual": "Used to verify internal failure classification.",
                "task": "chronos-2",
                "batch_size": 1,
                "load_retries": 1,
                "load_retry_backoff_seconds": 0.0,
            },
        ).json()
        broken_task_response = backend_request_raw(
            backend_app,
            "POST",
            "/api/v1/tasks/run",
            {"model_id": broken_model["model_id"], "batch_id": latest_batch},
        )
        assert broken_task_response.status_code == 500
        assert "intentional runner failure" in broken_task_response.text

        refreshed_user = backend_request(backend_app, "GET", "/api/v1/overview/user").json()
        assert refreshed_user["overall_leaderboard"]
        assert any(board["entries"] for board in refreshed_user["track_leaderboards"])
        assert all("rank_sum" in item for item in refreshed_user["overall_leaderboard"])
        assert all("track_ranks" in item for item in refreshed_user["overall_leaderboard"])
        rank_sums = [item["rank_sum"] for item in refreshed_user["overall_leaderboard"]]
        assert rank_sums == sorted(rank_sums)
        assert any(item["model_id"] == broken_model["model_id"] for item in refreshed_user["overall_leaderboard"])
        broken_entry = next(item for item in refreshed_user["overall_leaderboard"] if item["model_id"] == broken_model["model_id"])
        assert broken_entry["covered_tracks"] == 0
        for board in refreshed_user["track_leaderboards"]:
            model_ids = [item["model_id"] for item in board["entries"]]
            assert len(model_ids) == len(set(model_ids))

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
