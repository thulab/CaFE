from __future__ import annotations

import unittest

from backend.app.main import create_backend_app
from frontend.app import create_app
from test.support.helpers import (
    AsgiBackendProvider,
    FakeHuggingFaceRunner,
    backend_request,
    backend_request_raw,
    temporary_runtime_dir,
    write_demo_csv,
)


def run_smoke_flow() -> None:
    with temporary_runtime_dir(prefix="smoke-") as runtime_root:
        csv_path = write_demo_csv(runtime_root)

        backend_app = create_backend_app(runtime_root)
        backend_app.state.engine.huggingface_runner_factory = FakeHuggingFaceRunner

        assert backend_request(backend_app, "GET", "/health").status_code == 200

        user_overview_before = backend_request(backend_app, "GET", "/api/v1/overview/user").json()
        assert len(user_overview_before["models"]) >= 2
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
            {
                "model_id": "amazon-chronos-2",
                "batch_id": batch["batch_id"],
                "model_runtime_parameters": {"batch_size": 2, "use_covariates": True},
                "evaluation_metrics": ["mse", "smape", "composite_score"],
            },
        ).json()
        assert builtin_task["status"] == "succeeded"
        assert builtin_task["spec"]["model_runtime_parameters"]["batch_size"] == 2

        frontend_app = create_app(provider=AsgiBackendProvider(backend_app))
        frontend_client = frontend_app.test_client()

        user_page = frontend_client.get("/")
        assert user_page.status_code == 200
        page_text = user_page.get_data(as_text=True)
        assert "提交测试模型" in page_text
        assert "榜单模型" in page_text

        admin_page = frontend_client.get("/admin")
        assert admin_page.status_code == 200
        admin_page_text = admin_page.get_data(as_text=True)
        assert "数据集与任务管理页" in admin_page_text
        assert "批次数据" in admin_page_text
        assert "CSV 直接加载" in admin_page_text
        assert "注册通用模型" not in admin_page_text

        submitted_page = frontend_client.post(
            "/actions/models/submit",
            data={
                "huggingface_url": "https://huggingface.co/org/demo-forecast-model",
                "name": "Demo Forecast HF",
                "model_id": "",
                "manual": "用于验证 Hugging Face 提交流程。",
                "revision": "",
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
            "/actions/datasets/create",
            data={
                "source_mode": "csv",
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
            "/actions/datasets/create",
            data={
                "source_mode": "generate",
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
            data={
                "model_id": huggingface_model["model_id"],
                "batch_id": latest_batch,
                "runtime_param__integer__batch_size": "1",
                "evaluation_metrics": ["mse", "mae", "smape", "composite_score"],
            },
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

        invalid_csv_path = runtime_root / "invalid.csv"
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
                "delimiter": ",",
            },
        )
        assert invalid_csv_response.status_code == 400


class SmokeFlowIntegrationTest(unittest.TestCase):
    def test_smoke_flow(self) -> None:
        run_smoke_flow()


if __name__ == "__main__":
    run_smoke_flow()
