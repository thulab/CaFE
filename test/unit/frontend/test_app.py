from __future__ import annotations

import unittest

from frontend.app import create_app


class FakeBackendProvider:
    def __init__(self) -> None:
        self.last_submit_payload = None
        self.last_generate_payload = None
        self.last_csv_payload = None
        self.last_run_payload = None
        self.last_load_model_id = None
        self.last_v1_anchor_payload = None
        self.last_v1_benchmark_payload = None
        self.last_v1_eval_payload = None
        self.last_v1_report_payload = None

    def _track(self) -> dict[str, object]:
        return {
            "track": "forecast_accuracy",
            "track_variant_id": "forecast_accuracy.clean",
            "name": "Forecast Accuracy",
            "noise_mode": "clean",
        }

    def _model(self) -> dict[str, object]:
        return {
            "model_id": "amazon-chronos-2",
            "name": "Chronos 2",
            "adapter": "huggingface_chronos2",
            "source_type": "huggingface_hub",
            "manual": "A forecasting model.",
            "runtime_status": "ready",
            "last_error": None,
            "huggingface": {
                "repo_id": "amazon/chronos-2",
                "task": "chronos-2",
                "revision": None,
            },
            "spec": {
                "source": {
                    "huggingface_url": "https://huggingface.co/amazon/chronos-2",
                    "local_weight_path": "/models/chronos2",
                },
                "runtime_parameter_definitions": [
                    {
                        "name": "batch_size",
                        "label": "Batch Size",
                        "value_type": "integer",
                        "required": False,
                        "description": "batch size",
                    },
                    {
                        "name": "use_covariates",
                        "label": "Use Covariates",
                        "value_type": "boolean",
                        "required": True,
                        "description": "use covariates",
                    },
                ],
            },
        }

    def _recent_task(self) -> dict[str, object]:
        return {
            "task_id": "task-1",
            "model_id": "amazon-chronos-2",
            "model_name": "Chronos 2",
            "batch_id": "batch-1",
            "track": "forecast_accuracy",
            "track_variant_id": "forecast_accuracy.clean",
            "status": "succeeded",
            "primary_metric_id": "mse",
            "primary_metric_value": 0.123,
            "composite_score": 98.1,
            "execution_repeat_count": 3,
            "task_run_count": 3,
            "report_id": "report-task-1",
            "error_message": None,
            "model_runtime_parameters": {"batch_size": 2},
            "evaluation_metrics": ["mse", "smape", "composite_score"],
        }

    def _batch(self) -> dict[str, object]:
        return {
            "batch_id": "batch-1",
            "track": "forecast_accuracy",
            "track_variant_id": "forecast_accuracy.clean",
            "policy": "synthetic",
            "sample_count": 8,
            "context_length": 64,
            "horizon": 16,
            "validation_passed": True,
        }

    def fetch_user_overview(self, metric_id: str = "mse"):
        return {
            "models": [self._model()],
            "track_leaderboards": [
                {
                    "track": "forecast_accuracy.clean",
                    "track_label": "Forecast Accuracy",
                    "metric_id": metric_id,
                    "entries": [
                        {
                            "rank": 1,
                            "model_name": "Chronos 2",
                            "metric_id": metric_id,
                            "metric_value": 0.123,
                            "model_id": "amazon-chronos-2",
                            "batch_id": "batch-1",
                            "metric_snapshot": {"mse": 0.123, "smape": 0.222, "latency_ms": 10.0},
                        }
                    ],
                }
            ],
            "tracks": [self._track()],
            "overall_metric_id": metric_id,
            "overall_leaderboard": [
                {
                    "rank": 1,
                    "model_name": "Chronos 2",
                    "rank_sum": 1,
                    "covered_tracks": 1,
                    "mean_metric_value": 0.123,
                    "track_ranks": {"forecast_accuracy.clean": 1},
                }
            ],
        }

    def fetch_admin_overview(self, metric_id: str = "mse"):
        return {
            "tracks": [self._track()],
            "models": [self._model()],
            "batches": [self._batch()],
            "recent_tasks": [self._recent_task()],
            "leaderboard": [
                {
                    "rank": 1,
                    "model_name": "Chronos 2",
                    "rank_sum": 1,
                    "covered_tracks": 1,
                    "mean_metric_value": 0.123,
                    "track_ranks": {"forecast_accuracy.clean": 1},
                }
            ],
            "overall_metric_id": metric_id,
            "overall_leaderboard_strategy": "rank_sum",
        }

    def fetch_task(self, task_id: str):
        if task_id == "missing-task":
            raise RuntimeError("task missing")
        return {
            "task_id": task_id,
            "model_id": "amazon-chronos-2",
            "batch_id": "batch-1",
            "track": "forecast_accuracy",
            "track_variant_id": "forecast_accuracy.clean",
            "status": "succeeded",
            "spec": {
                "model_runtime_parameters": {"batch_size": 2},
                "evaluation_metrics": ["mse", "smape"],
                "execution_repeat_count": 3,
            },
            "metrics": {
                "mse": 0.123,
                "mae": 0.101,
                "smape": 0.222,
                "mean_latency_ms": 10.5,
                "mean_token_count": 128.0,
                "composite_score": 98.1,
                "stability_stats": {},
            },
            "report_id": "report-task-lookup",
            "task_runs": [{"run_id": "run-1"}, {"run_id": "run-2"}],
            "error_message": None,
        }

    def fetch_report(self, report_id: str):
        if report_id == "missing":
            raise RuntimeError("report missing")
        return {
            "report_id": report_id,
            "task_id": "task-1",
            "summary": "summary",
            "strengths": ["stable"],
            "risks": ["slow"],
            "bad_cases": ["series-a"],
            "distribution": {"mse_p50": 0.12},
            "run_ids": ["run-1"],
        }

    def generate_batch(self, payload):
        self.last_generate_payload = payload
        return {"batch_id": "batch-1"}

    def load_csv_batch(self, payload):
        self.last_csv_payload = payload
        return {"batch_id": "csv-1"}

    def fetch_v1_artifacts(self):
        return [
            {
                "kind": "benchmark",
                "path": "/runtime/generated/benchmark_v1/benchmark_v1.parquet",
                "benchmark_version": "v1-s7",
                "anchor_mode": "bootstrap",
                "n_series": 80,
                "validation_summary": {"n_series": 80, "anchor_mode": "bootstrap"},
            }
        ]

    def build_v1_anchor_stats(self, payload):
        self.last_v1_anchor_payload = payload
        return {"path": "/runtime/generated/benchmark_v1/anchor_stats.parquet"}

    def build_v1_benchmark(self, payload):
        self.last_v1_benchmark_payload = payload
        return {"path": "/runtime/generated/benchmark_v1/benchmark_v1.parquet"}

    def run_v1_eval(self, payload):
        self.last_v1_eval_payload = payload
        return {"path": "/runtime/generated/benchmark_v1/eval/last_value.parquet"}

    def make_v1_report(self, payload):
        self.last_v1_report_payload = payload
        return {"path": "/runtime/generated/benchmark_v1/reports"}

    def run_task(self, payload):
        self.last_run_payload = payload
        return {"task_id": "task-created"}

    def register_model(self, payload):
        return {"model_id": payload["model_id"]}

    def submit_huggingface_model(self, payload):
        self.last_submit_payload = payload
        return {"model_id": payload.get("model_id") or "hf-model"}

    def load_model(self, model_id: str):
        self.last_load_model_id = model_id
        return {"model_id": model_id}


class FrontendAppTest(unittest.TestCase):
    def test_user_page_only_shows_leaderboard_content(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.get("/")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("TSBenchmark 榜单", body)
        self.assertIn("总榜", body)
        self.assertIn("Forecast Accuracy", body)
        self.assertNotIn("提交测试模型", body)
        self.assertNotIn("榜单模型", body)
        self.assertNotIn("管理页", body)

    def test_submit_model_page_only_shows_upload_form(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.get("/submit-model")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("上传 Hugging Face 模型", body)
        self.assertIn("Hugging Face URL", body)
        self.assertNotIn("Overall Ranking", body)
        self.assertNotIn("总体排行榜", body)
        self.assertNotIn("批次数据", body)

    def test_submit_model_only_sends_source_fields_and_redirects_back_to_submit_page(self) -> None:
        provider = FakeBackendProvider()
        app = create_app(provider=provider)
        client = app.test_client()

        response = client.post(
            "/actions/models/submit",
            data={
                "huggingface_url": "https://huggingface.co/org/demo-model",
                "name": "Demo",
                "model_id": "",
                "manual": "manual",
                "revision": "",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/submit-model?message=", response.headers["Location"])
        self.assertEqual(
            provider.last_submit_payload,
            {
                "huggingface_url": "https://huggingface.co/org/demo-model",
                "name": "Demo",
                "model_id": None,
                "manual": "manual",
                "revision": None,
                "trust_remote_code": False,
            },
        )

    def test_admin_root_redirects_to_datasets_page(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.get("/admin", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/datasets", response.headers["Location"])

    def test_admin_datasets_page_shows_only_dataset_module(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.get("/admin/datasets")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("数据集管理", body)
        self.assertIn("创建或加载数据集", body)
        self.assertIn("CSV 直接加载", body)
        self.assertIn("V1 Benchmark", body)
        self.assertIn("Build Anchor Stats", body)
        self.assertIn("最近批次", body)
        self.assertNotIn("模型模块", body)
        self.assertNotIn("按 task_id 查询", body)

    def test_admin_datasets_v1_anchor_action_sends_payload(self) -> None:
        provider = FakeBackendProvider()
        app = create_app(provider=provider)
        client = app.test_client()

        response = client.post(
            "/actions/benchmark-v1/anchor-stats",
            data={
                "output_name": "anchor_smoke",
                "gift_root": "",
                "tfb_root": "",
                "n_clusters": "6",
                "bootstrap_size": "48",
                "seed": "7",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/datasets?message=", response.headers["Location"])
        self.assertEqual(
            provider.last_v1_anchor_payload,
            {
                "output_name": "anchor_smoke",
                "gift_root": None,
                "tfb_root": None,
                "n_clusters": 6,
                "bootstrap_size": 48,
                "seed": 7,
            },
        )

    def test_admin_benchmark_v1_page_shows_artifacts_and_forms(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.get("/admin/benchmark-v1")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("V1 Benchmark", body)
        self.assertIn("Build Anchor Stats", body)
        self.assertIn("Run Eval", body)
        self.assertIn("v1-s7", body)
        self.assertIn("bootstrap", body)

    def test_admin_benchmark_v1_eval_action_sends_payload(self) -> None:
        provider = FakeBackendProvider()
        app = create_app(provider=provider)
        client = app.test_client()

        response = client.post(
            "/actions/benchmark-v1/eval",
            data={
                "model": "last_value",
                "benchmark_path": "",
                "output_dir": "",
                "seeds": "0, 1",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/tasks?message=", response.headers["Location"])
        self.assertEqual(
            provider.last_v1_eval_payload,
            {"model": "last_value", "benchmark_path": None, "output_dir": None, "seeds": [0, 1]},
        )

    def test_admin_models_page_formats_models_card(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.get("/admin/models")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("模型模块", body)
        self.assertIn("模型链接", body)
        self.assertIn("https://huggingface.co/amazon/chronos-2", body)
        self.assertIn("权重文件位置", body)
        self.assertIn("/models/chronos2", body)
        self.assertIn("运行参数", body)
        self.assertIn("Batch Size", body)
        self.assertIn("Use Covariates", body)
        self.assertNotIn("请选择模型", body)
        self.assertNotIn("CSV 直接加载", body)

    def test_admin_tasks_page_shows_creation_and_query_modules(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.get("/admin/tasks")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("任务管理", body)
        self.assertIn("创建任务", body)
        self.assertIn("按 task_id 查询", body)
        self.assertIn("V1 模型评测", body)
        self.assertIn("Make V1 Report", body)
        self.assertIn("请选择模型", body)
        self.assertIn('name="runtime_param__integer__batch_size"', body)
        self.assertIn("最近任务", body)
        self.assertNotIn("CSV 直接加载", body)
        self.assertNotIn("总体排行榜", body)

    def test_admin_leaderboard_page_shows_only_leaderboard_module(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.get("/admin/leaderboard")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("榜单查看", body)
        self.assertIn("总体排行榜", body)
        self.assertIn("Rank Sum", body)
        self.assertNotIn("按 task_id 查询", body)
        self.assertNotIn("CSV 直接加载", body)

    def test_report_routes_use_new_admin_report_page_and_old_route_redirects(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.get("/admin/reports/report-task-1")
        redirect_response = client.get("/reports/report-task-1", follow_redirects=False)

        self.assertEqual(response.status_code, 200)
        self.assertIn("report-task-1", response.get_data(as_text=True))
        self.assertEqual(redirect_response.status_code, 302)
        self.assertIn("/admin/reports/report-task-1", redirect_response.headers["Location"])

    def test_report_detail_redirects_to_admin_tasks_when_missing(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.get("/admin/reports/missing", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/tasks?message=", response.headers["Location"])

    def test_register_model_invalid_json_redirects_to_admin_models(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.post(
            "/actions/models/register",
            data={
                "model_id": "model-a",
                "name": "Model A",
                "adapter": "recent_mean",
                "manual": "manual",
                "metadata_json": "{invalid",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("注册失败", response.get_data(as_text=True))
        self.assertIn("模型管理", response.get_data(as_text=True))

    def test_load_model_redirects_to_admin_models(self) -> None:
        provider = FakeBackendProvider()
        app = create_app(provider=provider)
        client = app.test_client()

        response = client.post(
            "/actions/models/load",
            data={"model_id": "amazon-chronos-2"},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/models?message=", response.headers["Location"])
        self.assertEqual(provider.last_load_model_id, "amazon-chronos-2")

    def test_run_task_invalid_runtime_parameter_value_redirects_to_admin_tasks(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.post(
            "/actions/run",
            data={
                "model_id": "amazon-chronos-2",
                "batch_id": "batch-1",
                "runtime_param__integer__batch_size": "invalid",
                "evaluation_metrics": "mse,smape",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("任务失败", body)
        self.assertIn("任务管理", body)

    def test_create_dataset_routes_generate_payload_from_unified_form_to_datasets_page(self) -> None:
        provider = FakeBackendProvider()
        app = create_app(provider=provider)
        client = app.test_client()

        response = client.post(
            "/actions/datasets/create",
            data={
                "source_mode": "generate",
                "track": "noise_robustness",
                "sample_count": "4",
                "context_length": "64",
                "horizon": "16",
                "seed": "13",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/datasets?message=", response.headers["Location"])
        self.assertEqual(
            provider.last_generate_payload,
            {
                "track": "noise_robustness",
                "sample_count": 4,
                "context_length": 64,
                "horizon": 16,
                "seed": 13,
            },
        )
        self.assertIsNone(provider.last_csv_payload)

    def test_create_dataset_routes_csv_payload_from_unified_form_to_datasets_page(self) -> None:
        provider = FakeBackendProvider()
        app = create_app(provider=provider)
        client = app.test_client()

        response = client.post(
            "/actions/datasets/create",
            data={
                "source_mode": "csv",
                "csv_path": "/tmp/demo.csv",
                "track": "forecast_accuracy",
                "context_length": "8",
                "horizon": "4",
                "max_samples": "1",
                "batch_id_prefix": "csv",
                "sample_id_column": "sample_id",
                "step_column": "step",
                "target_column": "target",
                "covariate_columns": "calendar_signal,load_signal",
                "delimiter": ",",
                "processors_json": "",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/datasets?message=", response.headers["Location"])
        self.assertEqual(
            provider.last_csv_payload,
            {
                "source_type": "csv",
                "csv_path": "/tmp/demo.csv",
                "track": "forecast_accuracy",
                "context_length": 8,
                "horizon": 4,
                "max_samples": 1,
                "batch_id_prefix": "csv",
                "sample_id_column": "sample_id",
                "step_column": "step",
                "target_column": "target",
                "covariate_columns": ["calendar_signal", "load_signal"],
                "delimiter": ",",
                "processors": [],
            },
        )
        self.assertIsNone(provider.last_generate_payload)

    def test_create_dataset_prefers_track_variant_id_when_provided(self) -> None:
        provider = FakeBackendProvider()
        app = create_app(provider=provider)
        client = app.test_client()

        response = client.post(
            "/actions/datasets/create",
            data={
                "source_mode": "generate",
                "track_variant_id": "univariate_forecast.noisy",
                "sample_count": "2",
                "context_length": "24",
                "horizon": "8",
                "seed": "7",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            provider.last_generate_payload,
            {
                "track_variant_id": "univariate_forecast.noisy",
                "sample_count": 2,
                "context_length": 24,
                "horizon": 8,
                "seed": 7,
            },
        )

    def test_run_task_uses_multi_select_evaluation_metrics_and_redirects_to_tasks_page(self) -> None:
        provider = FakeBackendProvider()
        app = create_app(provider=provider)
        client = app.test_client()

        response = client.post(
            "/actions/run",
            data={
                "model_id": "amazon-chronos-2",
                "batch_id": "batch-1",
                "runtime_param__integer__batch_size": "1",
                "runtime_param__boolean__use_covariates": "true",
                "evaluation_metrics": ["mse", "smape", "composite_score"],
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/tasks?message=", response.headers["Location"])
        self.assertEqual(
            provider.last_run_payload,
            {
                "model_id": "amazon-chronos-2",
                "batch_id": "batch-1",
                "model_runtime_parameters": {"batch_size": 1, "use_covariates": True},
                "evaluation_metrics": ["mse", "smape", "composite_score"],
            },
        )

    def test_run_task_includes_execution_repeat_count_when_provided(self) -> None:
        provider = FakeBackendProvider()
        app = create_app(provider=provider)
        client = app.test_client()

        response = client.post(
            "/actions/run",
            data={
                "model_id": "amazon-chronos-2",
                "batch_id": "batch-1",
                "execution_repeat_count": "5",
                "evaluation_metrics": ["mse"],
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            provider.last_run_payload,
            {
                "model_id": "amazon-chronos-2",
                "batch_id": "batch-1",
                "model_runtime_parameters": {},
                "evaluation_metrics": ["mse"],
                "execution_repeat_count": 5,
            },
        )

    def test_task_query_can_render_lookup_result(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.get("/admin/tasks?task_id=task-lookup")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("task-lookup", body)
        self.assertIn("report-task-lookup", body)
        self.assertIn("runs=2", body)
        self.assertIn("composite=98.1", body)

    def test_task_query_post_redirects_back_to_tasks_page(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.post(
            "/actions/tasks/query",
            data={"task_id": "task-lookup"},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin/tasks?task_id=task-lookup", response.headers["Location"])

    def test_task_query_failure_renders_error_message(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.get("/admin/tasks?task_id=missing-task")

        self.assertEqual(response.status_code, 200)
        self.assertIn("查询失败：task missing", response.get_data(as_text=True))
