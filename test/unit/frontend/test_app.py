from __future__ import annotations

import unittest

from frontend.app import create_app


class FakeBackendProvider:
    def __init__(self) -> None:
        self.last_submit_payload = None
        self.last_generate_payload = None
        self.last_csv_payload = None
        self.last_run_payload = None

    def fetch_user_overview(self, metric_id: str = "mse"):
        return {
            "models": [],
            "track_leaderboards": [],
            "tracks": [],
            "overall_metric_id": metric_id,
            "overall_leaderboard": [],
        }

    def fetch_admin_overview(self, metric_id: str = "mse"):
        return {
            "tracks": [],
            "models": [],
            "batches": [],
            "recent_tasks": [],
            "leaderboard": [],
            "overall_metric_id": metric_id,
            "overall_leaderboard_strategy": "rank_sum",
        }

    def fetch_report(self, report_id: str):
        if report_id == "missing":
            raise RuntimeError("report missing")
        return {"report_id": report_id}

    def generate_batch(self, payload):
        self.last_generate_payload = payload
        return {"batch_id": "batch-1"}

    def load_csv_batch(self, payload):
        self.last_csv_payload = payload
        return {"batch_id": "csv-1"}

    def run_task(self, payload):
        self.last_run_payload = payload
        return {"task_id": "task-1"}

    def register_model(self, payload):
        return {"model_id": payload["model_id"]}

    def submit_huggingface_model(self, payload):
        self.last_submit_payload = payload
        return {"model_id": payload.get("model_id") or "hf-model"}

    def load_model(self, model_id: str):
        return {"model_id": model_id}


class FrontendAppTest(unittest.TestCase):
    def test_admin_page_formats_model_management_card(self) -> None:
        class ProviderWithModel(FakeBackendProvider):
            def fetch_admin_overview(self, metric_id: str = "mse"):
                return {
                    "tracks": [],
                    "models": [
                        {
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
                    ],
                    "batches": [],
                    "recent_tasks": [],
                    "leaderboard": [],
                    "overall_metric_id": "mse",
                    "overall_leaderboard_strategy": "rank_sum",
                }

        app = create_app(provider=ProviderWithModel())
        client = app.test_client()

        response = client.get("/admin")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("模型链接", body)
        self.assertIn("https://huggingface.co/amazon/chronos-2", body)
        self.assertIn("权重文件位置", body)
        self.assertIn("/models/chronos2", body)
        self.assertIn("运行参数", body)
        self.assertIn("请选择模型", body)
        self.assertIn("选择模型后显示该模型对应的运行参数", body)
        self.assertIn("Batch Size", body)
        self.assertIn("integer", body)
        self.assertIn("Use Covariates", body)
        self.assertIn("boolean", body)
        self.assertIn("必填", body)
        self.assertIn('name="runtime_param__integer__batch_size"', body)
        self.assertIn('name="runtime_param__boolean__use_covariates"', body)
        self.assertNotIn("adapter=", body)
        self.assertNotIn("weights=", body)
        self.assertNotIn("repo=", body)
        self.assertNotIn("runtime params=", body)
        self.assertNotIn("runtime_parameters_json", body)

    def test_admin_page_hides_manual_model_registration_module(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.get("/admin")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("批次数据", body)
        self.assertIn("CSV 直接加载", body)
        self.assertIn("生成数据卡片", body)
        self.assertIn("CSV 加载卡片", body)
        self.assertNotIn("注册通用模型", body)

    def test_report_detail_redirects_with_error_message(self) -> None:
        app = create_app(provider=FakeBackendProvider())
        client = app.test_client()

        response = client.get("/reports/missing", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/admin?message=", response.headers["Location"])

    def test_register_model_invalid_json_redirects_to_admin(self) -> None:
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

    def test_run_task_invalid_runtime_parameter_value_redirects_to_admin(self) -> None:
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
        self.assertIn("任务失败", response.get_data(as_text=True))

    def test_create_dataset_routes_generate_payload_from_unified_form(self) -> None:
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

    def test_create_dataset_routes_csv_payload_from_unified_form(self) -> None:
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

    def test_run_task_uses_multi_select_evaluation_metrics(self) -> None:
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

    def test_submit_model_only_sends_source_fields(self) -> None:
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
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
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
