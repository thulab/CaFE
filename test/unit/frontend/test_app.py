from __future__ import annotations

import unittest

from frontend.app import create_app


class FakeBackendProvider:
    def fetch_user_overview(self):
        return {"models": [], "track_leaderboards": [], "tracks": [], "overall_leaderboard": []}

    def fetch_admin_overview(self):
        return {
            "tracks": [],
            "models": [],
            "batches": [],
            "recent_tasks": [],
            "leaderboard": [],
            "overall_leaderboard_strategy": "rank_sum",
        }

    def fetch_report(self, report_id: str):
        if report_id == "missing":
            raise RuntimeError("report missing")
        return {"report_id": report_id}

    def generate_batch(self, payload):
        return {"batch_id": "batch-1"}

    def load_csv_batch(self, payload):
        return {"batch_id": "csv-1"}

    def run_task(self, payload):
        return {"task_id": "task-1"}

    def register_model(self, payload):
        return {"model_id": payload["model_id"]}

    def submit_huggingface_model(self, payload):
        return {"model_id": payload.get("model_id") or "hf-model"}

    def load_model(self, model_id: str):
        return {"model_id": model_id}


class FrontendAppTest(unittest.TestCase):
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
