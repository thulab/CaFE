from __future__ import annotations

import os
from typing import Any, Protocol

import requests
from flask import Flask, redirect, render_template, request, url_for


class BackendProvider(Protocol):
    def fetch_user_overview(self) -> dict[str, Any]:
        ...

    def fetch_admin_overview(self) -> dict[str, Any]:
        ...

    def generate_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def run_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def submit_huggingface_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def load_model(self, model_id: str) -> dict[str, Any]:
        ...


class HttpBackendProvider:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}{path}", timeout=10)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}{path}", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def fetch_user_overview(self) -> dict[str, Any]:
        return self._get("/api/v1/overview/user")

    def fetch_admin_overview(self) -> dict[str, Any]:
        return self._get("/api/v1/overview/admin")

    def generate_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/datasets/generate", payload)

    def run_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/tasks/run", payload)

    def submit_huggingface_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/models/register/huggingface", payload)

    def load_model(self, model_id: str) -> dict[str, Any]:
        return self._post(f"/api/v1/models/{model_id}/load")


def create_app(provider: BackendProvider | None = None) -> Flask:
    app = Flask(__name__)
    app.config["BACKEND_PROVIDER"] = provider or HttpBackendProvider(
        os.environ.get("TSBENCHMARK_BACKEND_URL", "http://127.0.0.1:8000")
    )

    def backend() -> BackendProvider:
        return app.config["BACKEND_PROVIDER"]

    def redirect_with_message(target: str, message: str) -> Any:
        return redirect(url_for(target, message=message))

    def error_message(exc: Exception) -> str:
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                payload = response.json()
                detail = payload.get("detail")
                if detail:
                    return str(detail)
            except Exception:
                pass
            if getattr(response, "text", ""):
                return response.text
        return str(exc)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "frontend"}

    @app.get("/")
    def user_home():
        overview = backend().fetch_user_overview()
        return render_template(
            "user.html",
            overview=overview,
            message=request.args.get("message", ""),
            current_view="user",
        )

    @app.get("/admin")
    def admin():
        overview = backend().fetch_admin_overview()
        return render_template(
            "admin.html",
            overview=overview,
            message=request.args.get("message", ""),
            current_view="admin",
        )

    @app.get("/console")
    def console():
        return redirect(url_for("admin"))

    @app.post("/actions/models/submit")
    def submit_model():
        payload = {
            "repo_id": request.form["repo_id"],
            "name": request.form.get("name") or None,
            "model_id": request.form.get("model_id") or None,
            "manual": request.form["manual"],
            "task": request.form["task"],
            "revision": request.form.get("revision") or None,
            "trust_remote_code": request.form.get("trust_remote_code") == "on",
            "max_new_tokens": int(request.form["max_new_tokens"]),
            "do_sample": request.form.get("do_sample") == "on",
            "temperature": float(request.form["temperature"]),
            "top_p": float(request.form["top_p"]),
            "device_map": request.form.get("device_map") or None,
            "torch_dtype": request.form.get("torch_dtype") or None,
            "attn_implementation": request.form.get("attn_implementation") or None,
            "batch_size": int(request.form["batch_size"]),
            "context_length": int(request.form["context_length"]) if request.form.get("context_length") else None,
            "use_covariates": request.form.get("use_covariates") == "on",
            "cross_learning": request.form.get("cross_learning") == "on",
            "max_output_patches": int(request.form["max_output_patches"]) if request.form.get("max_output_patches") else None,
            "load_retries": int(request.form["load_retries"]),
            "load_retry_backoff_seconds": float(request.form["load_retry_backoff_seconds"]),
        }
        try:
            model = backend().submit_huggingface_model(payload)
            return redirect_with_message("user_home", f"模型 {model['model_id']} 已提交")
        except Exception as exc:
            return redirect_with_message("user_home", f"提交失败：{error_message(exc)}")

    @app.post("/actions/models/load")
    def load_model():
        try:
            model = backend().load_model(request.form["model_id"])
            return redirect_with_message("admin", f"模型 {model['model_id']} 已加载")
        except Exception as exc:
            return redirect_with_message("admin", f"加载失败：{error_message(exc)}")

    @app.post("/actions/generate")
    def generate():
        payload = {
            "track": request.form["track"],
            "sample_count": int(request.form["sample_count"]),
            "context_length": int(request.form["context_length"]),
            "horizon": int(request.form["horizon"]),
            "seed": int(request.form["seed"]),
        }
        try:
            batch = backend().generate_batch(payload)
            return redirect_with_message("admin", f"生成批次 {batch['batch_id']} 成功")
        except Exception as exc:
            return redirect_with_message("admin", f"生成失败：{error_message(exc)}")

    @app.post("/actions/run")
    def run_task():
        payload = {"model_id": request.form["model_id"], "batch_id": request.form["batch_id"]}
        try:
            task = backend().run_task(payload)
            return redirect_with_message("admin", f"评测任务 {task['task_id']} 已完成")
        except Exception as exc:
            return redirect_with_message("admin", f"任务失败：{error_message(exc)}")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8501, debug=False)
