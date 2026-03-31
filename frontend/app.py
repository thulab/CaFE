from __future__ import annotations

import os
from typing import Any, Protocol

import requests
from flask import Flask, redirect, render_template, request, url_for


class BackendProvider(Protocol):
    def fetch_overview(self) -> dict[str, Any]:
        ...

    def generate_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def run_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


class HttpBackendProvider:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}{path}", timeout=10)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}{path}", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()

    def fetch_overview(self) -> dict[str, Any]:
        return self._get("/api/v1/overview")

    def generate_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/datasets/generate", payload)

    def run_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/tasks/run", payload)


def create_app(provider: BackendProvider | None = None) -> Flask:
    app = Flask(__name__)
    app.config["BACKEND_PROVIDER"] = provider or HttpBackendProvider(
        os.environ.get("TSBENCHMARK_BACKEND_URL", "http://127.0.0.1:8000")
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "frontend"}

    @app.get("/")
    def index():
        provider_impl: BackendProvider = app.config["BACKEND_PROVIDER"]
        overview = provider_impl.fetch_overview()
        return render_template(
            "index.html",
            overview=overview,
            message=request.args.get("message", ""),
            current_view="gallery",
        )

    @app.get("/console")
    def console():
        provider_impl: BackendProvider = app.config["BACKEND_PROVIDER"]
        overview = provider_impl.fetch_overview()
        return render_template(
            "console.html",
            overview=overview,
            message=request.args.get("message", ""),
            current_view="console",
        )

    @app.post("/actions/generate")
    def generate():
        provider_impl: BackendProvider = app.config["BACKEND_PROVIDER"]
        payload = {
            "track": request.form["track"],
            "sample_count": int(request.form["sample_count"]),
            "context_length": int(request.form["context_length"]),
            "horizon": int(request.form["horizon"]),
            "seed": int(request.form["seed"]),
        }
        batch = provider_impl.generate_batch(payload)
        view = request.form.get("view", "gallery")
        target = "console" if view == "console" else "index"
        return redirect(url_for(target, message=f"生成批次 {batch['batch_id']} 成功"))

    @app.post("/actions/run")
    def run_task():
        provider_impl: BackendProvider = app.config["BACKEND_PROVIDER"]
        payload = {"model_id": request.form["model_id"], "batch_id": request.form["batch_id"]}
        task = provider_impl.run_task(payload)
        view = request.form.get("view", "gallery")
        target = "console" if view == "console" else "index"
        return redirect(url_for(target, message=f"评测任务 {task['task_id']} 已完成"))

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8501, debug=False)
