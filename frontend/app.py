from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

import requests
from flask import Flask, redirect, render_template, request, url_for

from backend.app.config import AppSettings, get_settings
from backend.app.task_management.domain import DEFAULT_EVALUATION_METRICS


class BackendProvider(Protocol):
    def fetch_user_overview(self, metric_id: str = "mse") -> dict[str, Any]:
        ...

    def fetch_admin_overview(self, metric_id: str = "mse") -> dict[str, Any]:
        ...

    def fetch_report(self, report_id: str) -> dict[str, Any]:
        ...

    def generate_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def load_csv_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def run_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def register_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def submit_huggingface_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def load_model(self, model_id: str) -> dict[str, Any]:
        ...


class HttpBackendProvider:
    def __init__(self, base_url: str, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self.settings.service.frontend.get_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.settings.service.frontend.post_timeout_seconds)
        response.raise_for_status()
        return response.json()

    def fetch_user_overview(self, metric_id: str = "mse") -> dict[str, Any]:
        return self._get("/api/v1/overview/user", params={"metric_id": metric_id})

    def fetch_admin_overview(self, metric_id: str = "mse") -> dict[str, Any]:
        return self._get("/api/v1/overview/admin", params={"metric_id": metric_id})

    def fetch_report(self, report_id: str) -> dict[str, Any]:
        return self._get(f"/api/v1/reports/{report_id}")

    def generate_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/datasets/generate", payload)

    def load_csv_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/datasets/load/csv", payload)

    def run_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/tasks/run", payload)

    def register_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/models/register", payload)

    def submit_huggingface_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/api/v1/models/register/huggingface", payload)

    def load_model(self, model_id: str) -> dict[str, Any]:
        return self._post(f"/api/v1/models/{model_id}/load")


def create_app(provider: BackendProvider | None = None, settings: AppSettings | None = None) -> Flask:
    app_settings = settings or get_settings()
    app = Flask(__name__)
    backend_url = os.environ.get("TSBENCHMARK_BACKEND_URL", app_settings.frontend_backend_base_url())
    app.config["BACKEND_PROVIDER"] = provider or HttpBackendProvider(backend_url, settings=app_settings)
    app.config["APP_SETTINGS"] = app_settings

    def backend() -> BackendProvider:
        return app.config["BACKEND_PROVIDER"]

    def settings() -> AppSettings:
        return app.config["APP_SETTINGS"]

    def redirect_with_message(target: str, message: str) -> Any:
        params: dict[str, Any] = {"message": message}
        current_metric_id = request.values.get("current_metric_id")
        if current_metric_id:
            params["metric_id"] = current_metric_id
        return redirect(url_for(target, **params))

    def parse_csv_list(raw_value: str) -> list[str]:
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    def parse_json_field(raw_value: str, *, default: Any, field_name: str) -> Any:
        if not raw_value.strip():
            return default
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} 不是合法 JSON: {exc}") from exc

    def parse_runtime_parameter_value(*, value_type: str, raw_value: str, field_name: str) -> Any:
        value = raw_value.strip()
        if value_type == "string":
            return value
        if value_type == "integer":
            try:
                return int(value)
            except ValueError as exc:
                raise ValueError(f"{field_name} 需要整数") from exc
        if value_type == "float":
            try:
                return float(value)
            except ValueError as exc:
                raise ValueError(f"{field_name} 需要浮点数") from exc
        if value_type == "boolean":
            normalized = value.lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
            raise ValueError(f"{field_name} 需要布尔值 true/false")
        raise ValueError(f"{field_name} 使用了不支持的参数类型 {value_type}")

    def parse_runtime_parameters() -> dict[str, Any]:
        runtime_parameters: dict[str, Any] = {}
        for key in request.form.keys():
            if not key.startswith("runtime_param__"):
                continue
            _, value_type, parameter_name = key.split("__", 2)
            raw_value = request.form.get(key, "")
            if not raw_value.strip():
                continue
            runtime_parameters[parameter_name] = parse_runtime_parameter_value(
                value_type=value_type,
                raw_value=raw_value,
                field_name=parameter_name,
            )
        return runtime_parameters

    def build_generate_payload() -> dict[str, Any]:
        payload = {
            "sample_count": int(request.form["sample_count"]),
            "context_length": int(request.form["context_length"]),
            "horizon": int(request.form["horizon"]),
            "seed": int(request.form["seed"]),
        }
        track_variant_id = request.form.get("track_variant_id", "").strip()
        if track_variant_id:
            payload["track_variant_id"] = track_variant_id
        else:
            payload["track"] = request.form["track"]
        return payload

    def build_csv_payload() -> dict[str, Any]:
        processors = parse_json_field(
            request.form.get("processors_json", ""),
            default=[],
            field_name="processors_json",
        )
        payload = {
            "source_type": "csv",
            "csv_path": request.form["csv_path"],
            "context_length": int(request.form["context_length"]),
            "horizon": int(request.form["horizon"]),
            "max_samples": int(request.form["max_samples"]) if request.form.get("max_samples") else None,
            "batch_id_prefix": request.form.get("batch_id_prefix") or "csv",
            "sample_id_column": request.form.get("sample_id_column") or "sample_id",
            "step_column": request.form.get("step_column") or "step",
            "target_column": request.form.get("target_column") or "target",
            "covariate_columns": parse_csv_list(request.form.get("covariate_columns", "")),
            "delimiter": request.form.get("delimiter") or ",",
            "processors": processors,
        }
        track_variant_id = request.form.get("track_variant_id", "").strip()
        if track_variant_id:
            payload["track_variant_id"] = track_variant_id
        else:
            payload["track"] = request.form["track"]
        return payload

    def handle_dataset_create(source_mode: str | None = None) -> Any:
        dataset_source = source_mode or request.form.get("source_mode") or "generate"
        if dataset_source == "generate":
            batch = backend().generate_batch(build_generate_payload())
            return redirect_with_message("admin", f"生成批次 {batch['batch_id']} 成功")
        if dataset_source == "csv":
            batch = backend().load_csv_batch(build_csv_payload())
            return redirect_with_message("admin", f"CSV 批次 {batch['batch_id']} 已导入")
        raise ValueError(f"source_mode 不支持: {dataset_source}")

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
        selected_metric_id = request.args.get("metric_id", "mse")
        overview = backend().fetch_user_overview(metric_id=selected_metric_id)
        return render_template(
            "user.html",
            overview=overview,
            user_defaults=settings().ui.user_model_submission,
            leaderboard_metric_options=DEFAULT_EVALUATION_METRICS,
            selected_metric_id=selected_metric_id,
            message=request.args.get("message", ""),
            current_view="user",
        )

    @app.get("/admin")
    def admin():
        selected_metric_id = request.args.get("metric_id", "mse")
        overview = backend().fetch_admin_overview(metric_id=selected_metric_id)
        return render_template(
            "admin.html",
            overview=overview,
            admin_defaults=settings().ui.admin_batch_generation,
            runtime_generated_dir=str(Path(settings().system.runtime.root) / "generated"),
            evaluation_metric_options=DEFAULT_EVALUATION_METRICS,
            leaderboard_metric_options=DEFAULT_EVALUATION_METRICS,
            selected_metric_id=selected_metric_id,
            message=request.args.get("message", ""),
            current_view="admin",
        )

    @app.get("/reports/<report_id>")
    def report_detail(report_id: str):
        try:
            report = backend().fetch_report(report_id)
            return render_template(
                "report.html",
                report=report,
                current_view="admin",
            )
        except Exception as exc:
            return redirect_with_message("admin", f"报告加载失败：{error_message(exc)}")

    @app.get("/console")
    def console():
        return redirect(url_for("admin"))

    @app.post("/actions/models/submit")
    def submit_model():
        payload = {
            "huggingface_url": request.form["huggingface_url"],
            "name": request.form.get("name") or None,
            "model_id": request.form.get("model_id") or None,
            "manual": request.form["manual"],
            "revision": request.form.get("revision") or None,
            "trust_remote_code": request.form.get("trust_remote_code") == "on",
        }
        try:
            model = backend().submit_huggingface_model(payload)
            return redirect_with_message("user_home", f"模型 {model['model_id']} 已提交")
        except Exception as exc:
            return redirect_with_message("user_home", f"提交失败：{error_message(exc)}")

    @app.post("/actions/models/register")
    def register_model():
        try:
            metadata = parse_json_field(
                request.form.get("metadata_json", ""),
                default={},
                field_name="metadata_json",
            )
            payload = {
                "model_id": request.form["model_id"],
                "name": request.form["name"],
                "adapter": request.form["adapter"],
                "source_type": request.form.get("source_type") or "uploaded_manual",
                "manual": request.form["manual"],
                "capabilities": parse_csv_list(request.form.get("capabilities", "")),
                "metadata": metadata,
            }
            model = backend().register_model(payload)
            return redirect_with_message("admin", f"模型 {model['model_id']} 已注册")
        except Exception as exc:
            return redirect_with_message("admin", f"注册失败：{error_message(exc)}")

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

    @app.post("/actions/datasets/create")
    def create_dataset():
        source_mode = request.form.get("source_mode") or "generate"
        try:
            return handle_dataset_create(source_mode)
        except Exception as exc:
            prefix = "CSV 导入" if source_mode == "csv" else "批次创建"
            return redirect_with_message("admin", f"{prefix}失败：{error_message(exc)}")

    @app.post("/actions/datasets/load_csv")
    def load_csv_dataset():
        try:
            return handle_dataset_create("csv")
        except Exception as exc:
            return redirect_with_message("admin", f"CSV 导入失败：{error_message(exc)}")

    @app.post("/actions/run")
    def run_task():
        try:
            payload = {
                "model_id": request.form["model_id"],
                "batch_id": request.form["batch_id"],
                "model_runtime_parameters": parse_runtime_parameters(),
                "evaluation_metrics": request.form.getlist("evaluation_metrics")
                or parse_csv_list(request.form.get("evaluation_metrics", "")),
            }
            if request.form.get("execution_repeat_count", "").strip():
                payload["execution_repeat_count"] = int(request.form["execution_repeat_count"])
            task = backend().run_task(payload)
            return redirect_with_message("admin", f"评测任务 {task['task_id']} 已完成")
        except Exception as exc:
            return redirect_with_message("admin", f"任务失败：{error_message(exc)}")

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    app.run(
        host=settings.service.frontend.host,
        port=settings.service.frontend.port,
        debug=settings.service.frontend.debug,
    )
