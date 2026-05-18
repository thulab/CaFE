import json
from pathlib import Path

from sqlmodel import Session, select

from app.models.benchmark import BenchmarkingRun, ForecastArtifact, Task, Unit
from app.models.metric import MetricResult
from app.models.model_registry import Model
from app.models.report import Report


def generate_run_report(session: Session, run_id: str, runtime_dir: Path) -> Report:
    run = session.get(BenchmarkingRun, run_id)
    report_dir = Path(runtime_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / f"{run_id}.json"
    units = session.exec(select(Unit).where(Unit.benchmarking_run_id == run_id)).all()
    tasks = session.exec(select(Task).where(Task.benchmarking_run_id == run_id)).all()
    metrics = session.exec(select(MetricResult).where(MetricResult.benchmarking_run_id == run_id)).all()
    artifacts = session.exec(select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run_id)).all()

    payload = {
        "benchmarking_run_id": run_id,
        "track_id": run.track_id,
        "status": run.status,
        "model_metrics": [_unit_metrics(session, unit, metrics) for unit in units],
        "task_summaries": [_task_summary(task, metrics) for task in tasks],
        "sample_forecast_links": _sample_links(run_id, artifacts),
        "cancellation_reason": "cancel_requested" if run.status == "cancelled" else None,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    existing = session.exec(select(Report).where(Report.benchmarking_run_id == run_id)).first()
    report = existing or Report(benchmarking_run_id=run_id, track_id=run.track_id)
    report.status = "ready"
    report.storage_uri = str(output)
    report.summary = {"status": run.status, "model_count": len(units), "task_count": len(tasks)}
    run.report_id = report.report_id
    session.add(report)
    session.add(run)
    session.commit()
    session.refresh(report)
    return report


def read_report(report: Report) -> dict:
    return json.loads(Path(report.storage_uri).read_text(encoding="utf-8"))


def _unit_metrics(session: Session, unit: Unit, metrics: list[MetricResult]) -> dict:
    model = session.get(Model, unit.model_id)
    return {
        "unit_id": unit.unit_id,
        "model_id": unit.model_id,
        "model_name": model.name if model else unit.model_id,
        "status": unit.status,
        "metrics": {
            metric.metric_id: metric.value
            for metric in metrics
            if metric.result_level == "unit" and metric.unit_id == unit.unit_id
        },
    }


def _task_summary(task: Task, metrics: list[MetricResult]) -> dict:
    return {
        "task_id": task.task_id,
        "unit_id": task.unit_id,
        "model_id": task.model_id,
        "capability_block_id": task.capability_block_id,
        "status": task.status,
        "error_code": task.error_code,
        "error_message": task.error_message,
        "metrics": {
            metric.metric_id: metric.value
            for metric in metrics
            if metric.result_level == "task" and metric.task_id == task.task_id
        },
    }


def _sample_links(run_id: str, artifacts: list[ForecastArtifact]) -> list[dict]:
    links = []
    for artifact in artifacts:
        with Path(artifact.storage_uri).open(encoding="utf-8") as file:
            for line in file:
                sample_id = json.loads(line)["sample_id"]
                links.append({"sample_id": sample_id, "run_id": run_id, "forecast_artifact_id": artifact.forecast_artifact_id})
    return links
