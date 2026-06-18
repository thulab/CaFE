import json

import pytest
from sqlmodel import Session, create_engine
from sqlmodel import select

from app.db.init_db import init_db
from app.models.benchmark import BenchmarkingRun, CapabilityBlock, ForecastArtifact, Task, Track, Unit
from app.models.metric import MetricResult
from app.models.model_registry import Model
from app.models.report import Report
from app.models.sample import SampleIndex
from app.services.report_service import generate_run_report, read_report
from app.services.run_executor import create_benchmarking_run, execute_run
from tests.run_helpers import create_loaded_track_with_models


def test_succeeded_run_generates_summary_report_json(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        report = generate_run_report(session, run.benchmarking_run_id, tmp_path / "runtime")

        payload = json.loads((tmp_path / "runtime" / "reports" / f"{run.benchmarking_run_id}.json").read_text())
        assert report.status == "ready"
        assert payload["benchmarking_run_id"] == run.benchmarking_run_id
        assert payload["model_metrics"][0]["model_id"] == models[0].model_id
        assert payload["task_summaries"][0]["status"] == "succeeded"
        assert payload["sample_forecast_links"]


def test_report_task_counts_use_forecast_artifacts_when_task_counter_is_stale(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        artifacts = session.exec(
            select(ForecastArtifact).where(ForecastArtifact.benchmarking_run_id == run.benchmarking_run_id)
        ).all()
        rows = []
        for artifact in artifacts:
            with open(artifact.storage_uri, encoding="utf-8") as file:
                rows.extend(json.loads(line) for line in file)
        expected_processed = len(rows)
        expected_failed = len([row for row in rows if row["status"] == "failed"])
        assert expected_processed > 1

        task = session.exec(select(Task).where(Task.benchmarking_run_id == run.benchmarking_run_id)).one()
        task.processed_sample_count = 1
        task.failed_sample_count = 1
        session.add(task)
        session.commit()

        generate_run_report(session, run.benchmarking_run_id, tmp_path / "runtime")

        payload = json.loads((tmp_path / "runtime" / "reports" / f"{run.benchmarking_run_id}.json").read_text())
        assert payload["task_summaries"][0]["processed_sample_count"] == expected_processed
        assert payload["task_summaries"][0]["failed_sample_count"] == expected_failed
        assert payload["capability_metrics"][0]["processed_sample_count"] == expected_processed
        assert payload["capability_metrics"][0]["failed_sample_count"] == expected_failed


def test_report_includes_capability_blocks_and_per_model_capability_metrics(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track = Track(name="mixed track", track_type="mixed_dataset")
        model = Model(name="Timer profile", model_family="Timer", model_version="profile", endpoint_uri="stub://profile")
        session.add(track)
        session.add(model)
        session.commit()
        session.refresh(track)
        session.refresh(model)

        real_block = CapabilityBlock(
            track_id=track.track_id,
            name="real validation",
            block_type="real",
            capability_type="real_data",
            shard_count=1,
            sample_count=5,
        )
        trend_block = CapabilityBlock(
            track_id=track.track_id,
            name="synthetic trend",
            block_type="synthetic",
            capability_type="trend",
            shard_count=2,
            sample_count=7,
            generation_config={
                "schema_version": "capability_block_generation.v1",
                "capability_id": "trend",
                "shard_generation_configs": [{"capability_label": "Trend"}],
            },
        )
        session.add(real_block)
        session.add(trend_block)
        session.commit()
        session.refresh(real_block)
        session.refresh(trend_block)

        run = BenchmarkingRun(
            track_id=track.track_id,
            model_ids=[model.model_id],
            status="succeeded",
            model_count=1,
            task_count=2,
            sample_count=12,
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        unit = Unit(
            benchmarking_run_id=run.benchmarking_run_id,
            model_id=model.model_id,
            status="succeeded",
            task_count=2,
            sample_count=12,
        )
        session.add(unit)
        session.commit()
        session.refresh(unit)

        real_task = Task(
            benchmarking_run_id=run.benchmarking_run_id,
            unit_id=unit.unit_id,
            model_id=model.model_id,
            capability_block_id=real_block.capability_block_id,
            status="succeeded",
            sample_count=5,
            processed_sample_count=5,
        )
        trend_task = Task(
            benchmarking_run_id=run.benchmarking_run_id,
            unit_id=unit.unit_id,
            model_id=model.model_id,
            capability_block_id=trend_block.capability_block_id,
            status="succeeded",
            sample_count=7,
            processed_sample_count=7,
        )
        session.add(real_task)
        session.add(trend_task)
        session.commit()
        session.refresh(real_task)
        session.refresh(trend_task)

        session.add(
            MetricResult(
                metric_id="mase",
                result_level="task",
                benchmarking_run_id=run.benchmarking_run_id,
                unit_id=unit.unit_id,
                task_id=real_task.task_id,
                model_id=model.model_id,
                capability_block_id=real_block.capability_block_id,
                value=0.9,
            )
        )
        session.add(
            MetricResult(
                metric_id="mase",
                result_level="task",
                benchmarking_run_id=run.benchmarking_run_id,
                unit_id=unit.unit_id,
                task_id=trend_task.task_id,
                model_id=model.model_id,
                capability_block_id=trend_block.capability_block_id,
                value=0.4,
            )
        )
        session.commit()

        generate_run_report(session, run.benchmarking_run_id, tmp_path / "runtime")

        payload = json.loads((tmp_path / "runtime" / "reports" / f"{run.benchmarking_run_id}.json").read_text())
        assert {block["capability_type"] for block in payload["capability_blocks"]} == {"real_data", "trend"}
        trend_summary = next(block for block in payload["capability_blocks"] if block["capability_type"] == "trend")
        assert trend_summary["capability_label"] == "Trend"
        assert trend_summary["sample_count"] == 7

        profile = {
            item["capability_block_id"]: item
            for item in payload["capability_metrics"]
        }
        assert profile[real_block.capability_block_id]["metrics"]["mase"] == 0.9
        assert profile[trend_block.capability_block_id]["metrics"]["mase"] == 0.4
        assert profile[trend_block.capability_block_id]["model_name"] == "Timer profile"


def test_report_sample_links_are_unique_and_readable_across_models(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=2)
        run = create_benchmarking_run(session, track.track_id, [model.model_id for model in models])
        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        payload = json.loads((tmp_path / "runtime" / "reports" / f"{run.benchmarking_run_id}.json").read_text())
        links = payload["sample_forecast_links"]
        sample_count = len(session.exec(select(SampleIndex)).all())

        assert len(links) == sample_count
        assert len({link["sample_id"] for link in links}) == sample_count
        assert all(link["model_count"] == 2 for link in links)
        assert all("sample_index" in link for link in links)
        assert all("horizon_start" in link and "horizon_end" in link for link in links)
        assert all("forecast_start_at" in link and "forecast_end_at" in link for link in links)


def test_report_sample_links_do_not_materialize_full_sample_values(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        def fail_read_by_ref(self, session, storage_ref):  # noqa: ANN001, ANN202
            raise AssertionError("report generation should not materialize full sample values for links")

        monkeypatch.setattr("app.services.report_service.SampleStore.read_by_ref", fail_read_by_ref)

        report = generate_run_report(session, run.benchmarking_run_id, tmp_path / "runtime")
        payload = read_report(report, sample_link_limit=1)

        link = payload["sample_forecast_links"][0]
        assert link["history_start_at"]
        assert link["history_end_at"]
        assert link["forecast_start_at"]
        assert link["forecast_end_at"]


def test_read_report_paginates_sample_forecast_links(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, tmp_path / "runtime", model_count=2)
        run = create_benchmarking_run(session, track.track_id, [model.model_id for model in models])
        execute_run(session, run.benchmarking_run_id, tmp_path / "runtime")

        report = generate_run_report(session, run.benchmarking_run_id, tmp_path / "runtime")
        payload = read_report(report, session=session, sample_link_limit=2, sample_link_offset=1)

        assert payload["sample_forecast_links_total"] == len(session.exec(select(SampleIndex)).all())
        assert payload["sample_forecast_links_limit"] == 2
        assert payload["sample_forecast_links_offset"] == 1
        assert len(payload["sample_forecast_links"]) == 2
        assert payload["sample_forecast_links"][0]["sample_index"] == 1


def test_read_report_filters_sample_links_by_capability_and_sorts_metric_error(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as session:
        track = Track(name="mixed track", primary_metric_id="mse", track_type="mixed_dataset")
        session.add(track)
        session.commit()
        session.refresh(track)

        block_a = CapabilityBlock(
            track_id=track.track_id,
            name="trend group",
            block_type="synthetic",
            capability_type="trend",
            sample_count=2,
        )
        block_b = CapabilityBlock(
            track_id=track.track_id,
            name="real group",
            block_type="real",
            capability_type="real_data",
            sample_count=1,
        )
        run = BenchmarkingRun(
            track_id=track.track_id,
            status="succeeded",
            model_ids=["model-a", "model-b"],
            model_count=2,
        )
        session.add(block_a)
        session.add(block_b)
        session.add(run)
        session.commit()
        session.refresh(block_a)
        session.refresh(block_b)
        session.refresh(run)

        unit = Unit(benchmarking_run_id=run.benchmarking_run_id, model_id="model-a", status="succeeded")
        task_a = Task(
            benchmarking_run_id=run.benchmarking_run_id,
            unit_id=unit.unit_id,
            model_id="model-a",
            capability_block_id=block_a.capability_block_id,
            status="succeeded",
        )
        task_b = Task(
            benchmarking_run_id=run.benchmarking_run_id,
            unit_id=unit.unit_id,
            model_id="model-a",
            capability_block_id=block_b.capability_block_id,
            status="succeeded",
        )
        session.add(unit)
        session.add(task_a)
        session.add(task_b)
        session.add_all(
            [
                SampleIndex(sample_id="sample-low", shard_id="shard-a", sample_index=0, context_start=0, context_end=5, horizon_start=6, horizon_end=8),
                SampleIndex(sample_id="sample-high", shard_id="shard-a", sample_index=1, context_start=3, context_end=8, horizon_start=9, horizon_end=11),
                SampleIndex(sample_id="sample-other", shard_id="shard-b", sample_index=2, context_start=6, context_end=11, horizon_start=12, horizon_end=14),
            ]
        )
        session.add_all(
            [
                MetricResult(
                    metric_id="mse",
                    result_level="sample",
                    benchmarking_run_id=run.benchmarking_run_id,
                    unit_id=unit.unit_id,
                    task_id=task_a.task_id,
                    sample_id="sample-low",
                    shard_id="shard-a",
                    model_id="model-a",
                    capability_block_id=block_a.capability_block_id,
                    value=0.2,
                ),
                MetricResult(
                    metric_id="mse",
                    result_level="sample",
                    benchmarking_run_id=run.benchmarking_run_id,
                    unit_id=unit.unit_id,
                    task_id=task_a.task_id,
                    sample_id="sample-low",
                    shard_id="shard-a",
                    model_id="model-b",
                    capability_block_id=block_a.capability_block_id,
                    value=0.4,
                ),
                MetricResult(
                    metric_id="mse",
                    result_level="sample",
                    benchmarking_run_id=run.benchmarking_run_id,
                    unit_id=unit.unit_id,
                    task_id=task_a.task_id,
                    sample_id="sample-high",
                    shard_id="shard-a",
                    model_id="model-a",
                    capability_block_id=block_a.capability_block_id,
                    value=0.9,
                ),
                MetricResult(
                    metric_id="mse",
                    result_level="sample",
                    benchmarking_run_id=run.benchmarking_run_id,
                    unit_id=unit.unit_id,
                    task_id=task_b.task_id,
                    sample_id="sample-other",
                    shard_id="shard-b",
                    model_id="model-a",
                    capability_block_id=block_b.capability_block_id,
                    value=0.1,
                ),
            ]
        )
        report_path = tmp_path / "runtime" / "reports" / "manual.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps({"model_metrics": [{"model_id": "model-a", "metrics": {"mse": 0.5}}], "sample_forecast_links": []}), encoding="utf-8")
        report = Report(
            benchmarking_run_id=run.benchmarking_run_id,
            track_id=track.track_id,
            status="ready",
            storage_uri=str(report_path),
        )
        session.add(report)
        session.commit()
        session.refresh(report)

        desc = read_report(
            report,
            session=session,
            sample_link_limit=10,
            sample_link_capability_block_id=block_a.capability_block_id,
            sample_link_sort="metric_desc",
        )
        asc = read_report(
            report,
            session=session,
            sample_link_limit=1,
            sample_link_capability_block_id=block_a.capability_block_id,
            sample_link_sort="metric_asc",
        )

        assert desc["sample_forecast_links_total"] == 2
        assert [link["sample_id"] for link in desc["sample_forecast_links"]] == ["sample-high", "sample-low"]
        assert desc["sample_forecast_links"][1]["metric_value"] == pytest.approx(0.3)
        assert desc["sample_forecast_links"][1]["model_count"] == 2
        assert all(link["capability_block_id"] == block_a.capability_block_id for link in desc["sample_forecast_links"])
        assert asc["sample_forecast_links_total"] == 2
        assert [link["sample_id"] for link in asc["sample_forecast_links"]] == ["sample-low"]
