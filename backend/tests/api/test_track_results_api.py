from sqlmodel import Session, select

from app.core.time import utc_now
from app.models.benchmark import BenchmarkingRun, CapabilityBlock, Task, Unit
from app.models.model_registry import Model
from tests.api.test_benchmarking_run_create import create_track


def _wait_for_terminal(client, run_id: str) -> dict:
    for _ in range(30):
        progress = client.get(f"/benchmarking-runs/{run_id}/progress").json()
        if progress["status"] in {"succeeded", "partial_succeeded", "failed", "cancelled"}:
            return progress
    raise AssertionError("run did not reach terminal state")


def test_track_results_summarize_latest_model_statuses_and_successful_metrics(client):
    track_id, model_a_id = create_track(client)
    with Session(client.app.state.engine) as session:
        model_b = Model(name="Model B", model_family="Timer", model_version="b", endpoint_uri="stub://b")
        model_c = Model(name="Model C", model_family="Timer", model_version="c", endpoint_uri="stub://c")
        session.add(model_b)
        session.add(model_c)
        session.commit()
        session.refresh(model_b)
        session.refresh(model_c)
        model_b_id = model_b.model_id
        model_c_id = model_c.model_id

    run = client.post("/benchmarking-runs", json={"track_id": track_id, "model_ids": [model_a_id, model_b_id]}).json()
    progress = _wait_for_terminal(client, run["benchmarking_run_id"])
    assert progress["status"] == "succeeded"

    with Session(client.app.state.engine) as session:
        block = session.exec(select(CapabilityBlock).where(CapabilityBlock.track_id == track_id)).one()
        failed_run = BenchmarkingRun(
            track_id=track_id,
            model_ids=[model_a_id],
            status="partial_succeeded",
            model_count=1,
            task_count=1,
            sample_count=block.sample_count,
            started_at=utc_now(),
            finished_at=utc_now(),
        )
        session.add(failed_run)
        session.commit()
        session.refresh(failed_run)
        failed_unit = Unit(
            benchmarking_run_id=failed_run.benchmarking_run_id,
            model_id=model_a_id,
            status="partial_succeeded",
            task_count=1,
            sample_count=block.sample_count,
            started_at=utc_now(),
            finished_at=utc_now(),
        )
        session.add(failed_unit)
        session.commit()
        session.refresh(failed_unit)
        session.add(
            Task(
                benchmarking_run_id=failed_run.benchmarking_run_id,
                unit_id=failed_unit.unit_id,
                model_id=model_a_id,
                capability_block_id=block.capability_block_id,
                status="partial_succeeded",
                shard_count=block.shard_count,
                sample_count=block.sample_count,
                processed_sample_count=block.sample_count,
                failed_sample_count=1,
                started_at=utc_now(),
                finished_at=utc_now(),
            )
        )
        session.commit()
        failed_run_id = failed_run.benchmarking_run_id

    response = client.get(f"/tracks/{track_id}/results", params={"sample_link_limit": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["track_id"] == track_id
    assert body["metric"] == "mse"
    statuses = {item["model_id"]: item for item in body["model_statuses"]}
    assert statuses[model_a_id]["evaluation_status"] == "run_failed"
    assert statuses[model_a_id]["run_id"] == failed_run_id
    assert statuses[model_a_id]["failed_sample_count"] == 1
    assert statuses[model_b_id]["evaluation_status"] == "evaluated"
    assert statuses[model_c_id]["evaluation_status"] == "not_evaluated"

    metric_models = {item["model_id"] for item in body["model_metrics"]}
    assert model_a_id in metric_models
    assert model_b_id in metric_models
    assert model_c_id not in metric_models
    assert body["capability_blocks"]
    assert {item["model_id"] for item in body["capability_metrics"]} == {model_a_id, model_b_id}

    assert body["sample_forecast_links_total"] > 2
    assert body["sample_forecast_links_limit"] == 2
    assert len(body["sample_forecast_links"]) == 2
    first_link = body["sample_forecast_links"][0]
    assert first_link["run_id"] == run["benchmarking_run_id"]
    assert first_link["metric_id"] == "mse"
    assert first_link["model_count"] == 2
    assert first_link["metric_value"] is not None
