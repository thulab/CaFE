from sqlmodel import Session, select

from app.core.config import get_settings
from app.models.benchmark import BenchmarkingRun, CapabilityBlockShard, ForecastArtifact, Task, Unit
from app.models.dataset import DatasetLoadJob, DatasetManifest, Shard
from app.models.metric import MetricResult
from app.models.ranking import RankingEntry, RankingList
from app.models.report import Report
from app.models.sample import SampleIndex
from app.models.series_point import SeriesPoint
from app.services.run_executor import create_benchmarking_run, execute_run
from tests.run_helpers import create_loaded_track_with_models


def _executed_chain(app) -> dict[str, str]:
    with Session(app.state.engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, get_settings().runtime_dir, model_count=1)
        manifest = session.exec(select(DatasetManifest)).one()
        shard = session.exec(select(Shard)).one()
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        execute_run(session, run.benchmarking_run_id, get_settings().runtime_dir)
        session.refresh(run)
        return {
            "manifest_id": manifest.dataset_manifest_id,
            "shard_id": shard.shard_id,
            "track_id": track.track_id,
            "model_id": models[0].model_id,
            "run_id": run.benchmarking_run_id,
            "report_id": run.report_id or "",
        }


def _queued_chain(app) -> dict[str, str]:
    with Session(app.state.engine) as session:
        track, _ranking, models = create_loaded_track_with_models(session, get_settings().runtime_dir, model_count=1)
        run = create_benchmarking_run(session, track.track_id, [models[0].model_id])
        return {
            "track_id": track.track_id,
            "model_id": models[0].model_id,
            "run_id": run.benchmarking_run_id,
        }


def test_archived_track_is_hidden_from_list_but_detail_remains(app, client):
    ids = _executed_chain(app)

    archive = client.post(f"/tracks/{ids['track_id']}/archive")
    assert archive.status_code == 200, archive.text
    assert archive.json()["archived_at"]

    default_list = client.get("/tracks")
    assert default_list.status_code == 200
    assert default_list.json()["total"] == 0

    archived_list = client.get("/tracks", params={"include_archived": True})
    assert archived_list.status_code == 200
    assert archived_list.json()["total"] == 1
    assert archived_list.json()["items"][0]["archived_at"]

    detail = client.get(f"/tracks/{ids['track_id']}")
    assert detail.status_code == 200
    assert detail.json()["track_id"] == ids["track_id"]
    assert detail.json()["archived_at"]


def test_archived_track_cannot_start_new_run(app, client):
    ids = _executed_chain(app)
    assert client.post(f"/tracks/{ids['track_id']}/archive").status_code == 200

    response = client.post("/benchmarking-runs", json={"track_id": ids["track_id"], "model_ids": [ids["model_id"]]})

    assert response.status_code == 409
    assert response.json()["error_code"] == "resource_archived"


def test_run_archive_hides_list_but_report_still_loads(app, client):
    ids = _executed_chain(app)

    archive = client.post(f"/benchmarking-runs/{ids['run_id']}/archive")
    assert archive.status_code == 200, archive.text

    default_list = client.get("/benchmarking-runs")
    assert default_list.status_code == 200
    assert default_list.json()["total"] == 0

    archived_list = client.get("/benchmarking-runs", params={"include_archived": True})
    assert archived_list.status_code == 200
    assert archived_list.json()["total"] == 1
    assert archived_list.json()["items"][0]["archived_at"]

    report = client.get(f"/reports/{ids['report_id']}")
    assert report.status_code == 200
    assert report.json()["report_id"] == ids["report_id"]


def test_dataset_impact_reports_downstream_counts(app, client):
    ids = _executed_chain(app)

    response = client.get(f"/dataset-manifests/{ids['manifest_id']}/deletion-impact")

    assert response.status_code == 200
    affected = response.json()["affected"]
    assert affected["dataset_manifests"] == 1
    assert affected["load_jobs"] == 1
    assert affected["shards"] == 1
    assert affected["series_points"] == 20
    assert affected["sample_indices"] == 4
    assert affected["capability_blocks"] == 1
    assert affected["tracks"] == 1
    assert affected["benchmarking_runs"] == 1
    assert affected["reports"] == 1
    assert response.json()["cascade_required"] is True


def test_non_cascade_track_purge_with_runs_returns_409(app, client):
    ids = _executed_chain(app)

    response = client.delete(f"/tracks/{ids['track_id']}")

    assert response.status_code == 409
    assert response.json()["error_code"] == "purge_requires_cascade"
    assert response.json()["details"]["impact"]["affected"]["benchmarking_runs"] == 1


def test_cascade_track_purge_removes_runs_reports_and_ranking(app, client):
    ids = _executed_chain(app)

    response = client.delete(f"/tracks/{ids['track_id']}", params={"cascade": True})
    assert response.status_code == 200, response.text

    assert client.get(f"/tracks/{ids['track_id']}").status_code == 404
    assert client.get(f"/benchmarking-runs/{ids['run_id']}/progress").status_code == 404
    assert client.get(f"/reports/{ids['report_id']}").status_code == 404
    with Session(app.state.engine) as session:
        assert session.exec(select(BenchmarkingRun)).all() == []
        assert session.exec(select(Unit)).all() == []
        assert session.exec(select(Task)).all() == []
        assert session.exec(select(Report)).all() == []
        assert session.exec(select(RankingList)).all() == []
        assert session.exec(select(RankingEntry)).all() == []
        assert session.exec(select(ForecastArtifact)).all() == []
        assert session.exec(select(MetricResult)).all() == []


def test_cascade_dataset_purge_removes_downstream_track_and_data(app, client):
    ids = _executed_chain(app)

    response = client.delete(f"/dataset-manifests/{ids['manifest_id']}", params={"cascade": True})
    assert response.status_code == 200, response.text

    assert client.get(f"/dataset-manifests/{ids['manifest_id']}").status_code == 404
    assert client.get(f"/shards/{ids['shard_id']}").status_code == 404
    assert client.get(f"/tracks/{ids['track_id']}").status_code == 404
    assert client.get(f"/benchmarking-runs/{ids['run_id']}/progress").status_code == 404
    with Session(app.state.engine) as session:
        assert session.exec(select(DatasetManifest)).all() == []
        assert session.exec(select(DatasetLoadJob)).all() == []
        assert session.exec(select(Shard)).all() == []
        assert session.exec(select(SeriesPoint)).all() == []
        assert session.exec(select(SampleIndex)).all() == []
        assert session.exec(select(CapabilityBlockShard)).all() == []
        assert session.exec(select(BenchmarkingRun)).all() == []


def test_running_run_cannot_be_purged(app, client):
    ids = _queued_chain(app)

    response = client.delete(f"/benchmarking-runs/{ids['run_id']}")

    assert response.status_code == 409
    assert response.json()["error_code"] == "run_not_terminal"
