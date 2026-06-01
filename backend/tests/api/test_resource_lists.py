"""列表端点测试：/dataset-manifests、/shards、/benchmarking-runs、/reports。

存在意义：前端列表页和首页卡片角标依赖这 4 个端点；recents store 删除后，没有这些端点页面会变空白。
"""

from sqlmodel import Session

from app.models.benchmark import BenchmarkingRun
from app.models.dataset import DatasetManifest, Shard
from app.models.report import Report


def _seed_manifest(session: Session, name: str) -> str:
    m = DatasetManifest(
        name=name,
        domain="energy",
        source_uri=f"/tmp/{name}.csv",
        time_column="time",
    )
    session.add(m)
    session.commit()
    return m.dataset_manifest_id


def _seed_shard(session: Session, manifest_id: str, name: str | None = None) -> str:
    s = Shard(dataset_manifest_id=manifest_id, source_uri=f"/tmp/{manifest_id}.shard", name=name)
    session.add(s)
    session.commit()
    return s.shard_id


def _seed_run(session: Session, track_id: str) -> str:
    r = BenchmarkingRun(track_id=track_id, model_ids=["m1"])
    session.add(r)
    session.commit()
    return r.benchmarking_run_id


def _seed_report(session: Session, run_id: str, track_id: str) -> str:
    rep = Report(benchmarking_run_id=run_id, track_id=track_id)
    session.add(rep)
    session.commit()
    return rep.report_id


def test_list_dataset_manifests_returns_created_at_desc_with_total(app, client):
    with Session(app.state.engine) as session:
        _seed_manifest(session, "older")
        _seed_manifest(session, "newer")

    resp = client.get("/dataset-manifests")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert body["limit"] == 50
    assert body["offset"] == 0
    assert [it["name"] for it in body["items"]] == ["newer", "older"]


def test_list_dataset_manifests_paginates(app, client):
    with Session(app.state.engine) as session:
        for i in range(3):
            _seed_manifest(session, f"m{i}")

    page1 = client.get("/dataset-manifests", params={"limit": 2, "offset": 0}).json()
    page2 = client.get("/dataset-manifests", params={"limit": 2, "offset": 2}).json()

    assert page1["total"] == 3 and len(page1["items"]) == 2
    assert page2["total"] == 3 and len(page2["items"]) == 1
    # 两页不重叠
    page1_ids = {it["dataset_manifest_id"] for it in page1["items"]}
    page2_ids = {it["dataset_manifest_id"] for it in page2["items"]}
    assert page1_ids.isdisjoint(page2_ids)


def test_list_shards_supports_dataset_manifest_id_filter(app, client):
    with Session(app.state.engine) as session:
        m1_id = _seed_manifest(session, "m1")
        m2_id = _seed_manifest(session, "m2")
        _seed_shard(session, m1_id)
        _seed_shard(session, m1_id)
        _seed_shard(session, m2_id)

    all_shards = client.get("/shards").json()
    assert all_shards["total"] == 3

    only_m1 = client.get("/shards", params={"dataset_manifest_id": m1_id}).json()
    assert only_m1["total"] == 2
    assert all(it["dataset_manifest_id"] == m1_id for it in only_m1["items"])


def test_list_shards_supports_name_search_and_dataset_name(app, client):
    with Session(app.state.engine) as session:
        energy_id = _seed_manifest(session, "hourly-energy")
        weather_id = _seed_manifest(session, "weather")
        _seed_shard(session, energy_id, "energy validation cases")
        _seed_shard(session, weather_id, "weather validation cases")

    body = client.get("/shards", params={"q": "energy"}).json()

    assert body["total"] == 1
    assert body["items"][0]["name"] == "energy validation cases"
    assert body["items"][0]["dataset_name"] == "hourly-energy"


def test_list_runs_supports_track_id_filter(app, client):
    with Session(app.state.engine) as session:
        _seed_run(session, "track-a")
        _seed_run(session, "track-a")
        _seed_run(session, "track-b")

    all_runs = client.get("/benchmarking-runs").json()
    assert all_runs["total"] == 3

    only_a = client.get("/benchmarking-runs", params={"track_id": "track-a"}).json()
    assert only_a["total"] == 2
    assert all(it["track_id"] == "track-a" for it in only_a["items"])


def test_list_reports_returns_items_with_total(app, client):
    with Session(app.state.engine) as session:
        _seed_report(session, "run-1", "track-1")
        _seed_report(session, "run-2", "track-1")

    body = client.get("/reports").json()
    assert body["total"] == 2
    assert len(body["items"]) == 2
    assert {it["benchmarking_run_id"] for it in body["items"]} == {"run-1", "run-2"}


def test_list_endpoints_require_auth(anon_client):
    # 匿名应被路由层 401（这 4 个端点都是 tier="authed"）
    for path in ("/dataset-manifests", "/shards", "/benchmarking-runs", "/reports"):
        assert anon_client.get(path).status_code == 401, path
