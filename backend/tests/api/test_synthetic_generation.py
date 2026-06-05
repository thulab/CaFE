from sqlmodel import Session, select

from app.models.benchmark import CapabilityBlock, CapabilityBlockShard, Track
from app.models.dataset import Shard
from app.models.sample import SampleIndex


def test_synthetic_capabilities_and_generation_materialize_shards(app, client):
    catalog = client.get("/synthetic/capabilities")
    assert catalog.status_code == 200
    capability_ids = {item["capability_id"] for item in catalog.json()["items"]}
    assert {"trend", "common_factor", "covariate_response"}.issubset(capability_ids)

    response = client.post(
        "/synthetic/shards",
        json={
            "name": "synthetic smoke",
            "capabilities": ["trend", "covariate_response"],
            "context_length": 12,
            "horizon": 4,
            "sample_count": 3,
            "difficulty": 3,
            "season_length": 8,
            "target_dim": 2,
            "seed": 42,
            "frequency": "h",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["shard_ids"]) == 2

    with Session(app.state.engine) as session:
        shards = session.exec(select(Shard).where(Shard.shard_id.in_(body["shard_ids"]))).all()
        assert {shard.shard_type for shard in shards} == {"synthetic"}
        trend = next(shard for shard in shards if shard.capability_type == "trend")
        covariate = next(shard for shard in shards if shard.capability_type == "covariate_response")
        assert trend.target_dim == 1
        assert covariate.target_dim == 2
        assert covariate.covariate_columns == ["weather", "event"]
        assert covariate.generation_config["anchor_mode"] == "fixed_mock"

        sample = session.exec(select(SampleIndex).where(SampleIndex.shard_id == covariate.shard_id)).first()
        assert sample is not None
        assert sample.sample_metadata["capability_id"] == "covariate_response"

    sample_list = client.get(f"/shards/{body['shard_ids'][1]}/samples").json()
    sample_id = sample_list["items"][0]["sample_id"]
    preview = client.get(f"/samples/{sample_id}/preview").json()
    assert len(preview["target_history"]) == 12
    assert len(preview["target_future"]) == 4
    assert len(preview["target_history"][0]) == 2
    assert len(preview["history_cov"][0]) == 2
    assert len(preview["future_cov"]) == 4


def test_track_from_shards_groups_synthetic_capabilities(app, client):
    generated = client.post(
        "/synthetic/shards",
        json={
            "name": "synthetic grouped",
            "capabilities": ["trend", "common_factor"],
            "context_length": 12,
            "horizon": 4,
            "sample_count": 2,
            "difficulty": 2,
            "season_length": 8,
            "target_dim": 3,
            "seed": 7,
            "frequency": "h",
        },
    ).json()

    response = client.post(
        "/wizard/track-from-shards",
        json={"name": "synthetic track", "shard_ids": generated["shard_ids"], "primary_metric_id": "mase"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["capability_block_ids"]) == 2

    with Session(app.state.engine) as session:
        track = session.get(Track, body["track_id"])
        assert track is not None
        assert track.track_type == "synthetic_dataset"
        blocks = session.exec(select(CapabilityBlock).where(CapabilityBlock.track_id == track.track_id)).all()
        assert {block.capability_type for block in blocks} == {"trend", "common_factor"}
        links = session.exec(select(CapabilityBlockShard)).all()
        assert {link.shard_id for link in links} == set(generated["shard_ids"])
