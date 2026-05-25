from fastapi.testclient import TestClient
from sqlmodel import Session, create_engine, select

from app.db.init_db import init_db
from app.main import create_app
from app.models.benchmark import CapabilityBlock, Track
from app.models.ranking import RankingList
from app.services.track_service import create_track_with_blocks


def make_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    return Session(engine)


def test_track_contains_blocks_without_referencing_manifest_and_creates_ranking_list(tmp_path):
    with make_session(tmp_path) as session:
        block = CapabilityBlock(name="real block", block_type="real", shard_count=1)
        session.add(block)
        session.commit()

        track, ranking = create_track_with_blocks(session, "real track", [block.capability_block_id], primary_metric_id="mae")

        loaded_block = session.get(CapabilityBlock, block.capability_block_id)
        assert loaded_block.track_id == track.track_id
        assert not hasattr(track, "dataset_manifest_id")
        assert ranking.default_metric_id == "mae"
        assert session.exec(select(RankingList).where(RankingList.track_id == track.track_id)).one().ranking_list_id == ranking.ranking_list_id


def test_create_track_route_defaults_primary_metric_to_mase():
    client = TestClient(create_app())
    with Session(client.app.state.engine) as session:
        block = CapabilityBlock(name="block", block_type="real", shard_count=1)
        session.add(block)
        session.commit()
        block_id = block.capability_block_id

    response = client.post("/tracks", json={"name": "default metric track", "capability_block_ids": [block_id]})

    assert response.status_code == 200
    track_id = response.json()["track_id"]
    with Session(client.app.state.engine) as session:
        track = session.get(Track, track_id)
        assert track.primary_metric_id == "mase"
