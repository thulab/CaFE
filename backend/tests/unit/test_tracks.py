from sqlmodel import Session, create_engine, select

from app.db.init_db import init_db
from app.models.benchmark import CapabilityBlock
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
