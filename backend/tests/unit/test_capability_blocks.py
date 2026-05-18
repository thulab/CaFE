import pytest
from sqlmodel import Session, create_engine

from app.core.errors import ApiError
from app.db.init_db import init_db
from app.models.dataset import Shard
from app.services.track_service import create_real_capability_block


def make_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    return Session(engine)


def test_real_capability_block_can_contain_one_or_more_real_shards(tmp_path):
    with make_session(tmp_path) as session:
        shard_1 = Shard(dataset_manifest_id="manifest-1", source_uri="file-1.csv", status="ready")
        shard_2 = Shard(dataset_manifest_id="manifest-2", source_uri="file-2.csv", status="ready")
        session.add(shard_1)
        session.add(shard_2)
        session.commit()

        block = create_real_capability_block(session, "real block", [shard_1.shard_id, shard_2.shard_id])

        assert block.block_type == "real"
        assert block.shard_count == 2
        assert block.sample_count == 0
        assert session.get(Shard, shard_1.shard_id).capability_block_id == block.capability_block_id
        assert session.get(Shard, shard_2.shard_id).capability_block_id == block.capability_block_id


def test_shard_belongs_to_only_one_capability_block(tmp_path):
    with make_session(tmp_path) as session:
        shard = Shard(dataset_manifest_id="manifest-1", source_uri="file.csv", status="ready")
        session.add(shard)
        session.commit()

        create_real_capability_block(session, "first", [shard.shard_id])

        with pytest.raises(ApiError) as exc:
            create_real_capability_block(session, "second", [shard.shard_id])
        assert exc.value.error_code == "shard_already_assigned"
