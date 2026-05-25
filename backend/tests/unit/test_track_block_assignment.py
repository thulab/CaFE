import pytest
from sqlmodel import Session, create_engine

from app.core.errors import ApiError
from app.db.init_db import init_db
from app.models.benchmark import CapabilityBlock
from app.services.track_service import create_track_with_blocks


def make_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    return Session(engine)


def test_create_track_rejects_block_already_assigned(tmp_path):
    with make_session(tmp_path) as session:
        block = CapabilityBlock(name="block", block_type="real", shard_count=1)
        session.add(block)
        session.commit()

        create_track_with_blocks(session, "first track", [block.capability_block_id], "mase")

        with pytest.raises(ApiError) as excinfo:
            create_track_with_blocks(session, "second track", [block.capability_block_id], "mase")

        assert excinfo.value.error_code == "capability_block_already_assigned"
        assert block.capability_block_id in excinfo.value.details["capability_block_ids"]
