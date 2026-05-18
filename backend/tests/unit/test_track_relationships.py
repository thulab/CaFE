from app.models.benchmark import CapabilityBlock, Track
from app.models.dataset import Shard


def test_track_references_blocks_and_never_references_dataset_manifest_directly():
    track = Track(name="real track", primary_metric_id="mse")
    block = CapabilityBlock(track_id=track.track_id, name="real data", shard_count=1)
    shard = Shard(dataset_manifest_id="manifest-1", capability_block_id=block.capability_block_id, source_uri="file.csv")

    assert block.track_id == track.track_id
    assert shard.capability_block_id == block.capability_block_id
    assert not hasattr(track, "dataset_manifest_id")
