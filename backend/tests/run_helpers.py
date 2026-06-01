from pathlib import Path

from sqlmodel import Session

from app.models.model_registry import Model
from app.services.dataset_load_service import DatasetLoadService
from app.models.dataset import DatasetManifest
from app.services.track_service import create_real_capability_block, create_track_with_blocks


def create_loaded_track_with_models(session: Session, runtime_dir: Path, model_count: int = 1):
    source = Path(__file__).parent / "fixtures" / "valid_hourly_20.csv"
    manifest = DatasetManifest(
        name="demo",
        domain="energy",
        source_uri=str(source),
        time_column="time",
    )
    session.add(manifest)
    session.commit()
    session.refresh(manifest)
    job = DatasetLoadService(runtime_dir).create_load_job(
        session,
        manifest.dataset_manifest_id,
        {"context_length": 6, "horizon": 3, "stride": 3, "target_columns": ["target"]},
    )
    block = create_real_capability_block(session, "real block", [job.output_shard_id])
    track, ranking = create_track_with_blocks(session, "real track", [block.capability_block_id], "mase")
    models = []
    for index in range(model_count):
        model = Model(name=f"Model {index}", model_family="Timer", model_version=str(index), endpoint_uri=f"stub://{index}")
        session.add(model)
        models.append(model)
    session.commit()
    for model in models:
        session.refresh(model)
    return track, ranking, models
