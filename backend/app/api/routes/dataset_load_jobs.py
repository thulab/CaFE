from fastapi import Depends
from pydantic import BaseModel
from sqlmodel import Session

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.core.config import get_settings
from app.models.dataset import DatasetLoadJob
from app.services.dataset_load_service import DatasetLoadService

router = make_router(prefix="/dataset-load-jobs", tags=["dataset-load-jobs"])


class DatasetLoadJobCreate(BaseModel):
    dataset_manifest_id: str
    split_config: dict
    seed: int = 0


@router.post("", tier="perm", perm="dataset.write")
def create_dataset_load_job(payload: DatasetLoadJobCreate, session: Session = Depends(get_db_session)) -> DatasetLoadJob:
    return DatasetLoadService(get_settings().runtime_dir).create_load_job(
        session,
        payload.dataset_manifest_id,
        payload.split_config,
        payload.seed,
    )


@router.get("/{load_job_id}", tier="authed")
def get_dataset_load_job(load_job_id: str, session: Session = Depends(get_db_session)) -> DatasetLoadJob:
    return session.get(DatasetLoadJob, load_job_id)
