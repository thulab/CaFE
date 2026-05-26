from fastapi import Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.core.config import get_settings
from app.core.errors import ApiError
from app.models.model_registry import Model
from app.services.model_catalog import list_models_for_backend, load_model_for_backend

router = make_router(prefix="/models", tags=["models"])


class ModelCreate(BaseModel):
    name: str
    model_family: str
    model_version: str


@router.get("", tier="authed")
def list_models(session: Session = Depends(get_db_session)) -> dict:
    return {"items": list_models_for_backend(session, get_settings())}


@router.post("/{model_id}/load", tier="perm", perm="run.execute")
def load_model(model_id: str, session: Session = Depends(get_db_session)) -> dict:
    return load_model_for_backend(session, get_settings(), model_id)


@router.post("", tier="perm", perm="model.register")
def create_model(payload: ModelCreate, session: Session = Depends(get_db_session)) -> Model:
    existing = session.exec(select(Model).where(Model.name == payload.name)).first()
    if existing is not None:
        raise ApiError("model_name_taken", "model name already exists", {"name": payload.name})
    model = Model(**payload.model_dump())
    session.add(model)
    session.commit()
    session.refresh(model)
    return model
