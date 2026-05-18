from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import get_db_session
from app.models.model_registry import Model

router = APIRouter(prefix="/models", tags=["models"])


class ModelCreate(BaseModel):
    name: str
    model_family: str
    model_version: str


@router.get("")
def list_models(session: Session = Depends(get_db_session)) -> dict:
    unique: dict[str, Model] = {}
    for model in session.exec(select(Model).order_by(Model.created_at)).all():
        unique.setdefault(model.name, model)
    return {"items": list(unique.values())}


@router.post("")
def create_model(payload: ModelCreate, session: Session = Depends(get_db_session)) -> Model:
    model = Model(**payload.model_dump())
    session.add(model)
    session.commit()
    session.refresh(model)
    return model
