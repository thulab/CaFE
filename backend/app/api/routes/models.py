from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import get_db_session
from app.core.config import get_settings
from app.models.model_registry import Model
from app.services.model_adapter import remote_model_id, service_loaded_model_ids

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

    # 用 timer-rest-service 的 /models/list 标注每个模型的实时加载状态；
    # 服务不可达或处于桩模式时降级为 None（未知）。
    loaded_ids = service_loaded_model_ids(get_settings())
    items = []
    for model in unique.values():
        item = model.model_dump()
        item["loaded"] = None if loaded_ids is None else (remote_model_id(model) in loaded_ids)
        items.append(item)
    return {"items": items}


@router.post("")
def create_model(payload: ModelCreate, session: Session = Depends(get_db_session)) -> Model:
    model = Model(**payload.model_dump())
    session.add(model)
    session.commit()
    session.refresh(model)
    return model
