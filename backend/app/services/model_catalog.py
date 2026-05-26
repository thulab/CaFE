from sqlmodel import Session, select

from app.core.config import Settings
from app.core.errors import ApiError
from app.core.time import utc_now
from app.models.model_registry import Model
from app.services.timer_rest_adapter import TimerRestAdapter, TimerServiceError


def list_models_for_backend(session: Session, settings: Settings) -> list[dict]:
    """Return the authoritative model catalog for the active adapter mode."""
    if settings.model_adapter != "rest":
        return [_local_model_item(model) for model in session.exec(select(Model).order_by(Model.created_at)).all()]

    adapter = TimerRestAdapter(
        base_url=settings.timer_service_base_url,
        api_prefix=settings.timer_service_api_prefix,
    )
    service_models = _list_service_models(adapter)
    local_by_remote_id = sync_timer_service_models(session, service_models)
    items: list[dict] = []
    for service_model in service_models:
        remote_id = str(service_model.get("model_id") or "").strip()
        if remote_id in local_by_remote_id:
            items.append(_service_model_item(service_model, local_by_remote_id[remote_id]))
    return items


def ensure_catalog_models_exist(session: Session, settings: Settings, model_ids: list[str]) -> None:
    """Ensure selected model IDs are present locally, syncing from timer service in REST mode."""
    if settings.model_adapter == "rest":
        adapter = TimerRestAdapter(
            base_url=settings.timer_service_base_url,
            api_prefix=settings.timer_service_api_prefix,
        )
        service_models = _list_service_models(adapter)
        sync_timer_service_models(session, service_models)
        service_ids = {str(model.get("model_id")) for model in service_models if model.get("model_id")}
        missing = [model_id for model_id in model_ids if model_id not in service_ids]
    else:
        missing = [model_id for model_id in model_ids if session.get(Model, model_id) is None]
    if missing:
        raise ApiError("model_not_found", "model not found", {"model_ids": missing}, status_code=404)


def load_model_for_backend(session: Session, settings: Settings, model_id: str) -> dict:
    """Load one model through timer-rest-service and return the refreshed catalog item."""
    if settings.model_adapter != "rest":
        model = session.get(Model, model_id)
        if model is None:
            raise ApiError("model_not_found", "model not found", {"model_ids": [model_id]}, status_code=404)
        return _local_model_item(model)

    adapter = TimerRestAdapter(
        base_url=settings.timer_service_base_url,
        api_prefix=settings.timer_service_api_prefix,
    )
    service_models = _list_service_models(adapter)
    service_ids = {str(model.get("model_id")) for model in service_models if model.get("model_id")}
    if model_id not in service_ids:
        raise ApiError("model_not_found", "model not found", {"model_ids": [model_id]}, status_code=404)
    sync_timer_service_models(session, service_models)

    try:
        adapter.ensure_model_loaded(
            {"model_id": model_id, "remote_model_id": model_id},
            timeout_seconds=settings.timer_service_model_load_timeout_seconds,
        )
    except TimerServiceError as exc:
        raise ApiError(
            "model_load_failed",
            "failed to load model from timer-rest-service",
            {"model_id": model_id, "error": str(exc)},
            status_code=503,
        ) from exc

    refreshed = _list_service_models(adapter)
    local_by_remote_id = sync_timer_service_models(session, refreshed)
    service_model = next((item for item in refreshed if item.get("model_id") == model_id), None)
    if service_model is None or model_id not in local_by_remote_id:
        raise ApiError("model_not_found", "model not found after load", {"model_ids": [model_id]}, status_code=404)
    return _service_model_item(service_model, local_by_remote_id[model_id])


def sync_timer_service_models(session: Session, service_models: list[dict]) -> dict[str, Model]:
    """Mirror timer-rest-service models into the local Model table using remote model_id as key."""
    now = utc_now()
    local_by_remote_id: dict[str, Model] = {}
    for service_model in service_models:
        remote_id = str(service_model.get("model_id") or "").strip()
        if not remote_id:
            continue
        model = session.get(Model, remote_id)
        if model is None:
            model = Model(
                model_id=remote_id,
                name=remote_id,
                model_family=str(service_model.get("model_type") or service_model.get("category") or "timer_service"),
                model_version=str(service_model.get("base_model_id") or ""),
                endpoint_uri=f"timer://{remote_id}",
            )
        model.name = remote_id
        model.model_family = str(service_model.get("model_type") or service_model.get("category") or model.model_family)
        model.model_version = str(service_model.get("base_model_id") or "")
        model.adapter_type = "timer_service"
        model.endpoint_uri = f"timer://{remote_id}"
        model.status = str(service_model.get("state") or "unknown")
        model.updated_at = now
        session.add(model)
        local_by_remote_id[remote_id] = model
    session.commit()
    for model in local_by_remote_id.values():
        session.refresh(model)
    return local_by_remote_id


def _list_service_models(adapter: TimerRestAdapter) -> list[dict]:
    try:
        return adapter.list_models()
    except TimerServiceError as exc:
        raise ApiError(
            "model_service_unavailable",
            "failed to read models from timer-rest-service",
            {"error": str(exc)},
            status_code=503,
        ) from exc


def _local_model_item(model: Model) -> dict:
    item = model.model_dump()
    item["loaded"] = None
    return item


def _service_model_item(service_model: dict, local_model: Model) -> dict:
    item = local_model.model_dump()
    item.update(
        {
            "name": str(service_model["model_id"]),
            "remote_model_id": str(service_model["model_id"]),
            "model_type": service_model.get("model_type"),
            "category": service_model.get("category"),
            "service_state": service_model.get("state"),
            "loaded": bool(service_model.get("loaded")),
            "loading": bool(service_model.get("loading")),
            "base_model_id": service_model.get("base_model_id"),
        }
    )
    return item
