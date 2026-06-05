from fastapi import Depends
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.core.config import get_settings
from app.services.synthetic_generation_service import (
    SyntheticGenerationConfig,
    generate_synthetic_shards,
    synthetic_capability_catalog,
)

router = make_router(prefix="/synthetic", tags=["synthetic"])


class SyntheticShardGenerate(BaseModel):
    name: str = Field(default="Synthetic test cases")
    capabilities: list[str]
    context_length: int = Field(default=60, ge=8, le=2048)
    horizon: int = Field(default=16, ge=1, le=512)
    sample_count: int = Field(default=32, ge=1, le=1000)
    difficulty: int = Field(default=3, ge=1, le=5)
    season_length: int = Field(default=24, ge=4, le=512)
    target_dim: int = Field(default=3, ge=1, le=16)
    seed: int = 0
    frequency: str = "h"


@router.get("/capabilities", tier="authed")
def list_synthetic_capabilities() -> dict:
    return {"items": synthetic_capability_catalog()}


@router.post("/shards", tier="perm", perm="dataset.write")
def create_synthetic_shards(payload: SyntheticShardGenerate, session: Session = Depends(get_db_session)) -> dict:
    config = SyntheticGenerationConfig(**payload.model_dump())
    shards = generate_synthetic_shards(session, get_settings().runtime_dir, config)
    return {
        "items": [shard.model_dump() for shard in shards],
        "shard_ids": [shard.shard_id for shard in shards],
    }
