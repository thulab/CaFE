from pydantic import BaseModel, Field


class ModelDTO(BaseModel):
    model_id: str
    name: str
    adapter_type: str
    forecast_limits: dict = Field(default_factory=dict)
