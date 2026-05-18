from pydantic import BaseModel


class ModelDTO(BaseModel):
    model_id: str
    name: str
    adapter_type: str
