from pydantic import BaseModel


class DatasetLoadJobCreateDTO(BaseModel):
    dataset_manifest_id: str
    split_config: dict
    seed: int = 0
