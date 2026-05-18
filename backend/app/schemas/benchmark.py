from pydantic import BaseModel


class RealDatasetTrackCreateDTO(BaseModel):
    name: str
    shard_ids: list[str]
    primary_metric_id: str = "mse"
