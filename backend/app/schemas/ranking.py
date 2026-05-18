from pydantic import BaseModel


class RankingRowDTO(BaseModel):
    model_id: str
    metric_value: float
    rank: int
