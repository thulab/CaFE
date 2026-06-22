from pydantic import BaseModel


class SamplePreviewDTO(BaseModel):
    sample_id: str
    target_history: list[list[float]]
    target_future: list[list[float]]
