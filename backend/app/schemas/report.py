from pydantic import BaseModel


class ReportDTO(BaseModel):
    report_id: str
    benchmarking_run_id: str
    status: str
