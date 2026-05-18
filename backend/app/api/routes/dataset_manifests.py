import csv
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel
from sqlmodel import Session

from app.api.deps import get_db_session
from app.core.config import get_settings
from app.core.ids import new_id
from app.core.time import utc_now
from app.models.dataset import DatasetManifest
from app.services.csv_dataset_reader import CsvDatasetReader

router = APIRouter(prefix="/dataset-manifests", tags=["dataset-manifests"])


class DatasetManifestCreate(BaseModel):
    name: str
    domain: str
    source_uri: str
    file_format: str = "csv"
    time_column: str
    target_columns: list[str]
    frequency: str | None = None
    timezone: str | None = None


@router.post("/upload")
async def upload_dataset_manifest(file: UploadFile = File(...)) -> dict:
    settings = get_settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    upload_id = new_id()
    target = settings.uploads_dir / f"{upload_id}-{Path(file.filename or 'upload.csv').name}"
    content = await file.read()
    target.write_bytes(content)

    reader = CsvDatasetReader()
    text, encoding = reader._read_text(target)
    delimiter = reader._detect_delimiter(text)
    rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
    columns = [name.strip().removeprefix("\ufeff") for name in rows[0]] if rows else []
    preview_rows = [dict(zip(columns, row, strict=False)) for row in rows[1:6]]
    return {
        "upload_id": upload_id,
        "source_uri": str(target),
        "filename": file.filename,
        "file_size": len(content),
        "detected_delimiter": delimiter,
        "encoding": encoding,
        "columns": [{"name": column, "inferred_type": "string", "nullable": False, "sample_values": []} for column in columns],
        "preview_rows": preview_rows,
        "row_count_estimate": max(len(rows) - 1, 0),
        "validation_summary": {"has_header": bool(columns), "duplicate_columns": len(set(columns)) != len(columns), "parse_warnings": []},
        "created_at": utc_now().isoformat(),
    }


@router.post("")
def create_dataset_manifest(payload: DatasetManifestCreate, session: Session = Depends(get_db_session)) -> DatasetManifest:
    manifest = DatasetManifest(**payload.model_dump())
    session.add(manifest)
    session.commit()
    session.refresh(manifest)
    return manifest


@router.get("/{dataset_manifest_id}")
def get_dataset_manifest(dataset_manifest_id: str, session: Session = Depends(get_db_session)) -> DatasetManifest:
    return session.get(DatasetManifest, dataset_manifest_id)
