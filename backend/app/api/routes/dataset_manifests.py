import csv
from pathlib import Path

from fastapi import Depends, File, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import get_db_session
from app.api.router_factory import make_router
from app.core.config import get_settings
from app.core.ids import new_id
from app.core.time import utc_now
from app.models.dataset import DatasetManifest
from app.services.csv_dataset_reader import CsvDatasetReader

router = make_router(prefix="/dataset-manifests", tags=["dataset-manifests"])


class DatasetManifestCreate(BaseModel):
    name: str
    domain: str
    source_uri: str
    file_format: str = "csv"
    time_column: str
    value_columns: list[str] = []
    frequency: str | None = None
    timezone: str | None = None


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _sniff_tsfile(upload_id: str, target: Path, filename: str | None, file_size: int) -> dict:
    """\u55c5\u63a2 .tsfile\uff1a\u8fd4\u56de\u8bbe\u5907 + \u7269\u7406\u91cf\u540d + \u884c\u6570\uff0c\u800c\u975e CSV \u5206\u9694\u7b26/\u7f16\u7801\u3002"""
    import tsfile

    tf = tsfile.TsFileDataFrame(str(target))
    try:
        series = list(tf.list_timeseries())
        # series \u5f62\u5982 "<table>.<device>.<measurement>"\uff1bdevice \u7ef4 = \u53bb\u6389\u6700\u540e\u4e00\u6bb5\u3002
        devices = sorted({".".join(name.split(".")[:-1]).split(".")[-1] for name in series})
        device = devices[0] if len(devices) == 1 else None
        measurements = [name.split(".")[-1] for name in series]
        row_count = 0
        if series:
            row_count = len(tf[series[0]].timestamps[:])
    finally:
        tf.__exit__(None, None, None)

    return {
        "upload_id": upload_id,
        "source_uri": str(target),
        "filename": filename,
        "file_size": file_size,
        "file_format": "tsfile",
        "device": device,
        "devices": devices,
        "columns": [
            {"name": measurement, "inferred_type": "numeric", "nullable": False, "sample_values": []}
            for measurement in measurements
        ],
        "preview_rows": [],
        "row_count_estimate": row_count,
        "validation_summary": {
            "has_header": True,
            "duplicate_columns": len(set(measurements)) != len(measurements),
            "multiple_devices": len(devices) != 1,
            "parse_warnings": [],
        },
        "created_at": utc_now().isoformat(),
    }


@router.post("/upload", tier="perm", perm="dataset.write")
async def upload_dataset_manifest(file: UploadFile = File(...)) -> dict:
    settings = get_settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    upload_id = new_id()
    filename = file.filename or "upload.csv"
    target = settings.uploads_dir / f"{upload_id}-{Path(filename).name}"
    content = await file.read()
    target.write_bytes(content)

    if Path(filename).suffix.lower() == ".tsfile":
        return _sniff_tsfile(upload_id, target, file.filename, len(content))

    reader = CsvDatasetReader()
    text, encoding = reader._read_text(target)
    delimiter = reader._detect_delimiter(text)
    rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
    first_row = [name.strip().removeprefix("\ufeff") for name in rows[0]] if rows else []
    # \u771f\u5b9e\u8868\u5934\u68c0\u6d4b\uff1a\u9996\u884c\u82e5\u53ef\u89e3\u6790\u4e3a\u6570\u636e\u884c\uff08\u65f6\u95f4\u5217 + \u6570\u503c\u5217\uff09\uff0c\u5219\u89c6\u4e3a\u65e0\u8868\u5934\u3002
    has_header = bool(first_row) and not reader._looks_like_data_row(first_row)
    columns = first_row if has_header else [f"column_{index + 1}" for index in range(len(first_row))]
    data_rows = rows[1:] if has_header else rows
    preview_rows = [dict(zip(columns, row, strict=False)) for row in data_rows[:20]]

    # \u9010\u5217\u63a8\u65ad numeric / string\uff1a\u53d6\u9996\u4e2a\u6570\u636e\u884c\u7684\u975e\u7a7a\u503c\uff1b\u7a7a\u7f3a\u5219\u56de\u9000 string\u3002
    column_types: list[str] = []
    for index in range(len(columns)):
        sample = next(
            (row[index].strip() for row in data_rows if index < len(row) and row[index].strip()),
            "",
        )
        column_types.append("numeric" if sample and _looks_numeric(sample) else "string")

    return {
        "upload_id": upload_id,
        "source_uri": str(target),
        "filename": file.filename,
        "file_size": len(content),
        "file_format": "csv",
        "detected_delimiter": delimiter,
        "encoding": encoding,
        "columns": [
            {"name": column, "inferred_type": column_types[index], "nullable": False, "sample_values": []}
            for index, column in enumerate(columns)
        ],
        "preview_rows": preview_rows,
        "row_count_estimate": len(data_rows),
        "validation_summary": {
            "has_header": has_header,
            "duplicate_columns": len(set(columns)) != len(columns),
            "parse_warnings": [],
        },
        "created_at": utc_now().isoformat(),
    }


@router.post("", tier="perm", perm="dataset.write")
def create_dataset_manifest(payload: DatasetManifestCreate, session: Session = Depends(get_db_session)) -> DatasetManifest:
    manifest = DatasetManifest(**payload.model_dump())
    session.add(manifest)
    session.commit()
    session.refresh(manifest)
    return manifest


@router.get("", tier="authed")
def list_dataset_manifests(
    limit: int = 50, offset: int = 0, session: Session = Depends(get_db_session)
) -> dict:
    total = len(session.exec(select(DatasetManifest)).all())
    items = session.exec(
        select(DatasetManifest)
        .order_by(DatasetManifest.created_at.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/{dataset_manifest_id}", tier="authed")
def get_dataset_manifest(dataset_manifest_id: str, session: Session = Depends(get_db_session)) -> DatasetManifest:
    return session.get(DatasetManifest, dataset_manifest_id)
