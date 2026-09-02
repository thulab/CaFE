#!/usr/bin/env python3
"""Build the complete, public-only CaFE reviewer artifact ZIP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ARCHIVE_ROOT = "cafe-reviewer-artifact-v1"
PRIVATE_MODEL_TOKENS = ("Timer-4.0", "Timer-4_0")
MODEL_ID_FIELDS = {
    "model_id",
    "left_model_id",
    "right_model_id",
    "benchmark_model_id",
}
TEXT_SUFFIXES = {".json", ".csv", ".md", ".txt", ".yaml", ".yml", ".toml"}
STORED_SUFFIXES = {
    ".arrow",
    ".feather",
    ".gz",
    ".npz",
    ".parquet",
    ".pdf",
    ".png",
    ".safetensors",
    ".zip",
}


@dataclass(frozen=True)
class Source:
    path: Path
    archive_path: str
    source_class: str


DROP = object()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contains_private_model(value: str) -> bool:
    return any(token in value for token in PRIVATE_MODEL_TOKENS)


def sanitize_string(value: str) -> str:
    if not any(
        marker in value
        for marker in (
            *PRIVATE_MODEL_TOKENS,
            "/data/xmy",
            "192.168.99.",
            "timecho",
        )
    ):
        return value
    replacements = (
        ("/data/xmy/CaFE", "${CAFE_ROOT}"),
        ("/data/xmy/chronos-forecasting", "${CHRONOS_ROOT}"),
        ("/data/xmy/timer-rest-service", "${MODEL_RUNTIME_ROOT}"),
        ("/data/xmy", "${DATA_ROOT}"),
    )
    for source, replacement in replacements:
        value = value.replace(source, replacement)
    value = re.sub(r"https?://192\.168\.99\.\d+(?::\d+)?", "${WORKER_ENDPOINT}", value)
    value = re.sub(r"\b192\.168\.99\.\d+\b", "${WORKER_IP}", value)
    value = re.sub(r"\btimecho\d+\b", "${WORKER_HOST}", value)
    for token in PRIVATE_MODEL_TOKENS:
        value = value.replace(token, "${OMITTED_UNRELEASED_MODEL}")
    return value


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        for field in MODEL_ID_FIELDS:
            raw = value.get(field)
            if isinstance(raw, str) and contains_private_model(raw):
                return DROP
        output: dict[str, Any] = {}
        for key, item in value.items():
            if contains_private_model(str(key)):
                continue
            projected = sanitize_json_value(item)
            if projected is not DROP:
                output[sanitize_string(str(key))] = projected
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            if isinstance(item, str) and contains_private_model(item):
                continue
            projected = sanitize_json_value(item)
            if projected is not DROP:
                output.append(projected)
        return output
    if isinstance(value, str):
        return sanitize_string(value)
    return value


def needs_text_projection(data: bytes) -> bool:
    return any(
        marker in data
        for marker in (
            b"Timer-4.0",
            b"Timer-4_0",
            b"/data/xmy",
            b"192.168.99.",
            b"timecho",
        )
    )


def project_json(source: Path) -> tuple[bytes, str]:
    raw = source.read_bytes()
    original_hash = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw.decode("utf-8"))
    projected = sanitize_json_value(value)
    if projected is DROP:
        projected = {}
    if isinstance(projected, dict):
        projected["_reviewer_projection"] = {
            "public_models_only": True,
            "source_sha256": original_hash,
            "note": (
                "Host-specific values and the unreleased internal model were "
                "removed from this reviewer-facing projection."
            ),
        }
    return (
        json.dumps(projected, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n",
        original_hash,
    )


def project_csv(source: Path) -> tuple[bytes, str]:
    raw = source.read_bytes()
    original_hash = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for row in reader:
        if any(contains_private_model(cell) for cell in row):
            continue
        writer.writerow([sanitize_string(cell) for cell in row])
    return output.getvalue().encode("utf-8"), original_hash


def project_text(source: Path) -> tuple[bytes, str]:
    raw = source.read_bytes()
    original_hash = hashlib.sha256(raw).hexdigest()
    lines = raw.decode("utf-8").splitlines()
    projected = [sanitize_string(line) for line in lines if not contains_private_model(line)]
    return ("\n".join(projected) + "\n").encode("utf-8"), original_hash


def parquet_projection_columns(source: Path) -> tuple[list[str], list[str]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(source)
    string_columns = [
        field.name
        for field in schema
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    ]
    filter_columns = [
        name
        for name in string_columns
        if name in MODEL_ID_FIELDS
        or name in {"model", "models"}
        or "model_id" in name
        or name.endswith("_model")
    ]
    text_columns = [
        name
        for name in string_columns
        if name.endswith("json") or "path" in name.lower()
    ]
    return filter_columns, text_columns


def sanitize_embedded_json(value: str) -> tuple[str, bool]:
    if sanitize_string(value) == value:
        return value, True
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return sanitize_string(value), True
    projected = sanitize_json_value(parsed)
    if projected is DROP:
        return "{}", False
    return json.dumps(projected, sort_keys=True, separators=(",", ":")), True


def project_parquet(
    source: Path,
    destination: Path,
    filter_columns: list[str],
    text_columns: list[str],
) -> str:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    original_hash = sha256_path(source)
    parquet_file = pq.ParquetFile(source)
    schema = parquet_file.schema_arrow.remove_metadata()
    with pq.ParquetWriter(destination, schema, compression="zstd") as writer:
        for batch in parquet_file.iter_batches(batch_size=65_536):
            table = pa.Table.from_batches([batch])
            mask = pa.array([True] * table.num_rows)
            for column in filter_columns:
                values = pc.cast(table[column], pa.string())
                keep = pc.invert(
                    pc.is_in(values, value_set=pa.array(PRIVATE_MODEL_TOKENS))
                )
                mask = pc.and_(mask, pc.fill_null(keep, True))
            for column in text_columns:
                field_index = table.schema.get_field_index(column)
                field = table.schema.field(field_index)
                values = table[column].to_pylist()
                projected_values: list[str | None] = []
                embedded_keep: list[bool] = []
                for value in values:
                    if value is None:
                        projected_values.append(None)
                        embedded_keep.append(True)
                    elif column.endswith("json"):
                        projected, retain = sanitize_embedded_json(value)
                        projected_values.append(projected)
                        embedded_keep.append(retain)
                    else:
                        projected_values.append(sanitize_string(value))
                        embedded_keep.append(True)
                table = table.set_column(
                    field_index,
                    field,
                    pa.array(projected_values, type=field.type),
                )
                if not all(embedded_keep):
                    mask = pc.and_(mask, pa.array(embedded_keep))
            writer.write_table(table.filter(mask))
    return original_hash


def should_skip(path: Path) -> bool:
    if path.name.endswith(".log") or path.name in {"__pycache__", ".DS_Store"}:
        return True
    return any(contains_private_model(part) for part in path.parts)


def iter_source_files(source: Source) -> Iterable[tuple[Path, str]]:
    if source.path.is_file():
        yield source.path, source.archive_path
        return
    for root, directory_names, file_names in os.walk(source.path):
        root_path = Path(root)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not should_skip(root_path / name)
            and not (root_path / name).is_symlink()
        )
        for name in sorted(file_names):
            path = root_path / name
            if should_skip(path) or path.is_symlink():
                continue
            relative = path.relative_to(source.path).as_posix()
            yield path, f"{source.archive_path}/{relative}"


def source_specs(cafe_root: Path, chronos_root: Path) -> list[Source]:
    experiment_root = cafe_root / "runtime" / "experiments"
    sources = [
        Source(cafe_root / "data" / "gift-eval", "data/gift-eval", "benchmark_source"),
        Source(
            cafe_root / "data" / "fev-mini-v0.8.0",
            "data/fev-mini-v0.8.0",
            "benchmark_source",
        ),
    ]
    main_ids = (
        "gift-v15-short-qualified-feasible-seed2026082701",
        "gift-v15-medium-qualified-feasible-seed2026082701",
        "gift-v15-long-qualified-feasible-seed2026082701",
        "fev-mini20-full-v6",
    )
    for experiment_id in main_ids:
        sources.append(
            Source(
                experiment_root / experiment_id,
                f"experiments/main/{experiment_id}",
                "main_experiment",
            )
        )
    for seed in range(2026082701, 2026082711):
        experiment_id = f"gift-v15-short-stability10-head78ef32f-seed{seed}"
        sources.append(
            Source(
                experiment_root / experiment_id,
                f"experiments/stability/{experiment_id}",
                "stability_experiment",
            )
        )
    stability_summary = (
        cafe_root
        / "runtime"
        / "orchestration"
        / "short_stability10_inference_3node_78ef32f_20260831"
        / "stability"
    )
    sources.append(Source(stability_summary, "stability-summary", "stability_summary"))
    for experiment_id in (
        "gift-v15-short-qualified-feasible-moirai16k-seed2026082701-r1",
        "gift-v15-short-qualified-feasible-moirai16k-seed2026082702-r1",
    ):
        sources.append(
            Source(
                experiment_root / experiment_id,
                f"experiments/finetuning-contracts/{experiment_id}",
                "finetuning_contract",
            )
        )
    for experiment_id in (
        "gift-v15-seed2026082701-target-only-v1",
        "fev-mini20-full-v6-target-only-v1",
    ):
        sources.append(
            Source(
                cafe_root / "runtime" / "ablation_trials" / experiment_id,
                f"experiments/ablation/{experiment_id}",
                "ablation_summary",
            )
        )
    fine_root = chronos_root / "chronos-2-finetuned"
    fine_specs = (
        (
            "cafe-v15-qf-moirai16k-window10-replacement-40k",
            "default",
        ),
        (
            "cafe-v15-qf-moirai16k-window10-nrmse-replacement-40k",
            "effect-nrmse",
        ),
    )
    for directory, label in fine_specs:
        for child in ("models", "results"):
            sources.append(
                Source(
                    fine_root / directory / child,
                    f"finetuning/{label}/{child}",
                    "finetuning_output",
                )
            )
    return sources


class ArchiveWriter:
    def __init__(self, archive: zipfile.ZipFile) -> None:
        self.archive = archive
        self.records: list[dict[str, Any]] = []

    def _info(self, archive_path: str, *, stored: bool) -> zipfile.ZipInfo:
        info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{archive_path}")
        info.date_time = datetime.now().timetuple()[:6]
        info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        return info

    def add_bytes(
        self,
        data: bytes,
        archive_path: str,
        *,
        source_class: str,
        projected: bool = False,
        source_sha256: str | None = None,
    ) -> None:
        info = self._info(archive_path, stored=Path(archive_path).suffix in STORED_SUFFIXES)
        with self.archive.open(info, "w", force_zip64=True) as output:
            output.write(data)
        record: dict[str, Any] = {
            "path": archive_path,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "source_class": source_class,
            "projected": projected,
        }
        if source_sha256 is not None:
            record["source_sha256"] = source_sha256
        self.records.append(record)

    def add_path(
        self,
        source: Path,
        archive_path: str,
        *,
        source_class: str,
        projected: bool = False,
        source_sha256: str | None = None,
    ) -> None:
        info = self._info(archive_path, stored=source.suffix in STORED_SUFFIXES)
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as input_handle, self.archive.open(
            info, "w", force_zip64=True
        ) as output:
            for chunk in iter(lambda: input_handle.read(8 * 1024 * 1024), b""):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        record: dict[str, Any] = {
            "path": archive_path,
            "size_bytes": size,
            "sha256": digest.hexdigest(),
            "source_class": source_class,
            "projected": projected,
        }
        if source_sha256 is not None:
            record["source_sha256"] = source_sha256
        self.records.append(record)


def code_archive(repository: Path, revision: str, destination: Path) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", f"{revision}^{{commit}}"],
        cwd=repository,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(destination), commit],
        cwd=repository,
        check=True,
    )
    return commit


def artifact_readme(commit: str) -> bytes:
    text = f"""# CaFE complete reviewer artifact

This archive accompanies CaFE source commit `{commit}`. Extract
`code/CaFE-research-{commit[:12]}.tar.gz`, then follow
`reproducibility/README.md` for verification, figure reconstruction, and
end-to-end commands.

Run `sha256sum -c SHA256SUMS` from this directory before use. The checksums
cover every payload file and `MANIFEST.json`. The manifest records files that
were projected from immutable internal runs to remove an unreleased model and
deployment-specific paths. Dense fine-tuning caches and the public Chronos-2
base checkpoint are regenerated/downloaded by the included scripts.
"""
    return text.encode("utf-8")


def build(args: argparse.Namespace) -> None:
    sources = source_specs(args.cafe_root, args.chronos_root)
    missing = [source.path for source in sources if not source.path.exists()]
    if missing:
        raise FileNotFoundError("missing artifact sources:\n" + "\n".join(map(str, missing)))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for source in sources:
            print(f"{source.source_class:24s} {source.path} -> {source.archive_path}")
        return

    with tempfile.TemporaryDirectory(prefix="cafe-reviewer-artifact-") as temporary:
        temporary_root = Path(temporary)
        code_tar = temporary_root / "code.tar.gz"
        commit = code_archive(args.cafe_root, args.code_revision, code_tar)
        with zipfile.ZipFile(
            args.output,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            writer = ArchiveWriter(archive)
            writer.add_bytes(
                artifact_readme(commit),
                "README.md",
                source_class="artifact_metadata",
            )
            writer.add_path(
                code_tar,
                f"code/CaFE-research-{commit[:12]}.tar.gz",
                source_class="source_snapshot",
            )
            file_count = 0
            for source_spec in sources:
                for source, archive_path in iter_source_files(source_spec):
                    file_count += 1
                    if file_count % 250 == 0:
                        print(f"archived {file_count} files", flush=True)
                    suffix = source.suffix.lower()
                    if suffix == ".parquet":
                        filter_columns, text_columns = parquet_projection_columns(source)
                        if filter_columns or text_columns:
                            projected_path = temporary_root / "projected.parquet"
                            original_hash = project_parquet(
                                source,
                                projected_path,
                                filter_columns,
                                text_columns,
                            )
                            writer.add_path(
                                projected_path,
                                archive_path,
                                source_class=source_spec.source_class,
                                projected=True,
                                source_sha256=original_hash,
                            )
                            projected_path.unlink()
                            continue
                    if suffix in TEXT_SUFFIXES:
                        raw = source.read_bytes()
                        if needs_text_projection(raw):
                            if suffix == ".json":
                                data, original_hash = project_json(source)
                            elif suffix == ".csv":
                                data, original_hash = project_csv(source)
                            else:
                                data, original_hash = project_text(source)
                            writer.add_bytes(
                                data,
                                archive_path,
                                source_class=source_spec.source_class,
                                projected=True,
                                source_sha256=original_hash,
                            )
                            continue
                    writer.add_path(
                        source,
                        archive_path,
                        source_class=source_spec.source_class,
                    )

            writer.records.sort(key=lambda record: record["path"])
            manifest = {
                "schema_version": "cafe.reviewer_artifact.v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "cafe_commit": commit,
                "archive_root": ARCHIVE_ROOT,
                "public_models_only": True,
                "files": writer.records,
            }
            manifest_data = (
                json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            )
            writer.add_bytes(
                manifest_data,
                "MANIFEST.json",
                source_class="artifact_metadata",
            )
            checksum_records = sorted(writer.records, key=lambda record: record["path"])
            checksums = "".join(
                f"{record['sha256']}  {record['path']}\n" for record in checksum_records
            ).encode("utf-8")
            writer.add_bytes(
                checksums,
                "SHA256SUMS",
                source_class="artifact_metadata",
            )
    zip_hash = sha256_path(args.output)
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar.write_text(f"{zip_hash}  {args.output.name}\n", encoding="utf-8")
    print(f"created {args.output}")
    print(f"sha256 {zip_hash}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cafe-root", type=Path, required=True)
    parser.add_argument("--chronos-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", default="HEAD")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.cafe_root = args.cafe_root.resolve()
    args.chronos_root = args.chronos_root.resolve()
    args.output = args.output.resolve()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
