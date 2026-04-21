from __future__ import annotations

from pathlib import Path

import pandas as pd

from .validation import validate_benchmark, validate_external_alignment
from .utils import adjacent_meta_path, read_json, write_json, write_parquet


def _collect_eval_files(eval_dir: Path) -> pd.DataFrame:
    files = sorted(eval_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no eval parquet files found in {eval_dir}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _write_markdown_summary(output_dir: Path, overall: pd.DataFrame, eval_frame: pd.DataFrame, validation: dict[str, object]) -> None:
    runtime_totals = eval_frame.groupby("model")["runtime_ms"].sum().div(1000.0)
    lines = [
        "# TSBenchmark v1 Report",
        "",
        f"- Evaluated series: {int(validation['n_series'])}",
        f"- Anchor mode: {validation.get('anchor_mode', 'unknown')}",
        f"- Median non-target drift: {float(validation['median_non_target_drift']):.4f}",
        "",
        "## Overall",
        "",
        "| Model | MASE | sMAPE | Relative Skill | Avg Runtime | Total Runtime |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall.sort_values("mase", na_position="last").itertuples(index=False):
        lines.append(
            f"| {row.model} | {float(row.mase):.4f} | {float(row.smape):.4f} | {float(row.relative_skill):.4f} | "
            f"{float(row.runtime_ms):.2f} ms | {float(runtime_totals.get(row.model, 0.0)):.1f}s |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_report_artifacts(
    benchmark_path: Path,
    eval_dir: Path,
    output_dir: Path,
    real_eval_path: Path | None,
) -> Path:
    benchmark = pd.read_parquet(benchmark_path)
    meta_path = adjacent_meta_path(benchmark_path)
    benchmark_meta = read_json(meta_path) if meta_path.exists() else {}
    eval_frame = _collect_eval_files(eval_dir)
    benchmark_version = str(benchmark_meta.get("benchmark_version", benchmark.get("benchmark_version", pd.Series(["unknown"])).iloc[0]))
    if "benchmark_version" in eval_frame.columns:
        eval_versions = {str(value) for value in eval_frame["benchmark_version"].dropna().unique().tolist()}
        if eval_versions and eval_versions != {benchmark_version}:
            raise ValueError(f"eval dir {eval_dir} contains benchmark versions {sorted(eval_versions)}, expected {benchmark_version}")
    if "baseline_mase" not in benchmark.columns:
        raise ValueError(f"benchmark {benchmark_path} is missing cached baseline_mase")
    merged = eval_frame.merge(benchmark, left_on="series_id", right_on="id", how="left")
    merged["relative_skill"] = 1.0 - merged["mase"] / (merged["baseline_mase"] + 1e-8)

    overall = merged.groupby("model")[["mase", "smape", "runtime_ms", "relative_skill"]].mean().reset_index()
    by_track = merged.groupby(["model", "track"])[["mase", "smape", "runtime_ms", "relative_skill"]].mean().reset_index()
    by_family_difficulty = merged.groupby(["model", "family", "difficulty"])[["mase", "smape", "relative_skill"]].mean().reset_index()
    by_horizon = merged.groupby(["model", "horizon_ratio"])[["mase", "smape", "relative_skill"]].mean().reset_index()

    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(overall, output_dir / "overall.parquet")
    write_parquet(by_track, output_dir / "by_track.parquet")
    write_parquet(by_family_difficulty, output_dir / "by_family_difficulty.parquet")
    write_parquet(by_horizon, output_dir / "by_horizon_ratio.parquet")

    validation = benchmark_meta.get("validation_summary") or validate_benchmark(benchmark_path)
    external_alignment = validate_external_alignment(eval_frame=eval_frame, benchmark_frame=benchmark, real_eval_path=real_eval_path)
    summary = {
        "benchmark_version": benchmark_version,
        "overall": overall.to_dict(orient="records"),
        "validation": validation,
        "external_alignment": external_alignment,
        "known_limitations": benchmark_meta.get("known_limitations", []),
    }
    write_json(summary, output_dir / "summary.json")
    _write_markdown_summary(output_dir, overall, eval_frame, validation)
    return output_dir
