#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from synthetic_feature_profile import (  # noqa: E402
    profile_gefcom2014_load,
    profile_input,
    profile_m5_covariate,
    profile_m5_hierarchy,
    profile_tsf_panel,
)


DEFAULT_DATA_DIR = Path("runtime/research")
DEFAULT_OUTPUT_DIR = Path("runtime/research/synthetic-v2-profile-smoke")
DEFAULT_REPORT_PATH = Path("docs/superpowers/baselines/2026-06-29-synthetic-v2-profile-smoke.md")


@dataclass(frozen=True)
class DatasetAsset:
    asset_id: str
    label: str
    url: str
    filename: str


@dataclass(frozen=True)
class ProfileSpec:
    profile_id: str
    title: str
    asset_id: str
    domain: str
    context_length: int
    horizon: int
    stride: int
    max_windows: int
    season_length: int
    target_max_multiplier: float = 1.5
    target_dim: int = 1
    target_features: tuple[str, ...] = ()
    profile_kind: str = "auto"
    max_series: int = 240
    max_groups: int = 20
    task: int = 1


ASSETS = {
    "us_births": DatasetAsset(
        asset_id="us_births",
        label="US Births Dataset",
        url="https://zenodo.org/records/4656049/files/us_births_dataset.zip?download=1",
        filename="us_births_dataset.zip",
    ),
    "m4_hourly": DatasetAsset(
        asset_id="m4_hourly",
        label="M4 Hourly Dataset",
        url="https://zenodo.org/records/4656589/files/m4_hourly_dataset.zip?download=1",
        filename="m4_hourly_dataset.zip",
    ),
    "electricity_hourly": DatasetAsset(
        asset_id="electricity_hourly",
        label="Electricity Hourly Dataset",
        url="https://zenodo.org/records/4656140/files/electricity_hourly_dataset.zip?download=1",
        filename="electricity_hourly_dataset.zip",
    ),
    "traffic_hourly": DatasetAsset(
        asset_id="traffic_hourly",
        label="Traffic Hourly Dataset",
        url="https://zenodo.org/records/4656132/files/traffic_hourly_dataset.zip?download=1",
        filename="traffic_hourly_dataset.zip",
    ),
    "m5": DatasetAsset(
        asset_id="m5",
        label="M5 Forecasting Accuracy Dataset",
        url="https://zenodo.org/records/12636070/files/m5-forecasting-accuracy.zip?download=1",
        filename="m5-forecasting-accuracy.zip",
    ),
    "gefcom2014": DatasetAsset(
        asset_id="gefcom2014",
        label="GEFCom2014 Dataset",
        url="https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=1",
        filename="GEFCom2014.zip",
    ),
}


PROFILE_SPECS = (
    ProfileSpec(
        profile_id="us_births_weekly",
        title="US Births daily, weekly seasonality",
        asset_id="us_births",
        domain="demography",
        context_length=365,
        horizon=30,
        stride=30,
        max_windows=20,
        season_length=7,
    ),
    ProfileSpec(
        profile_id="us_births_annual_diagnostic",
        title="US Births daily, annual seasonality diagnostic",
        asset_id="us_births",
        domain="demography",
        context_length=365,
        horizon=30,
        stride=30,
        max_windows=20,
        season_length=365,
    ),
    ProfileSpec(
        profile_id="m4_hourly_daily_96ctx",
        title="M4 Hourly, daily seasonality, 96 context",
        asset_id="m4_hourly",
        domain="macro",
        context_length=96,
        horizon=24,
        stride=24,
        max_windows=2000,
        season_length=24,
    ),
    ProfileSpec(
        profile_id="m4_hourly_daily_168ctx",
        title="M4 Hourly, daily seasonality, 168 context",
        asset_id="m4_hourly",
        domain="macro",
        context_length=168,
        horizon=24,
        stride=24,
        max_windows=2000,
        season_length=24,
    ),
    ProfileSpec(
        profile_id="m4_hourly_weekly",
        title="M4 Hourly, weekly seasonality",
        asset_id="m4_hourly",
        domain="macro",
        context_length=336,
        horizon=48,
        stride=48,
        max_windows=1000,
        season_length=168,
    ),
    ProfileSpec(
        profile_id="electricity_hourly_daily_168ctx",
        title="Electricity Hourly, daily seasonality, 168 context",
        asset_id="electricity_hourly",
        domain="energy",
        context_length=168,
        horizon=24,
        stride=24,
        max_windows=2000,
        season_length=24,
    ),
    ProfileSpec(
        profile_id="electricity_hourly_panel_168ctx",
        title="Electricity Hourly panel, 3-target common factors",
        asset_id="electricity_hourly",
        domain="energy",
        context_length=168,
        horizon=24,
        stride=24,
        max_windows=2000,
        season_length=24,
        target_dim=3,
        target_features=("pca_top1_explained", "avg_abs_target_corr"),
    ),
    ProfileSpec(
        profile_id="traffic_hourly_daily_168ctx",
        title="Traffic Hourly, daily seasonality, 168 context",
        asset_id="traffic_hourly",
        domain="traffic",
        context_length=168,
        horizon=24,
        stride=24,
        max_windows=2000,
        season_length=24,
    ),
    ProfileSpec(
        profile_id="traffic_hourly_panel_168ctx",
        title="Traffic Hourly panel, 3-target common factors",
        asset_id="traffic_hourly",
        domain="traffic",
        context_length=168,
        horizon=24,
        stride=24,
        max_windows=2000,
        season_length=24,
        target_dim=3,
        target_features=("pca_top1_explained", "avg_abs_target_corr"),
    ),
    ProfileSpec(
        profile_id="m5_daily_covariate_365ctx_28h",
        title="M5 daily sales with calendar, SNAP, and price covariates",
        asset_id="m5",
        domain="retail",
        context_length=365,
        horizon=28,
        stride=28,
        max_windows=2000,
        season_length=7,
        target_features=("future_abs_covariate_target_corr", "avg_abs_covariate_target_corr", "event_lift_abs"),
        profile_kind="m5_covariate",
        max_series=240,
    ),
    ProfileSpec(
        profile_id="m5_daily_hierarchy_365ctx_28h",
        title="M5 daily store-category additive hierarchy",
        asset_id="m5",
        domain="retail",
        context_length=365,
        horizon=28,
        stride=28,
        max_windows=1000,
        season_length=7,
        target_dim=3,
        target_features=("hierarchy_residual_mean_abs", "avg_abs_target_corr"),
        profile_kind="m5_hierarchy",
        max_groups=20,
    ),
    ProfileSpec(
        profile_id="gefcom2014_load_hourly_covariate_168ctx_24h",
        title="GEFCom2014 Load hourly temperature covariates",
        asset_id="gefcom2014",
        domain="energy",
        context_length=168,
        horizon=24,
        stride=24,
        max_windows=2000,
        season_length=24,
        target_features=("future_abs_covariate_target_corr", "avg_abs_covariate_target_corr"),
        profile_kind="gefcom2014_load",
        task=1,
    ),
)

SPEC_MAIN_FEATURES = (
    "trend_strength",
    "multi_period_score",
    "change_point_shift_energy",
    "nonlinear_lag1_gain",
    "burst_rate",
    "pca_top1_explained",
    "future_abs_covariate_target_corr",
    "hierarchy_residual_mean_abs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic v2 real-data profile smoke experiments.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    asset_paths = ensure_assets(args.data_dir, force=args.force_download, skip_download=args.skip_download)
    profiles: dict[str, dict[str, Any]] = {}
    for spec in PROFILE_SPECS:
        if spec.profile_kind == "m5_covariate":
            profile = profile_m5_covariate(
                asset_paths[spec.asset_id],
                context_length=spec.context_length,
                horizon=spec.horizon,
                stride=spec.stride,
                max_windows=spec.max_windows,
                max_series=spec.max_series,
                season_length=spec.season_length,
                domain=spec.domain,
                dataset_name=spec.title,
                target_features=list(spec.target_features),
                target_max_multiplier=spec.target_max_multiplier,
            )
        elif spec.profile_kind == "m5_hierarchy":
            profile = profile_m5_hierarchy(
                asset_paths[spec.asset_id],
                context_length=spec.context_length,
                horizon=spec.horizon,
                stride=spec.stride,
                max_windows=spec.max_windows,
                max_groups=spec.max_groups,
                season_length=spec.season_length,
                domain=spec.domain,
                dataset_name=spec.title,
                target_features=list(spec.target_features),
                target_max_multiplier=spec.target_max_multiplier,
            )
        elif spec.profile_kind == "gefcom2014_load":
            profile = profile_gefcom2014_load(
                asset_paths[spec.asset_id],
                context_length=spec.context_length,
                horizon=spec.horizon,
                stride=spec.stride,
                max_windows=spec.max_windows,
                season_length=spec.season_length,
                task=spec.task,
                domain=spec.domain,
                dataset_name=spec.title,
                target_features=list(spec.target_features),
                target_max_multiplier=spec.target_max_multiplier,
            )
        elif spec.target_dim > 1:
            profile = profile_tsf_panel(
                asset_paths[spec.asset_id],
                context_length=spec.context_length,
                horizon=spec.horizon,
                stride=spec.stride,
                max_windows=spec.max_windows,
                season_length=spec.season_length,
                target_dim=spec.target_dim,
                domain=spec.domain,
                dataset_name=spec.title,
                target_features=list(spec.target_features),
                target_max_multiplier=spec.target_max_multiplier,
            )
        else:
            profile = profile_input(
                asset_paths[spec.asset_id],
                input_format="auto",
                context_length=spec.context_length,
                horizon=spec.horizon,
                stride=spec.stride,
                max_windows=spec.max_windows,
                season_length=spec.season_length,
                domain=spec.domain,
                dataset_name=spec.title,
                target_max_multiplier=spec.target_max_multiplier,
            )
        profile["profile_id"] = spec.profile_id
        profile["profile_title"] = spec.title
        profile["source_url"] = ASSETS[spec.asset_id].url
        out_path = args.output_dir / f"{spec.profile_id}.json"
        out_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        profiles[spec.profile_id] = profile

    report = render_report(profiles, output_dir=args.output_dir, data_dir=args.data_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"wrote report: {args.report}")
    print(f"wrote profiles: {args.output_dir}")
    return 0


def ensure_assets(data_dir: Path, *, force: bool, skip_download: bool) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for asset_id, asset in ASSETS.items():
        path = data_dir / asset.filename
        if not path.exists() or force:
            if skip_download:
                raise FileNotFoundError(f"missing dataset asset: {path}")
            download(asset.url, path)
        paths[asset_id] = path
    return paths


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as response:
        tmp.write_bytes(response.read())
    tmp.replace(path)


def render_report(profiles: dict[str, dict[str, Any]], *, output_dir: Path, data_dir: Path) -> str:
    rows = [
        "| Profile | 窗口数 | 序列数 | target_dim | cov_dim | Trend p50/p95/cap | Seasonal p50/p95/cap | Slope p95/cap | Curvature p95/cap | Noise p95 | PCA1 p50/p95/cap | Corr p50/p95/cap | Future cov corr p50/p95/cap | Hierarchy residual p95/cap |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for spec in PROFILE_SPECS:
        profile = profiles[spec.profile_id]
        rows.append(
            "| "
            + " | ".join(
                [
                    spec.profile_id,
                    str(profile.get("window_count", 0)),
                    str(profile.get("used_series_count", "-")),
                    str(profile.get("bucket", {}).get("target_dim", 1)),
                    str(profile.get("bucket", {}).get("covariate_dim", 0)),
                    feature_with_cap(profile, "trend_strength"),
                    feature_with_cap(profile, "seasonal_strength"),
                    feature_with_cap(profile, "slope_abs"),
                    feature_with_cap(profile, "curvature_abs"),
                    metric(profile, "noise_ratio", "p95"),
                    feature_with_cap(profile, "pca_top1_explained"),
                    feature_with_cap(profile, "avg_abs_target_corr"),
                    feature_with_cap(profile, "future_abs_covariate_target_corr"),
                    f"{metric(profile, 'hierarchy_residual_mean_abs', 'p95')}/{cap(profile, 'hierarchy_residual_mean_abs')}",
                ]
            )
            + " |"
        )

    return "\n".join(
        [
            "# Synthetic v2 真实数据 Profile 烟测",
            "",
            "日期：2026-07-08",
            "",
            "## 目的",
            "",
            "本次烟测验证 synthetic v2 第一版显式 feature profiler 路径：公开真实数据能否稳定转换成特征分位数 profile，以及目标特征是否能按固定倍数得到可解释上限。",
            "",
            "## 输入",
            "",
            f"- US Births Dataset: {ASSETS['us_births'].url}",
            f"- M4 Hourly Dataset: {ASSETS['m4_hourly'].url}",
            f"- Electricity Hourly Dataset: {ASSETS['electricity_hourly'].url}",
            f"- Traffic Hourly Dataset: {ASSETS['traffic_hourly'].url}",
            f"- M5 Forecasting Accuracy Dataset: {ASSETS['m5'].url}",
            f"- GEFCom2014 Dataset: {ASSETS['gefcom2014'].url}",
            f"- 本地数据缓存：`{data_dir}`",
            f"- JSON profile 输出：`{output_dir}`",
            "- 目标特征上限规则：`p95 * 1.5`；天然有界特征额外截断到 `1.0`。",
            "",
            "## Profile 汇总",
            "",
            *rows,
            "",
            "## Spec 主特征覆盖",
            "",
            *spec_feature_rows(profiles),
            "",
            "## 观察",
            "",
            "- profiler 现在可以读取带非 UTF-8 元数据的 Monash TSF zip，并且 TSF 输入的 `max_windows` 已按全数据集统一限流。",
            "- US Births 适合作为小型日频 sanity check。周季节性有清晰信号；年季节性这里只作为诊断项，因为 `365+30` 窗口不足两个完整年周期，`seasonal_strength=0` 不代表真实数据没有年季节性。",
            "- M4 Hourly 更适合作为第一版小时级 trend 和 seasonality anchor：它有数百条序列，日季节性强，并且更长 context 能暴露更强的趋势变化。",
            "- Electricity Hourly 补充能源负荷基底，可用于校准 hourly 多目标 common-factor、日/周季节性和低秩结构。",
            "- Traffic Hourly 补充交通占用率基底，可用于校准更强的跨序列相关、lead-lag 和系统性 regime shift 结构。",
            "- M5 补充零售日频 known-future covariates 和 store-category additive hierarchy，可填补 covariate / hierarchy profile 缺口。",
            "- GEFCom2014 Load 补充小时级 load-temperature covariate response，可和现有 hourly energy profile 对齐。",
            "- cap multiplier 能避免目标特征增强无限偏离真实分布。对于 `trend_strength` / `seasonal_strength` 等天然 `[0, 1]` 特征，截断逻辑已经生效。",
            "",
            "## 决策",
            "",
            "- `m4_hourly_daily_168ctx` 作为 trend 和 multi-seasonal v2 pilot 的主小时级 anchor。",
            "- `electricity_hourly_daily_168ctx` 和 `traffic_hourly_daily_168ctx` 作为额外 hourly 单变量控制 profile。",
            "- `electricity_hourly_panel_168ctx` 和 `traffic_hourly_panel_168ctx` 作为多目标 common-factor / lead-lag profile。",
            "- `m5_daily_covariate_365ctx_28h` 与 `gefcom2014_load_hourly_covariate_168ctx_24h` 作为 known-future covariate profile。",
            "- `m5_daily_hierarchy_365ctx_28h` 作为 additive hierarchy profile。",
            "- `us_births_weekly` 保留为小型日频回归 / sanity anchor。",
            "- 第一版 pilot 使用 `target_max_multiplier=1.5`，暂不放宽，因为多个真实 profile 的 p95 已经接近有界特征上限。",
            "",
            "## 复现",
            "",
            "```bash",
            "python3 scripts/run_synthetic_v2_profile_smoke.py",
            "```",
            "",
        ]
    )


def spec_feature_rows(profiles: dict[str, dict[str, Any]]) -> list[str]:
    rows = [
        "| Feature | Profiles with p95 | Max p95 |",
        "| --- | --- | ---: |",
    ]
    for feature in SPEC_MAIN_FEATURES:
        values = []
        for spec in PROFILE_SPECS:
            p95 = profiles.get(spec.profile_id, {}).get("features", {}).get(feature, {}).get("p95")
            if p95 is not None:
                values.append((spec.profile_id, float(p95)))
        rows.append(
            "| "
            + " | ".join(
                [
                    feature,
                    ", ".join(profile_id for profile_id, _ in values) if values else "-",
                    format_number(max(value for _, value in values)) if values else "-",
                ]
            )
            + " |"
        )
    return rows


def feature_with_cap(profile: dict[str, Any], feature: str) -> str:
    return f"{metric(profile, feature, 'p50')}/{metric(profile, feature, 'p95')}/{cap(profile, feature)}"


def metric(profile: dict[str, Any], feature: str, key: str) -> str:
    value = profile.get("features", {}).get(feature, {}).get(key)
    if value is None:
        return "-"
    return format_number(float(value))


def cap(profile: dict[str, Any], feature: str) -> str:
    value = profile.get("target_feature_caps", {}).get(feature, {}).get("max_allowed")
    if value is None:
        return "-"
    return format_number(float(value))


def format_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    raise SystemExit(main())
