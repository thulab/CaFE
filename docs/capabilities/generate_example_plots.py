#!/usr/bin/env python3
"""Generate legacy paired mechanism illustrations for the HTML atlas.

These plots use the committed pre-real-bounded online artifact and are not
Paper v4 qualification evidence. For each capability the script uses one
registered online profile and ranks a fixed visualization-only seed bank. A representative seed must pass all
online gates at intensities 1, 3, and 5, have a strictly increasing primary
feature, and minimize normalized distance to the selected dataset/profile's
three legacy local q10/q50/q90 targets.
The default pool has 64 seeds; the high-variance nonlinear statistic uses a
predeclared 256-seed pool. Reusing the seed keeps all nuisance draws and
structural clocks paired.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
    }
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.synthetic_feature_gate import (  # noqa: E402
    evaluate_feature_support_gate,
)
from app.services.synthetic_generation_service import (  # noqa: E402
    _generate_sample_values,
    _normalize_covariates,
    _realized_features,
    _regime_clock_history_incremental_r2,
    _standardize_by_context,
    _standardize_hierarchy_by_context,
)
from app.services.synthetic_generator_conditioning import (  # noqa: E402
    resolve_generator_conditioning,
)
from app.services.synthetic_near_distance_gate import (  # noqa: E402
    evaluate_near_distance_gate,
)


ARTIFACT_DIR = REPO_ROOT / "backend/app/data"
GENERATOR_ARTIFACT_PATH = (
    ARTIFACT_DIR / "synthetic_v2_generator_conditioning_artifact.json"
)
FEATURE_GATE_ARTIFACT_PATH = (
    ARTIFACT_DIR / "synthetic_v2_feature_gate_artifact.json"
)
NEAR_GATE_ARTIFACT_PATH = (
    ARTIFACT_DIR / "synthetic_v2_near_distance_artifact.json"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "assets/examples"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

INTENSITIES = (1, 3, 5)
SEARCH_SEED = 2026071805
REPRESENTATIVE_SEED_CANDIDATES = 64
REPRESENTATIVE_SEED_CANDIDATES_BY_CAPABILITY = {
    "nonlinear_persistence": 256,
}

PRIMARY_FEATURE = {
    "trend": "trend_strength",
    "multi_seasonal": "multi_period_score",
    "time_varying_seasonality": "seasonal_amplitude_modulation",
    "regime_switching": "regime_clock_history_incremental_r2",
    "nonlinear_persistence": "nonlinear_conditional_gain",
    "predictable_intermittency": "spike_rate",
    "common_factor": "pca_top1_explained",
    "hierarchical_coherence": "hierarchy_child_heterogeneity",
    "covariate_response": "covariate_incremental_r2",
}


@dataclass(frozen=True)
class CapabilityPlotSpec:
    capability_id: str
    title_zh: str
    profile_id: str
    context_length: int
    horizon: int
    target_dim: int
    season_length: int
    hierarchy: bool = False


CAPABILITIES = (
    CapabilityPlotSpec(
        "trend",
        "趋势外推",
        "m4_hourly_daily_168ctx",
        168,
        24,
        1,
        24,
    ),
    CapabilityPlotSpec(
        "multi_seasonal",
        "多重季节性",
        "m4_hourly_daily_168ctx",
        168,
        24,
        1,
        24,
    ),
    CapabilityPlotSpec(
        "time_varying_seasonality",
        "时变季节性",
        "electricity_hourly_daily_168ctx",
        168,
        24,
        1,
        24,
    ),
    CapabilityPlotSpec(
        "regime_switching",
        "状态切换",
        "m4_hourly_daily_168ctx",
        168,
        24,
        1,
        24,
    ),
    CapabilityPlotSpec(
        "nonlinear_persistence",
        "非线性持续性",
        "traffic_hourly_daily_168ctx",
        168,
        24,
        1,
        24,
    ),
    CapabilityPlotSpec(
        "predictable_intermittency",
        "可预测间歇性",
        "electricity_hourly_daily_168ctx",
        168,
        24,
        1,
        24,
    ),
    CapabilityPlotSpec(
        "common_factor",
        "公共因子",
        "traffic_hourly_panel_168ctx",
        168,
        24,
        3,
        24,
    ),
    CapabilityPlotSpec(
        "hierarchical_coherence",
        "层级一致性",
        "m5_daily_hierarchy_365ctx_28h",
        365,
        28,
        3,
        7,
        hierarchy=True,
    ),
    CapabilityPlotSpec(
        "covariate_response",
        "协变量响应",
        "gefcom2014_load_hourly_covariate_168ctx_24h",
        168,
        24,
        1,
        24,
    ),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_seed(capability_id: str, candidate_index: int) -> int:
    payload = f"{SEARCH_SEED}:{capability_id}:{candidate_index}".encode()
    return int(hashlib.blake2s(payload, digest_size=8).hexdigest(), 16) % (
        2**32 - 1
    )


def representative_seed_bank_size(capability_id: str) -> int:
    return int(
        REPRESENTATIVE_SEED_CANDIDATES_BY_CAPABILITY.get(
            capability_id,
            REPRESENTATIVE_SEED_CANDIDATES,
        )
    )


def make_view(
    target: np.ndarray,
    covariates: np.ndarray | None,
    *,
    context_length: int,
    hierarchy: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    target_view = np.asarray(target, dtype=float)
    if hierarchy:
        target_view = _standardize_hierarchy_by_context(
            target_view,
            context_length,
        )
    else:
        target_view = _standardize_by_context(target_view, context_length)
    covariate_view = None
    if covariates is not None:
        covariate_view = _normalize_covariates(
            np.asarray(covariates, dtype=float),
            context_length,
        )
    return target_view, covariate_view


def qualify_candidate(
    spec: CapabilityPlotSpec,
    seed: int,
    *,
    generator_artifact: dict[str, Any],
    feature_artifact: dict[str, Any],
    near_artifact: dict[str, Any],
) -> dict[int, dict[str, Any]] | None:
    conditioning = resolve_generator_conditioning(
        capability_id=spec.capability_id,
        profile_id=spec.profile_id,
        context_length=spec.context_length,
        horizon=spec.horizon,
        target_dim=spec.target_dim,
        artifact=generator_artifact,
    )
    if conditioning is None:
        raise RuntimeError(
            f"missing conditioning for {spec.profile_id}/"
            f"{spec.capability_id}"
        )

    result: dict[int, dict[str, Any]] = {}
    for intensity in INTENSITIES:
        target, metadata, covariates = _generate_sample_values(
            spec.capability_id,
            spec.context_length + spec.horizon,
            spec.context_length,
            spec.target_dim,
            spec.season_length,
            intensity,
            np.random.default_rng(seed),
            generator_conditioning=conditioning,
        )
        if not bool(
            metadata.get("predictability", {}).get(
                "construction_validated",
                False,
            )
        ):
            return None

        plot_target, plot_covariates = make_view(
            target,
            covariates,
            context_length=spec.context_length,
            hierarchy=spec.hierarchy,
        )
        plot_features = _realized_features(
            plot_target,
            plot_covariates,
            spec.season_length,
            spec.context_length,
        )
        if spec.capability_id == "regime_switching":
            plot_features["regime_clock_history_incremental_r2"] = (
                _regime_clock_history_incremental_r2(
                    plot_target,
                    context_length=spec.context_length,
                    season_length=spec.season_length,
                    cut_points=metadata["cut_points"],
                    dwell_length=int(metadata["dwell_length"]),
                )
            )
        feature_gate = evaluate_feature_support_gate(
            capability_id=spec.capability_id,
            features=plot_features,
            profile_ids=(spec.profile_id,),
            context_length=spec.context_length,
            horizon=spec.horizon,
            target_dim=spec.target_dim,
            artifact=feature_artifact,
        )
        near_gate = evaluate_near_distance_gate(
            target=plot_target,
            features=plot_features,
            profile_ids=(spec.profile_id,),
            context_length=spec.context_length,
            horizon=spec.horizon,
            artifact=near_artifact,
        )
        if not bool(
            feature_gate.get("enforced")
            and feature_gate.get("accepted")
            and near_gate.get("enforced")
            and near_gate.get("accepted")
        ):
            return None
        result[intensity] = {
            "target": plot_target,
            "covariates": plot_covariates,
            "features": plot_features,
            "metadata": metadata,
            "gate_views": [
                {
                    "context_length": spec.context_length,
                    "profile_id": spec.profile_id,
                    "feature_gate_status": feature_gate.get("status"),
                    "near_distance_status": near_gate.get("status"),
                    "feature_gate_normalized_score": feature_gate.get(
                        "normalized_score"
                    ),
                    "strict_risk": near_gate.get("strict_risk"),
                    "combined_risk": near_gate.get("combined_risk"),
                }
            ],
        }
    return result


def find_paired_examples(
    spec: CapabilityPlotSpec,
    *,
    generator_artifact: dict[str, Any],
    feature_artifact: dict[str, Any],
    near_artifact: dict[str, Any],
) -> tuple[int, dict[int, dict[str, Any]], int, float]:
    anchors = np.asarray(
        generator_artifact["profiles"][spec.profile_id]["capabilities"][
            spec.capability_id
        ]["target_values"],
        dtype=float,
    )[[intensity - 1 for intensity in INTENSITIES]]
    anchor_range = max(float(anchors[-1] - anchors[0]), 1e-9)
    candidates: list[
        tuple[float, int, int, dict[int, dict[str, Any]]]
    ] = []
    seed_bank_size = representative_seed_bank_size(spec.capability_id)
    for candidate_index in range(seed_bank_size):
        seed = paired_seed(spec.capability_id, candidate_index)
        result = qualify_candidate(
            spec,
            seed,
            generator_artifact=generator_artifact,
            feature_artifact=feature_artifact,
            near_artifact=near_artifact,
        )
        if result is None:
            continue
        primary = PRIMARY_FEATURE[spec.capability_id]
        realized = np.asarray(
            [
                result[intensity]["features"][primary]
                for intensity in INTENSITIES
            ],
            dtype=float,
        )
        if not bool(np.all(np.diff(realized) > 0.0)):
            continue
        score = float(np.mean(np.abs(realized - anchors)) / anchor_range)
        candidates.append(
            (score, candidate_index + 1, seed, result)
        )
    if candidates:
        score, candidate_number, seed, result = min(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        return seed, result, candidate_number, score
    raise RuntimeError(
        f"no paired qualified seed for {spec.capability_id} after "
        f"{seed_bank_size} visualization candidates"
    )


def style_axis(
    axis: plt.Axes,
    *,
    horizon: int,
    with_xlabel: bool = True,
) -> None:
    axis.axvspan(0, horizon - 1, color="#fef3c7", alpha=0.70, zorder=0)
    axis.axvline(0, color="#b45309", linewidth=1.2, linestyle="--", alpha=0.9)
    axis.grid(axis="y", color="#cbd5e1", linewidth=0.65, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#94a3b8")
    axis.tick_params(colors="#475569", labelsize=8)
    axis.set_ylabel("标准化值", color="#334155", fontsize=9)
    if with_xlabel:
        axis.set_xlabel("相对预测起点的时间步", color="#334155", fontsize=9)


def nonlinear_partial_relationship(
    target: np.ndarray,
    season_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(target, dtype=float)[:, 0]
    seasonal_lag = max(4, int(season_length))
    nonlinear_lag = max(2, seasonal_lag // 2)
    start = max(1, seasonal_lag, nonlinear_lag)
    response = values[start:]
    linear_design = np.column_stack(
        [
            np.ones(len(response)),
            values[start - 1 : -1],
            values[
                start - seasonal_lag : len(values) - seasonal_lag
            ],
            values[
                start - nonlinear_lag : len(values) - nonlinear_lag
            ],
        ]
    )
    nonlinear_term = (
        np.sin(
            1.1
            * values[
                start - nonlinear_lag : len(values) - nonlinear_lag
            ]
        )
        ** 2
    )
    response_residual = response - linear_design @ np.linalg.lstsq(
        linear_design,
        response,
        rcond=None,
    )[0]
    nonlinear_residual = nonlinear_term - linear_design @ np.linalg.lstsq(
        linear_design,
        nonlinear_term,
        rcond=None,
    )[0]
    slope = float(
        nonlinear_residual @ response_residual
        / max(nonlinear_residual @ nonlinear_residual, 1e-12)
    )
    return (
        nonlinear_residual,
        response_residual,
        slope * nonlinear_residual,
    )


def plot_example(
    spec: CapabilityPlotSpec,
    intensity: int,
    seed: int,
    sample: dict[str, Any],
    dataset_local_target: float,
) -> Path:
    target = np.asarray(sample["target"], dtype=float)
    covariates = sample["covariates"]
    x = np.arange(-spec.context_length, spec.horizon)
    primary = PRIMARY_FEATURE[spec.capability_id]
    realized = float(sample["features"][primary])
    target_colors = ("#1d4ed8", "#0f766e", "#7e22ce", "#be123c")

    diagnostic_axis: plt.Axes | None = None
    if (
        covariates is None
        and spec.capability_id == "nonlinear_persistence"
    ):
        figure, (axis, diagnostic_axis) = plt.subplots(
            2,
            1,
            figsize=(10.2, 6.2),
            dpi=160,
            gridspec_kw={
                "height_ratios": (2.1, 1.0),
                "hspace": 0.34,
            },
        )
        axes = [axis, diagnostic_axis]
    elif covariates is None:
        figure, axis = plt.subplots(figsize=(10.2, 4.8), dpi=160)
        axes = [axis]
    else:
        figure, (axis, cov_axis) = plt.subplots(
            2,
            1,
            figsize=(10.2, 6.0),
            dpi=160,
            sharex=True,
            gridspec_kw={"height_ratios": (2.1, 1.0), "hspace": 0.10},
        )
        axes = [axis, cov_axis]

    for channel in range(target.shape[1]):
        if spec.hierarchy:
            label = "父节点 Σ" if channel == 0 else f"子节点 {channel}"
            linewidth = 2.2 if channel == 0 else 1.35
            alpha = 1.0 if channel == 0 else 0.82
        elif target.shape[1] > 1:
            label = f"目标 {channel + 1}"
            linewidth = 1.6
            alpha = 0.92
        else:
            label = "目标序列"
            linewidth = 1.8
            alpha = 1.0
        axis.plot(
            x,
            target[:, channel],
            color=target_colors[channel % len(target_colors)],
            linewidth=linewidth,
            alpha=alpha,
            label=label,
        )
    style_axis(
        axis,
        horizon=spec.horizon,
        with_xlabel=(
            covariates is None and diagnostic_axis is None
        ),
    )
    axis.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        edgecolor="#e2e8f0",
        fontsize=8,
        ncol=min(3, target.shape[1]),
    )

    if diagnostic_axis is not None:
        nonlinear_x, response_y, fitted_y = (
            nonlinear_partial_relationship(
                target,
                spec.season_length,
            )
        )
        diagnostic_axis.scatter(
            nonlinear_x,
            response_y,
            color="#2563eb",
            alpha=0.42,
            s=12,
            linewidths=0,
            label="残差化观测",
        )
        order = np.argsort(nonlinear_x)
        diagnostic_axis.plot(
            nonlinear_x[order],
            fitted_y[order],
            color="#c2410c",
            linewidth=1.8,
            label="条件非线性拟合",
        )
        diagnostic_axis.axhline(
            0.0,
            color="#94a3b8",
            linewidth=0.8,
        )
        diagnostic_axis.grid(
            color="#cbd5e1",
            linewidth=0.65,
            alpha=0.65,
        )
        diagnostic_axis.spines[["top", "right"]].set_visible(False)
        diagnostic_axis.spines[["left", "bottom"]].set_color("#94a3b8")
        diagnostic_axis.tick_params(colors="#475569", labelsize=8)
        diagnostic_axis.set_xlabel(
            "控制线性滞后后的 nonlinear lag feature",
            color="#334155",
            fontsize=9,
        )
        diagnostic_axis.set_ylabel(
            "目标残差",
            color="#334155",
            fontsize=9,
        )
        diagnostic_axis.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            edgecolor="#e2e8f0",
            fontsize=8,
            ncol=2,
        )
    elif covariates is not None:
        covariate_array = np.asarray(covariates, dtype=float)
        cov_axis.plot(
            x,
            covariate_array[:, 0],
            color="#0f766e",
            linewidth=1.45,
            label="已知未来天气",
        )
        cov_axis.step(
            x,
            covariate_array[:, 1],
            where="post",
            color="#c2410c",
            linewidth=1.6,
            label="已知未来事件",
        )
        style_axis(cov_axis, horizon=spec.horizon, with_xlabel=True)
        cov_axis.set_ylabel("协变量", color="#334155", fontsize=9)
        cov_axis.legend(
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            edgecolor="#e2e8f0",
            fontsize=8,
            ncol=2,
        )

    figure.patch.set_facecolor("#f8fafc")
    for current_axis in axes:
        current_axis.set_facecolor("#ffffff")
    figure.suptitle(
        f"{spec.title_zh} · {spec.capability_id} · intensity={intensity}",
        x=0.075,
        y=0.985,
        ha="left",
        color="#0f172a",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.075,
        0.936,
        (
            f"v2 paired seed {seed}  ·  "
            f"C={spec.context_length}, H={spec.horizon}  ·  "
            "浅黄色为预测区间"
        ),
        ha="left",
        color="#64748b",
        fontsize=8.6,
    )
    figure.text(
        0.925,
        0.952,
        f"{primary}\n{realized:.4f}  /  local target {dataset_local_target:.4f}",
        ha="right",
        va="top",
        color="#1e3a8a",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#eff6ff",
            "edgecolor": "#bfdbfe",
            "linewidth": 0.8,
        },
    )
    figure.subplots_adjust(
        top=0.84 if len(axes) == 1 else 0.86,
        left=0.075,
        right=0.965,
        bottom=0.12 if len(axes) == 1 else 0.10,
    )

    path = OUTPUT_DIR / f"{spec.capability_id}-intensity-{intensity}.png"
    figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    required = (
        GENERATOR_ARTIFACT_PATH,
        FEATURE_GATE_ARTIFACT_PATH,
        NEAR_GATE_ARTIFACT_PATH,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing capts-paper-v2 artifacts:\n" + "\n".join(missing))

    generator_artifact = read_json(GENERATOR_ARTIFACT_PATH)
    feature_artifact = read_json(FEATURE_GATE_ARTIFACT_PATH)
    near_artifact = read_json(NEAR_GATE_ARTIFACT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for spec in CAPABILITIES:
        seed, examples, candidate_number, selection_score = find_paired_examples(
            spec,
            generator_artifact=generator_artifact,
            feature_artifact=feature_artifact,
            near_artifact=near_artifact,
        )
        capability_record = generator_artifact["profiles"][
            spec.profile_id
        ]["capabilities"][spec.capability_id]
        anchors = capability_record["target_values"]
        for intensity in INTENSITIES:
            sample = examples[intensity]
            image_path = plot_example(
                spec,
                intensity,
                seed,
                sample,
                float(anchors[intensity - 1]),
            )
            primary = PRIMARY_FEATURE[spec.capability_id]
            manifest_rows.append(
                {
                    "capability_id": spec.capability_id,
                    "intensity": intensity,
                    "paired_seed": seed,
                    "seed_candidate_number": candidate_number,
                    "visualization_seed_bank_size": (
                        representative_seed_bank_size(
                            spec.capability_id
                        )
                    ),
                    "representative_selection_score": round(
                        selection_score,
                        8,
                    ),
                    "generator_profile_id": spec.profile_id,
                    "plot_context_length": spec.context_length,
                    "horizon": spec.horizon,
                    "primary_feature": primary,
                    "primary_feature_value": round(
                        float(sample["features"][primary]),
                        8,
                    ),
                    "dataset_local_target_value": float(anchors[intensity - 1]),
                    "image": image_path.relative_to(
                        Path(__file__).resolve().parent
                    ).as_posix(),
                    "online_profile_gates_passed": True,
                    "gate_views": sample["gate_views"],
                }
            )
        print(
            f"{spec.capability_id}: paired seed={seed}, "
            f"candidate={candidate_number}, score={selection_score:.4f}",
            flush=True,
        )

    manifest = {
        "schema_version": "capability_example_manifest.v3",
        "intensity_policy": generator_artifact["intensity_policy"],
        "source_artifacts": [
            str(path.relative_to(REPO_ROOT)) for path in required
        ],
        "selection_policy": (
            "visualization only: from a frozen 64-seed bank (256 seeds for "
            "the predeclared high-variance nonlinear statistic), choose one "
            "nuisance-paired seed per capability whose intensities 1, 3, and 5 "
            "all pass construction, feature-support, and near-distance gates; "
            "require a strictly increasing primary feature and minimize its "
            "normalized distance to the selected dataset/profile's three local "
            "q10/q50/q90 targets; forecast "
            "error and capability contrast are never used for selection, and "
            "this policy is not part of online sample acceptance"
        ),
        "examples": manifest_rows,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(manifest_rows)} images and {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
