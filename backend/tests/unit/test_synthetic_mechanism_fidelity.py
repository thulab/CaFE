import json

import numpy as np
import pytest

from app.services.synthetic_generation_service import (
    PAPER_CAPABILITY_IDS,
    _generate_sample_values,
    _normalize_covariates,
    _standardize_by_context,
    _standardize_hierarchy_by_context,
)
from app.services.synthetic_mechanism_fidelity import (
    capability_score,
    evaluate_mechanism_fidelity,
)


CONTEXT_LENGTH = 168
HORIZON = 48
SEASON_LENGTH = 24


def generated_case(
    capability_id: str,
    *,
    seed: int = 11,
    intensity: int = 5,
) -> tuple[np.ndarray, dict, np.ndarray | None]:
    target_dim = (
        3
        if capability_id in {"common_factor", "hierarchical_coherence"}
        else 1
    )
    raw, latent, covariates = _generate_sample_values(
        capability_id,
        CONTEXT_LENGTH + HORIZON,
        CONTEXT_LENGTH,
        target_dim,
        SEASON_LENGTH,
        intensity,
        np.random.default_rng(seed),
    )
    target = (
        _standardize_hierarchy_by_context(raw, CONTEXT_LENGTH)
        if capability_id == "hierarchical_coherence"
        else _standardize_by_context(raw, CONTEXT_LENGTH)
    )
    normalized_covariates = (
        _normalize_covariates(covariates, CONTEXT_LENGTH)
        if covariates is not None
        else None
    )
    return target, latent, normalized_covariates


@pytest.mark.parametrize("capability_id", PAPER_CAPABILITY_IDS)
def test_oracle_output_gets_full_mechanism_fidelity(capability_id: str):
    target, latent, covariates = generated_case(capability_id)

    result = evaluate_mechanism_fidelity(
        capability_id=capability_id,
        history=target[:CONTEXT_LENGTH],
        target_future=target[CONTEXT_LENGTH:],
        forecast=target[CONTEXT_LENGTH:],
        season_length=SEASON_LENGTH,
        latent_params=latent,
        intensity=5,
        forecast_start_index=CONTEXT_LENGTH,
        covariates=covariates,
    )

    assert result["mechanism_fidelity_score"] == pytest.approx(1.0)
    assert all(
        0.0 <= value <= 1.0
        for value in result["component_scores"].values()
    )
    assert result["causal_mechanism_claim"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("capability_id", PAPER_CAPABILITY_IDS)
def test_constant_output_cannot_match_oracle_mechanism_score(
    capability_id: str,
):
    scores = []
    for seed in range(4):
        target, latent, covariates = generated_case(
            capability_id,
            seed=seed,
        )
        constant = np.repeat(
            target[CONTEXT_LENGTH - 1 : CONTEXT_LENGTH],
            HORIZON,
            axis=0,
        )
        result = evaluate_mechanism_fidelity(
            capability_id=capability_id,
            history=target[:CONTEXT_LENGTH],
            target_future=target[CONTEXT_LENGTH:],
            forecast=constant,
            season_length=SEASON_LENGTH,
            latent_params=latent,
            intensity=5,
            forecast_start_index=CONTEXT_LENGTH,
            covariates=covariates,
        )
        scores.append(result["mechanism_fidelity_score"])

    assert float(np.mean(scores)) < 0.5, (capability_id, scores)


def test_covariate_score_is_diagnostic_without_paired_ablation():
    target, latent, covariates = generated_case("covariate_response")

    observational = evaluate_mechanism_fidelity(
        capability_id="covariate_response",
        history=target[:CONTEXT_LENGTH],
        target_future=target[CONTEXT_LENGTH:],
        forecast=target[CONTEXT_LENGTH:],
        season_length=SEASON_LENGTH,
        latent_params=latent,
        intensity=5,
        forecast_start_index=CONTEXT_LENGTH,
        covariates=covariates,
    )
    paired = evaluate_mechanism_fidelity(
        capability_id="covariate_response",
        history=target[:CONTEXT_LENGTH],
        target_future=target[CONTEXT_LENGTH:],
        forecast=target[CONTEXT_LENGTH:],
        counterfactual_forecast=np.zeros_like(target[CONTEXT_LENGTH:]),
        season_length=SEASON_LENGTH,
        latent_params=latent,
        intensity=5,
        forecast_start_index=CONTEXT_LENGTH,
        covariates=covariates,
    )

    assert observational["formal_score_eligible"] is False
    assert (
        observational["diagnostics"]["evaluation_mode"]
        == "observational_future_covariate_projection"
    )
    assert paired["formal_score_eligible"] is True
    assert (
        paired["diagnostics"]["evaluation_mode"]
        == "paired_future_covariate_ablation"
    )


def test_capability_score_only_penalizes_models_worse_than_blind_reference():
    assert capability_score(
        mechanism_fidelity_score=0.8,
        model_point_loss=0.5,
        blind_point_loss=1.0,
    ) == pytest.approx(0.8)
    assert capability_score(
        mechanism_fidelity_score=0.8,
        model_point_loss=2.0,
        blind_point_loss=1.0,
    ) == pytest.approx(0.4)


def score_forecast(
    capability_id: str,
    target: np.ndarray,
    latent: dict,
    forecast: np.ndarray,
    covariates: np.ndarray | None = None,
) -> dict:
    return evaluate_mechanism_fidelity(
        capability_id=capability_id,
        history=target[:CONTEXT_LENGTH],
        target_future=target[CONTEXT_LENGTH:],
        forecast=forecast,
        season_length=SEASON_LENGTH,
        latent_params=latent,
        intensity=5,
        forecast_start_index=CONTEXT_LENGTH,
        covariates=covariates,
    )


def test_primary_period_only_loses_multi_seasonal_coverage():
    target, latent, _ = generated_case("multi_seasonal", seed=31)
    future = target[CONTEXT_LENGTH:]
    time = np.arange(HORIZON, dtype=float)
    primary = int(latent["periods"][0])
    design = np.column_stack(
        [
            np.ones(HORIZON),
            time,
            np.sin(2 * np.pi * time / primary),
            np.cos(2 * np.pi * time / primary),
        ]
    )
    primary_only = design @ np.linalg.lstsq(design, future, rcond=None)[0]

    ablated = score_forecast(
        "multi_seasonal",
        target,
        latent,
        primary_only,
    )

    assert ablated["mechanism_fidelity_score"] < 0.8
    assert ablated["detection_score"] < 1.0


def test_fixed_carrier_loses_time_varying_modulation():
    target, latent, _ = generated_case(
        "time_varying_seasonality",
        seed=37,
    )
    future = target[CONTEXT_LENGTH:]
    time = np.arange(
        CONTEXT_LENGTH,
        CONTEXT_LENGTH + HORIZON,
        dtype=float,
    )
    carrier = np.sin(
        2 * np.pi * time / int(latent["primary_period"])
        + float(latent["carrier_phase_by_target"][0])
    )
    design = np.column_stack(
        [np.ones(HORIZON), time - time.mean(), carrier]
    )
    fixed_carrier = design @ np.linalg.lstsq(
        design,
        future,
        rcond=None,
    )[0]

    ablated = score_forecast(
        "time_varying_seasonality",
        target,
        latent,
        fixed_carrier,
    )

    assert ablated["mechanism_fidelity_score"] < 0.5
    assert ablated["magnitude_score"] < 0.5


def test_smoothed_regime_forecast_loses_switch_timing_and_magnitude():
    target, latent, _ = generated_case("regime_switching", seed=41)
    future = target[CONTEXT_LENGTH:, 0]
    smoothed = np.convolve(future, np.ones(17) / 17, mode="same")[:, None]

    ablated = score_forecast(
        "regime_switching",
        target,
        latent,
        smoothed,
    )

    assert ablated["mechanism_fidelity_score"] < 0.8
    assert (
        ablated["timing_score"] < 1.0
        or ablated["magnitude_score"] < 0.8
    )


def test_linear_background_loses_intermittent_pulse():
    target, latent, _ = generated_case(
        "predictable_intermittency",
        seed=43,
    )
    future = target[CONTEXT_LENGTH:]
    design = np.column_stack(
        [np.ones(HORIZON), np.linspace(-1.0, 1.0, HORIZON)]
    )
    background = design @ np.linalg.lstsq(design, future, rcond=None)[0]

    ablated = score_forecast(
        "predictable_intermittency",
        target,
        latent,
        background,
    )

    assert ablated["mechanism_fidelity_score"] == pytest.approx(0.0)
    assert ablated["detection_score"] == pytest.approx(0.0)


def test_incoherent_parent_is_penalized_even_with_exact_children():
    target, latent, _ = generated_case(
        "hierarchical_coherence",
        seed=47,
    )
    forecast = target[CONTEXT_LENGTH:].copy()
    forecast[:, 0] += 2.0

    ablated = score_forecast(
        "hierarchical_coherence",
        target,
        latent,
        forecast,
    )

    assert ablated["mechanism_fidelity_score"] < 1.0
    assert ablated["selectivity_score"] < 0.5
