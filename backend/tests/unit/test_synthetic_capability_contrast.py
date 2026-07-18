import numpy as np

from app.services.synthetic_capability_contrast import (
    NONLINEAR_RECURSIVE_SHRINKAGE,
    _nonlinear_forecast,
    capability_contrast_forecasts,
    evaluate_capability_contrast,
    summarize_capability_contrasts,
)
from app.services.synthetic_generation_service import (
    CAPABILITIES_BY_ID,
    PAPER_CAPABILITY_IDS,
    _generate_sample_values,
    _standardize_by_context,
    _standardize_hierarchy_by_context,
)


NONSEASONAL_CAPABILITIES = (
    "trend",
    "regime_switching",
    "nonlinear_persistence",
    "predictable_intermittency",
    "common_factor",
    "hierarchical_coherence",
    "covariate_response",
)


def _target_dim(capability_id: str) -> int:
    return (
        3
        if CAPABILITIES_BY_ID[capability_id].target_dim_mode == "multi"
        else 1
    )


def test_capability_forecasts_cannot_read_future_targets():
    target, latent, covariates = _generate_sample_values(
        "covariate_response",
        192,
        168,
        1,
        24,
        5,
        np.random.default_rng(11),
    )
    first = capability_contrast_forecasts(
        capability_id="covariate_response",
        history=target[:168],
        horizon=24,
        season_length=24,
        latent_params=latent,
        covariates=covariates,
    )
    changed_future = np.array(target, copy=True)
    changed_future[168:] += 10_000.0
    second = capability_contrast_forecasts(
        capability_id="covariate_response",
        history=changed_future[:168],
        horizon=24,
        season_length=24,
        latent_params=latent,
        covariates=covariates,
    )

    np.testing.assert_allclose(first["blind"], second["blind"])
    np.testing.assert_allclose(first["aware"], second["aware"])
    assert first["future_target_used_for_forecast"] is False
    assert (
        first["aware_method"]
        == "history_estimated_known_future_covariate_regression"
    )
    assert first["aware_information_set"].endswith(
        "_plus_known_future_covariates"
    )


def test_clock_aware_forecasts_ignore_recorded_future_event_positions():
    context_length = 168
    horizon = 24
    for capability_id, metadata_key in (
        ("regime_switching", "cut_points"),
        ("predictable_intermittency", "pulse_centers"),
    ):
        target, latent, covariates = _generate_sample_values(
            capability_id,
            context_length + horizon,
            context_length,
            1,
            24,
            5,
            np.random.default_rng(17),
        )
        first = capability_contrast_forecasts(
            capability_id=capability_id,
            history=target[:context_length],
            horizon=horizon,
            season_length=24,
            latent_params=latent,
            covariates=covariates,
        )
        changed_latent = dict(latent)
        historical_events = [
            int(value)
            for value in latent[metadata_key]
            if int(value) < context_length
        ]
        changed_latent[metadata_key] = [
            *historical_events,
            context_length + 1,
            context_length + 3,
        ]
        second = capability_contrast_forecasts(
            capability_id=capability_id,
            history=target[:context_length],
            horizon=horizon,
            season_length=24,
            latent_params=changed_latent,
            covariates=covariates,
        )

        np.testing.assert_allclose(first["blind"], second["blind"])
        np.testing.assert_allclose(first["aware"], second["aware"])


def test_nonseasonal_capabilities_do_not_share_a_seasonal_naive_shortcut():
    context_length = 168
    horizon = 24
    season_length = 24
    for capability_id in NONSEASONAL_CAPABILITIES:
        seasonal_errors: list[float] = []
        last_errors: list[float] = []
        for seed in range(48):
            target, _, _ = _generate_sample_values(
                capability_id,
                context_length + horizon,
                context_length,
                _target_dim(capability_id),
                season_length,
                5,
                np.random.default_rng(seed),
            )
            target = (
                _standardize_hierarchy_by_context(target, context_length)
                if capability_id == "hierarchical_coherence"
                else _standardize_by_context(target, context_length)
            )
            history = target[:context_length]
            future = target[context_length:]
            seasonal_errors.append(
                float(np.mean(np.abs(future - history[-season_length:])))
            )
            last_errors.append(
                float(np.mean(np.abs(future - history[-1:])))
            )
        seasonal_to_last = float(
            np.mean(seasonal_errors) / np.mean(last_errors)
        )
        assert seasonal_to_last > 0.85, (
            capability_id,
            seasonal_to_last,
        )


def test_high_intensity_capability_contrast_passes_for_all_capabilities():
    context_length = 168
    horizon = 24
    season_length = 24
    for capability_id in PAPER_CAPABILITY_IDS:
        rows = []
        for seed in range(128):
            target, latent, covariates = _generate_sample_values(
                capability_id,
                context_length + horizon,
                context_length,
                _target_dim(capability_id),
                season_length,
                5,
                np.random.default_rng(seed),
            )
            rows.append(
                evaluate_capability_contrast(
                    capability_id=capability_id,
                    target=target,
                    context_length=context_length,
                    season_length=season_length,
                    intensity=5,
                    latent_params=latent,
                    covariates=covariates,
                    evaluation_scale="generator_raw",
                )
            )
        summary = summarize_capability_contrasts(rows)
        assert summary["passed"] is True, summary


def test_contrast_summary_uses_paired_difference_of_mean_loss():
    rows = [
        {
            "capability_id": "trend",
            "intensity": 5,
            "blind_composite_loss": 1.0,
            "aware_composite_loss": 0.0,
            "relative_loss_gain": 1.0,
            "aware_wins": True,
        },
        {
            "capability_id": "trend",
            "intensity": 5,
            "blind_composite_loss": 100.0,
            "aware_composite_loss": 90.0,
            "relative_loss_gain": 0.1,
            "aware_wins": True,
        },
    ]

    summary = summarize_capability_contrasts(
        rows,
        minimum_sample_count=2,
        minimum_win_rate=0.0,
        minimum_mean_relative_gain=-1.0,
    )

    assert summary["aggregation"] == (
        "paired_difference_of_mean_composite_loss"
    )
    assert np.isclose(summary["mean_relative_loss_gain"], 11.0 / 101.0)


def test_hierarchy_aware_forecast_is_exactly_coherent():
    target, latent, _ = _generate_sample_values(
        "hierarchical_coherence",
        192,
        168,
        3,
        24,
        5,
        np.random.default_rng(7),
    )
    forecasts = capability_contrast_forecasts(
        capability_id="hierarchical_coherence",
        history=target[:168],
        horizon=24,
        season_length=24,
        latent_params=latent,
    )
    aware = forecasts["aware"]
    np.testing.assert_allclose(
        aware[:, 0],
        np.sum(aware[:, 1:], axis=1),
        atol=1e-12,
    )


def test_nonlinear_aware_forecast_uses_frozen_recursive_shrinkage():
    target, latent, _ = _generate_sample_values(
        "nonlinear_persistence",
        192,
        168,
        1,
        24,
        5,
        np.random.default_rng(23),
    )
    history = target[:168]
    forecasts = capability_contrast_forecasts(
        capability_id="nonlinear_persistence",
        history=history,
        horizon=24,
        season_length=24,
        latent_params=latent,
    )
    raw_nonlinear = _nonlinear_forecast(
        history,
        24,
        24,
        latent,
    )
    expected = forecasts["blind"] + NONLINEAR_RECURSIVE_SHRINKAGE * (
        raw_nonlinear - forecasts["blind"]
    )

    np.testing.assert_allclose(forecasts["aware"], expected)
    assert forecasts["aware_method"] == (
        "shrunken_nonlinear_multi_lag_recurrence"
    )
