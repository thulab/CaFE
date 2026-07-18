import numpy as np

from app.services.synthetic_capability_contrast import (
    capability_contrast_forecasts,
    evaluate_capability_contrast,
)
from app.services.synthetic_generation_service import (
    _generate_sample_values,
    _realized_features,
)


CONTEXT_LENGTH = 365
HORIZON = 28
SEASON_LENGTH = 7
TARGET_DIM = 3


def _generate(intensity: int, seed: int, *, horizon: int = HORIZON):
    return _generate_sample_values(
        "hierarchical_coherence",
        CONTEXT_LENGTH + horizon,
        CONTEXT_LENGTH,
        TARGET_DIM,
        SEASON_LENGTH,
        intensity,
        np.random.default_rng(seed),
    )


def test_m5_shape_is_exactly_coherent_with_sample_specific_local_laws():
    root_pairs: set[tuple[float, float]] = set()
    path_checksums: set[str] = set()
    for seed in range(48):
        values, metadata, _ = _generate(5, seed)
        processes = metadata["local_process"]["component_processes"]

        np.testing.assert_allclose(
            values[:, 0],
            np.sum(values[:, 1:], axis=1),
            atol=1e-12,
        )
        assert metadata["hierarchy"] == "target_0=sum(target_1:)"
        assert metadata["child_count"] == 2
        assert metadata["local_contrast_rank"] == 1
        assert (
            metadata["predictability"]["evidence"]["future_only_shock_count"]
            == 0
        )
        assert (
            metadata["predictability"]["evidence"][
                "component_laws_constant_across_boundary"
            ]
            is True
        )
        root_pairs.add(
            (
                round(processes[0]["slow_root"], 6),
                round(processes[0]["fast_root"], 6),
            )
        )
        path_checksums.add(metadata["local_latent_paths_checksum"])

    assert len(root_pairs) == 48
    assert len(path_checksums) == 48


def test_intensity_only_rescales_the_same_local_heterogeneity_path():
    low, low_metadata, _ = _generate(1, 73)
    middle, middle_metadata, _ = _generate(3, 73)
    high, high_metadata, _ = _generate(5, 73)

    invariant_keys = (
        "child_count",
        "local_contrast_rank",
        "shared_component_checksum",
        "local_latent_paths_checksum",
        "local_components_checksum",
        "common_noise_checksum",
        "idiosyncratic_noise_checksum",
        "local_contrast_loadings",
        "local_amplitude",
        "noise_scale",
    )
    for key in invariant_keys:
        assert low_metadata[key] == middle_metadata[key] == high_metadata[key]

    np.testing.assert_allclose(low[:, 0], middle[:, 0], atol=1e-12)
    np.testing.assert_allclose(low[:, 0], high[:, 0], atol=1e-12)
    np.testing.assert_allclose(
        np.sum(high[:, 1:] - low[:, 1:], axis=1),
        0.0,
        atol=1e-12,
    )

    low_to_middle = (
        middle[:, 1:] - low[:, 1:]
    ) / (
        middle_metadata["heterogeneity_strength"]
        - low_metadata["heterogeneity_strength"]
    )
    low_to_high = (
        high[:, 1:] - low[:, 1:]
    ) / (
        high_metadata["heterogeneity_strength"]
        - low_metadata["heterogeneity_strength"]
    )
    np.testing.assert_allclose(low_to_middle, low_to_high, atol=1e-12)


def test_high_intensity_child_heterogeneity_is_visibly_stronger():
    low_features: list[float] = []
    high_features: list[float] = []
    for seed in range(64):
        low, _, _ = _generate(1, seed)
        high, _, _ = _generate(5, seed)
        low_features.append(
            _realized_features(
                low,
                None,
                SEASON_LENGTH,
                CONTEXT_LENGTH,
            )["hierarchy_child_heterogeneity"]
        )
        high_features.append(
            _realized_features(
                high,
                None,
                SEASON_LENGTH,
                CONTEXT_LENGTH,
            )["hierarchy_child_heterogeneity"]
        )

    assert float(np.quantile(high_features, 0.1)) > 2.0 * float(
        np.quantile(low_features, 0.9)
    )


def test_history_only_contrast_is_exactly_coherent_and_ignores_latent_paths():
    target, metadata, covariates = _generate(5, 89)
    first = capability_contrast_forecasts(
        capability_id="hierarchical_coherence",
        history=target[:CONTEXT_LENGTH],
        horizon=HORIZON,
        season_length=SEASON_LENGTH,
        latent_params=metadata,
        covariates=covariates,
    )
    changed = {
        **metadata,
        "local_contrast_loadings": [[10_000.0], [-10_000.0]],
        "local_latent_paths_checksum": "changed",
        "local_components_checksum": "changed",
    }
    second = capability_contrast_forecasts(
        capability_id="hierarchical_coherence",
        history=target[:CONTEXT_LENGTH],
        horizon=HORIZON,
        season_length=SEASON_LENGTH,
        latent_params=changed,
        covariates=covariates,
    )

    assert (
        first["aware_method"]
        == "history_estimated_aggregate_plus_local_contrasts"
    )
    np.testing.assert_allclose(first["aware"], second["aware"])
    np.testing.assert_allclose(
        first["aware"][:, 0],
        np.sum(first["aware"][:, 1:], axis=1),
        atol=1e-12,
    )


def test_m5_horizon_prefix_and_high_intensity_contrast_are_stable():
    wins = 0
    gains: list[float] = []
    for seed in range(64):
        short, metadata, covariates = _generate(5, seed)
        long, long_metadata, _ = _generate(5, seed, horizon=2 * HORIZON)
        np.testing.assert_allclose(
            short,
            long[: CONTEXT_LENGTH + HORIZON],
            atol=1e-12,
        )
        assert (
            metadata["local_latent_paths_checksum"]
            != long_metadata["local_latent_paths_checksum"]
        )
        result = evaluate_capability_contrast(
            capability_id="hierarchical_coherence",
            target=short,
            context_length=CONTEXT_LENGTH,
            season_length=SEASON_LENGTH,
            intensity=5,
            latent_params=metadata,
            covariates=covariates,
            evaluation_scale="generator_raw",
        )
        wins += int(result["aware_wins"])
        gains.append(result["relative_loss_gain"])

    assert wins >= 48
    assert float(np.median(gains)) > 0.05
