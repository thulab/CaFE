from __future__ import annotations

import pytest

from cafe.analysis import runner


def _effect(
    *,
    table: str,
    pair_id: str,
    nrmse: float,
    excluded: bool,
    source_pair_id: str | None = None,
) -> dict[str, object]:
    return {
        "benchmark_track": runner.REAL_ANCHORED_BENCHMARK_TRACK,
        "evaluation_table": table,
        "dataset_id": "gift_fixture",
        "capability_id": "common_factor",
        "model_id": "model-a",
        "background_id": "background-0",
        "master_counterfactual_pair_id": pair_id,
        "input_ablation_source_pair_id": source_pair_id,
        "intensity": 5,
        "dose_value": 2.0,
        "counterfactual_effect_nrmse": nrmse,
        "truth_effect_rms": 0.5,
        "forecast_effect_rms": 0.4,
        "excluded_from_primary_score": excluded,
    }


def test_structural_input_ablation_is_reported_but_not_score_weighted() -> None:
    main_pair = "pair-main"
    result = runner.real_anchored_input_ablation_attribution(
        [
            _effect(
                table=runner.REAL_ANCHORED_BENCHMARK_TRACK,
                pair_id=main_pair,
                nrmse=0.25,
                excluded=False,
            ),
            _effect(
                table=runner.REAL_ANCHORED_INPUT_ABLATION_TABLE,
                pair_id="pair-ablation",
                source_pair_id=main_pair,
                nrmse=0.75,
                excluded=True,
            ),
        ]
    )
    assert result["rows"][0]["schema_version"] == (
        "cafe.real_anchored_input_ablation_attribution.v1"
    )
    assert result["rows"][0]["effect_nrmse_increase"] == pytest.approx(0.5)
    assert result["rows"][0]["primary_score_weight"] == 0.0
    assert result["summaries"][0]["included_in_primary_score_or_rank"] is False


def test_structural_input_ablation_is_mandatory_for_each_main_pair() -> None:
    with pytest.raises(ValueError, match="mandatory structural input ablations"):
        runner.real_anchored_input_ablation_attribution(
            [
                _effect(
                    table=runner.REAL_ANCHORED_BENCHMARK_TRACK,
                    pair_id="pair-main",
                    nrmse=0.25,
                    excluded=False,
                )
            ]
        )


def test_auxiliary_sensitivities_are_projected_without_ranking() -> None:
    d2_pair = "pair-d2-main"
    zero_pair = "pair-zero-main"
    effects = [
        _effect(
            table=runner.REAL_ANCHORED_STRUCTURAL_SENSITIVITY_TABLE,
            pair_id=d2_pair,
            nrmse=0.3,
            excluded=True,
        ),
        _effect(
            table=(
                runner.REAL_ANCHORED_STRUCTURAL_SENSITIVITY_ABLATION_TABLE
            ),
            pair_id="pair-d2-ablation",
            source_pair_id=d2_pair,
            nrmse=0.7,
            excluded=True,
        ),
        {
            **_effect(
                table=runner.REAL_ANCHORED_BENCHMARK_TRACK,
                pair_id=zero_pair,
                nrmse=0.2,
                excluded=False,
            ),
            "capability_id": "nonlinear_persistence",
        },
        {
            **_effect(
                table=(
                    runner.REAL_ANCHORED_NONLINEAR_REPLAY_SENSITIVITY_TABLE
                ),
                pair_id="pair-replay",
                nrmse=0.5,
                excluded=True,
            ),
            "capability_id": "nonlinear_persistence",
            "sensitivity_source_pair_id": zero_pair,
        },
    ]

    result = runner.real_anchored_sensitivity_analysis(effects)

    assert len(result["effects"]) == 3
    d2 = result["structural_d2_input_ablation"]
    assert d2["rows"][0]["effect_nrmse_increase"] == pytest.approx(0.4)
    nonlinear = result["nonlinear_replay_comparisons"][0]
    assert nonlinear["effect_nrmse_difference"] == pytest.approx(0.3)
    assert result["nonlinear_replay_summaries"][0][
        "included_in_primary_score_or_rank"
    ] is False
