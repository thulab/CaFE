from sqlmodel import Session, select

from app.models.benchmark import CapabilityBlock, CapabilityBlockShard, Track
from app.models.dataset import Shard
from app.models.sample import SampleIndex


def test_synthetic_capabilities_and_generation_materialize_shards(app, client):
    catalog = client.get("/synthetic/capabilities")
    assert catalog.status_code == 200
    capability_ids = {item["capability_id"] for item in catalog.json()["items"]}
    assert capability_ids == {
        "trend",
        "multi_seasonal",
        "time_varying_seasonality",
        "regime_switching",
        "nonlinear_persistence",
        "predictable_intermittency",
        "common_factor",
        "hierarchical_coherence",
        "covariate_response",
    }
    trend_capability = next(item for item in catalog.json()["items"] if item["capability_id"] == "trend")
    assert trend_capability["label_i18n"]["zh-CN"] == "趋势"
    assert trend_capability["paper_included"] is True
    assert trend_capability["paper_track"] == "univariate"
    assert trend_capability["generator_version"] == "capts-paper-v1"
    assert trend_capability["predictability_contract"]
    assert trend_capability["intensity_features"]["trend_strength"] == "increase"
    assert trend_capability["limits"]["context_length"]["min"] == 16

    response = client.post(
        "/synthetic/shards",
        json={
            "name": "synthetic smoke",
            "capabilities": ["trend", "covariate_response"],
            "context_length": 168,
            "horizon": 24,
            "sample_count": 3,
            "intensity": 3,
            "season_length": 8,
            "target_dim": 2,
            "seed": 42,
            "frequency": "h",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["shard_ids"]) == 2

    with Session(app.state.engine) as session:
        shards = session.exec(select(Shard).where(Shard.shard_id.in_(body["shard_ids"]))).all()
        assert {shard.shard_type for shard in shards} == {"synthetic"}
        trend = next(shard for shard in shards if shard.capability_type == "trend")
        covariate = next(shard for shard in shards if shard.capability_type == "covariate_response")
        assert trend.target_dim == 1
        assert covariate.target_dim == 1
        assert covariate.covariate_columns == ["weather", "event"]
        assert covariate.generation_config["intensity"] == 3
        assert covariate.generation_config["difficulty"] == 3
        assert covariate.generation_config["requested_season_length"] == 8
        assert covariate.generation_config["season_length"] == 24
        assert covariate.generation_config["season_length_source"] == "preselected_anchor_profile"
        assert covariate.generation_config["anchor_mode"] == "preselected_profile_conditioned"
        assert covariate.generation_config["anchor_profiles"] == [
            "gefcom2014_load_hourly_covariate_168ctx_24h",
        ]
        assert covariate.generation_config["anchor_profile_selection"] == "balanced_uniform"

        trend_sample = session.exec(select(SampleIndex).where(SampleIndex.shard_id == trend.shard_id)).first()
        assert trend_sample is not None
        trend_metadata = trend_sample.sample_metadata
        assert trend_metadata["intensity"] == 3
        assert trend_metadata["difficulty"] == 3
        assert trend_metadata["latent_params"]["generator_version"] == "capts-paper-v1"
        assert trend_metadata["latent_params"]["predictability"]["construction_validated"] is True
        assert trend_metadata["latent_params"]["acceptance"]["accepted"] is True
        assert trend_metadata["latent_params"]["acceptance"]["attempts"] == 1
        assert trend_metadata["latent_params"]["acceptance"]["profile_selection_stage"] == "pre_generation"
        assert trend_metadata["latent_params"]["generator_conditioning"]["profile_id"] == trend_metadata["anchor_profile_id"]
        assert trend_metadata["latent_params"]["acceptance"]["validation"]["schema_version"] == "synthetic_post_generation_validation.v3"
        assert "trend_strength" in trend_metadata["latent_params"]["acceptance"]["validation"]["target_features"]
        feature_gate = trend_metadata["latent_params"]["acceptance"]["validation"]["feature_gate"]
        assert feature_gate["accepted"] is True
        assert feature_gate["enforced"] is True
        assert feature_gate["support_method"] == "shrunk_robust_mahalanobis"
        assert feature_gate["calibration_coverage"] == 0.95
        assert feature_gate["artifact_schema_version"] == "synthetic_v2_feature_gate_online.v1"
        assert feature_gate["matched_profile_id"] in {
            "m4_hourly_daily_168ctx",
            "electricity_hourly_daily_168ctx",
            "traffic_hourly_daily_168ctx",
        }
        assert feature_gate["matched_profile_id"] == trend_metadata["anchor_profile_id"]
        assert feature_gate["target_percentile_diagnostics"]["trend_strength"]
        assert trend_metadata["latent_params"]["acceptance"]["validation"]["near_distance_gate"]["accepted"] is True
        assert trend_metadata["latent_params"]["acceptance"]["validation"]["predictability_gate"]["accepted"] is True
        assert trend_metadata["latent_params"]["acceptance"]["validation"]["novelty_check"] == "online_dcr_nndr_gate"
        assert "trend_strength" in trend_metadata["realized_features"]
        assert "nonlinear_lag1_gain" in trend_metadata["realized_features"]

        sample = session.exec(select(SampleIndex).where(SampleIndex.shard_id == covariate.shard_id)).first()
        assert sample is not None
        assert sample.sample_metadata["capability_id"] == "covariate_response"
        assert "avg_abs_covariate_target_corr" in sample.sample_metadata["realized_features"]

    sample_list = client.get(f"/shards/{body['shard_ids'][1]}/samples").json()
    sample_id = sample_list["items"][0]["sample_id"]
    preview = client.get(f"/samples/{sample_id}/preview").json()
    assert len(preview["target_history"]) == 168
    assert len(preview["target_future"]) == 24
    assert len(preview["target_history"][0]) == 1
    assert len(preview["history_cov"][0]) == 2
    assert len(preview["future_cov"]) == 24


def test_uncalibrated_synthetic_window_is_rejected(app, client):
    response = client.post(
        "/synthetic/shards",
        json={
            "name": "uncalibrated smoke",
            "capabilities": ["trend"],
            "context_length": 16,
            "horizon": 4,
            "sample_count": 1,
            "intensity": 3,
            "target_dim": 1,
            "seed": 42,
            "frequency": "h",
        },
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "synthetic_near_distance_not_calibrated"


def test_generation_can_pin_a_preselected_anchor_profile(app, client):
    response = client.post(
        "/synthetic/shards",
        json={
            "name": "traffic-conditioned trend",
            "capabilities": ["trend"],
            "context_length": 168,
            "horizon": 24,
            "sample_count": 2,
            "intensity": 3,
            "target_dim": 1,
            "seed": 9,
            "frequency": "h",
            "anchor_profile_ids": {"trend": "traffic_hourly_daily_168ctx"},
        },
    )

    assert response.status_code == 200, response.text
    shard_id = response.json()["shard_ids"][0]
    with Session(app.state.engine) as session:
        shard = session.get(Shard, shard_id)
        assert shard is not None
        assert shard.generation_config["anchor_profile_selection"] == "explicit"
        assert shard.generation_config["anchor_profiles"] == ["traffic_hourly_daily_168ctx"]
        samples = session.exec(select(SampleIndex).where(SampleIndex.shard_id == shard_id)).all()
        assert {sample.sample_metadata["anchor_profile_id"] for sample in samples} == {
            "traffic_hourly_daily_168ctx"
        }
        assert all(
            sample.sample_metadata["latent_params"]["acceptance"]["validation"]["feature_gate"][
                "matched_profile_id"
            ]
            == "traffic_hourly_daily_168ctx"
            for sample in samples
        )


def test_generation_rejects_an_incompatible_anchor_profile(client):
    response = client.post(
        "/synthetic/shards",
        json={
            "capabilities": ["trend"],
            "context_length": 168,
            "horizon": 24,
            "sample_count": 1,
            "intensity": 3,
            "target_dim": 1,
            "seed": 9,
            "frequency": "h",
            "anchor_profile_ids": {"trend": "m5_daily_covariate_365ctx_28h"},
        },
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "synthetic_anchor_profile_invalid"


def test_track_from_shards_groups_synthetic_capabilities(app, client):
    generated = client.post(
        "/synthetic/shards",
        json={
            "name": "synthetic grouped",
            "capabilities": ["trend", "common_factor"],
            "context_length": 168,
            "horizon": 24,
            "sample_count": 2,
            "difficulty": 2,
            "season_length": 8,
            "target_dim": 3,
            "seed": 7,
            "frequency": "h",
        },
    ).json()

    response = client.post(
        "/wizard/track-from-shards",
        json={"name": "synthetic track", "shard_ids": generated["shard_ids"], "primary_metric_id": "mase"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["capability_block_ids"]) == 2

    with Session(app.state.engine) as session:
        track = session.get(Track, body["track_id"])
        assert track is not None
        assert track.track_type == "synthetic_dataset"
        blocks = session.exec(select(CapabilityBlock).where(CapabilityBlock.track_id == track.track_id)).all()
        assert {block.capability_type for block in blocks} == {"trend", "common_factor"}
        links = session.exec(select(CapabilityBlockShard)).all()
        assert {link.shard_id for link in links} == set(generated["shard_ids"])


def test_predictable_event_capabilities_generate_json_serializable_metadata(app, client):
    response = client.post(
        "/synthetic/shards",
        json={
            "name": "regime smoke",
            "capabilities": ["regime_switching", "predictable_intermittency"],
            "context_length": 168,
            "horizon": 24,
            "sample_count": 2,
            "intensity": 4,
            "season_length": 8,
            "target_dim": 3,
            "seed": 13,
            "frequency": "h",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["shard_ids"]) == 2

    with Session(app.state.engine) as session:
        samples = session.exec(select(SampleIndex).where(SampleIndex.shard_id.in_(body["shard_ids"]))).all()
        assert samples
        regime_sample = next(sample for sample in samples if sample.sample_metadata["capability_id"] == "regime_switching")
        intermittent_sample = next(sample for sample in samples if sample.sample_metadata["capability_id"] == "predictable_intermittency")
        cut_points = regime_sample.sample_metadata["latent_params"]["cut_points"]
        assert cut_points
        assert all(type(point) is int for point in cut_points)
        assert regime_sample.sample_metadata["latent_params"]["forecast_switch"] == 1
        assert sum(point < 168 for point in cut_points) >= 2
        assert regime_sample.sample_metadata["latent_params"]["future_switch_at"] >= 168
        pulse_centers = intermittent_sample.sample_metadata["latent_params"]["pulse_centers"]
        assert sum(center < 168 for center in pulse_centers) >= 2
        assert any(center >= 168 for center in pulse_centers)


def test_new_synthetic_capabilities_generate_expected_metadata(app, client):
    seasonal_response = client.post(
        "/synthetic/shards",
        json={
            "name": "time varying smoke",
            "capabilities": ["time_varying_seasonality"],
            "context_length": 168,
            "horizon": 24,
            "sample_count": 2,
            "intensity": 4,
            "season_length": 8,
            "target_dim": 1,
            "seed": 23,
            "frequency": "h",
        },
    )
    hierarchy_response = client.post(
        "/synthetic/shards",
        json={
            "name": "hierarchy smoke",
            "capabilities": ["hierarchical_coherence"],
            "context_length": 365,
            "horizon": 28,
            "sample_count": 2,
            "intensity": 4,
            "season_length": 8,
            "target_dim": 3,
            "seed": 24,
            "frequency": "d",
        },
    )

    assert seasonal_response.status_code == 200, seasonal_response.text
    assert hierarchy_response.status_code == 200, hierarchy_response.text
    shard_ids = [*seasonal_response.json()["shard_ids"], *hierarchy_response.json()["shard_ids"]]
    assert len(shard_ids) == 2

    with Session(app.state.engine) as session:
        shards = session.exec(select(Shard).where(Shard.shard_id.in_(shard_ids))).all()
        assert {shard.capability_type for shard in shards} == {"time_varying_seasonality", "hierarchical_coherence"}
        seasonal = next(shard for shard in shards if shard.capability_type == "time_varying_seasonality")
        hierarchy = next(shard for shard in shards if shard.capability_type == "hierarchical_coherence")
        assert seasonal.target_dim == 1
        assert hierarchy.target_dim == 3
        assert seasonal.generation_config["requested_season_length"] == 8
        assert seasonal.generation_config["season_length"] == 24
        assert hierarchy.generation_config["requested_season_length"] == 8
        assert hierarchy.generation_config["season_length"] == 7

        samples = session.exec(select(SampleIndex).where(SampleIndex.shard_id.in_(shard_ids))).all()
        seasonal_sample = next(sample for sample in samples if sample.sample_metadata["capability_id"] == "time_varying_seasonality")
        hierarchy_sample = next(sample for sample in samples if sample.sample_metadata["capability_id"] == "hierarchical_coherence")
        assert seasonal_sample.sample_metadata["latent_params"]["amplitude_delta_mean"] > 0
        assert hierarchy_sample.sample_metadata["latent_params"]["hierarchy"] == "target_0=sum(target_1:)"
        assert hierarchy_sample.sample_metadata["season_length"] == 7
        assert hierarchy_sample.sample_metadata["requested_season_length"] == 8
        assert hierarchy_sample.sample_metadata["realized_features"]["hierarchy_residual_mean_abs"] < 1e-8
