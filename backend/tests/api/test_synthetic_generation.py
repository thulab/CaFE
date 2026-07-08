from sqlmodel import Session, select

from app.models.benchmark import CapabilityBlock, CapabilityBlockShard, Track
from app.models.dataset import Shard
from app.models.sample import SampleIndex


def test_synthetic_capabilities_and_generation_materialize_shards(app, client):
    catalog = client.get("/synthetic/capabilities")
    assert catalog.status_code == 200
    capability_ids = {item["capability_id"] for item in catalog.json()["items"]}
    assert {"trend", "common_factor", "covariate_response", "time_varying_seasonality", "hierarchical_coherence"}.issubset(capability_ids)
    trend_capability = next(item for item in catalog.json()["items"] if item["capability_id"] == "trend")
    assert trend_capability["label_i18n"]["zh-CN"] == "趋势"
    assert trend_capability["limits"]["context_length"]["min"] == 16

    response = client.post(
        "/synthetic/shards",
        json={
            "name": "synthetic smoke",
            "capabilities": ["trend", "covariate_response"],
            "context_length": 16,
            "horizon": 4,
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
        assert covariate.target_dim == 2
        assert covariate.covariate_columns == ["weather", "event"]
        assert covariate.generation_config["intensity"] == 3
        assert covariate.generation_config["difficulty"] == 3
        assert covariate.generation_config["anchor_mode"] == "profile_calibrated"
        assert covariate.generation_config["anchor_profiles"] == [
            "m4_hourly_daily_96ctx",
            "m4_hourly_daily_168ctx",
            "m4_hourly_weekly",
            "us_births_weekly",
            "us_births_annual_diagnostic",
        ]

        trend_sample = session.exec(select(SampleIndex).where(SampleIndex.shard_id == trend.shard_id)).first()
        assert trend_sample is not None
        trend_metadata = trend_sample.sample_metadata
        assert trend_metadata["intensity"] == 3
        assert trend_metadata["difficulty"] == 3
        assert trend_metadata["latent_params"]["generator_version"] == "v2-pilot"
        assert trend_metadata["latent_params"]["acceptance"]["accepted"] is True
        assert trend_metadata["latent_params"]["acceptance"]["validation"]["schema_version"] == "synthetic_post_generation_validation.v1"
        assert "trend_strength" in trend_metadata["latent_params"]["acceptance"]["validation"]["target_features"]
        assert "trend_strength" in trend_metadata["realized_features"]
        assert "nonlinear_lag1_gain" in trend_metadata["realized_features"]

        sample = session.exec(select(SampleIndex).where(SampleIndex.shard_id == covariate.shard_id)).first()
        assert sample is not None
        assert sample.sample_metadata["capability_id"] == "covariate_response"
        assert "avg_abs_covariate_target_corr" in sample.sample_metadata["realized_features"]

    sample_list = client.get(f"/shards/{body['shard_ids'][1]}/samples").json()
    sample_id = sample_list["items"][0]["sample_id"]
    preview = client.get(f"/samples/{sample_id}/preview").json()
    assert len(preview["target_history"]) == 16
    assert len(preview["target_future"]) == 4
    assert len(preview["target_history"][0]) == 2
    assert len(preview["history_cov"][0]) == 2
    assert len(preview["future_cov"]) == 4


def test_track_from_shards_groups_synthetic_capabilities(app, client):
    generated = client.post(
        "/synthetic/shards",
        json={
            "name": "synthetic grouped",
            "capabilities": ["trend", "common_factor"],
            "context_length": 16,
            "horizon": 4,
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


def test_regime_capabilities_generate_json_serializable_metadata(app, client):
    response = client.post(
        "/synthetic/shards",
        json={
            "name": "regime smoke",
            "capabilities": ["regime_switching", "coherent_regime_shift"],
            "context_length": 16,
            "horizon": 4,
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
        coherent_sample = next(sample for sample in samples if sample.sample_metadata["capability_id"] == "coherent_regime_shift")
        cut_points = regime_sample.sample_metadata["latent_params"]["cut_points"]
        assert cut_points
        assert all(type(point) is int for point in cut_points)
        assert regime_sample.sample_metadata["latent_params"]["forecast_switch"] == 1
        assert coherent_sample.sample_metadata["latent_params"]["shift_at"] >= 16


def test_new_synthetic_capabilities_generate_expected_metadata(app, client):
    response = client.post(
        "/synthetic/shards",
        json={
            "name": "new capability smoke",
            "capabilities": ["time_varying_seasonality", "hierarchical_coherence"],
            "context_length": 24,
            "horizon": 6,
            "sample_count": 2,
            "intensity": 4,
            "season_length": 8,
            "target_dim": 3,
            "seed": 23,
            "frequency": "h",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["shard_ids"]) == 2

    with Session(app.state.engine) as session:
        shards = session.exec(select(Shard).where(Shard.shard_id.in_(body["shard_ids"]))).all()
        assert {shard.capability_type for shard in shards} == {"time_varying_seasonality", "hierarchical_coherence"}
        seasonal = next(shard for shard in shards if shard.capability_type == "time_varying_seasonality")
        hierarchy = next(shard for shard in shards if shard.capability_type == "hierarchical_coherence")
        assert seasonal.target_dim == 1
        assert hierarchy.target_dim == 3

        samples = session.exec(select(SampleIndex).where(SampleIndex.shard_id.in_(body["shard_ids"]))).all()
        seasonal_sample = next(sample for sample in samples if sample.sample_metadata["capability_id"] == "time_varying_seasonality")
        hierarchy_sample = next(sample for sample in samples if sample.sample_metadata["capability_id"] == "hierarchical_coherence")
        assert seasonal_sample.sample_metadata["latent_params"]["amplitude_delta_mean"] > 0
        assert hierarchy_sample.sample_metadata["latent_params"]["hierarchy"] == "target_0=sum(target_1:)"
        assert hierarchy_sample.sample_metadata["realized_features"]["hierarchy_residual_mean_abs"] < 1e-8
