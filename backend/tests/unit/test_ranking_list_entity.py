from app.models.ranking import RankingEntry, RankingList


def test_ranking_list_supports_metric_and_policy_snapshot_entries():
    ranking = RankingList(track_id="track-1", default_metric_id="mse")
    entry = RankingEntry(
        ranking_list_id=ranking.ranking_list_id,
        track_id="track-1",
        metric_id="mae",
        policy="best_result",
        model_id="model-1",
        benchmarking_run_id="run-1",
        unit_id="unit-1",
        metric_value=0.2,
        rank=1,
    )

    assert ranking.default_policy == "latest_valid_result"
    assert ranking.supported_policies == ["latest_valid_result", "best_result"]
    assert entry.policy == "best_result"
    assert entry.metric_id == "mae"
