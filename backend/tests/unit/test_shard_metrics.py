from app.services.metric_service import aggregate_metric


def test_shard_metrics_average_successful_samples_and_count_failures():
    result = aggregate_metric([{"mse": 1.0}, None, {"mse": 3.0}], metric_name="mse")

    assert result == {"value": 2.0, "success_count": 2, "failure_count": 1}


def test_all_failed_shard_produces_no_metric():
    result = aggregate_metric([None, None], metric_name="mse")

    assert result is None
