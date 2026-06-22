from app.services.metric_service import aggregate_metric


def test_task_metric_averages_successful_shards():
    result = aggregate_metric([{"mse": 2.0}, {"mse": 4.0}], metric_name="mse")

    assert result["value"] == 3.0
    assert result["success_count"] == 2


def test_unit_metric_averages_successful_tasks():
    result = aggregate_metric([{"mae": 1.0}, {"mae": 2.0}, None], metric_name="mae")

    assert result == {"value": 1.5, "success_count": 2, "failure_count": 1}
