from fastapi.testclient import TestClient

from app.main import create_app


def test_api_errors_use_unified_contract(client):
    response = client.get("/__test__/error-contract")

    assert response.status_code == 400
    assert response.json() == {
        "error_code": "invalid_test_input",
        "message": "invalid input",
        "details": {"field": "name"},
    }
