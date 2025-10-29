"""
Test suite for Car Price Prediction API

Tests verify:
- Health endpoint works
- Prediction endpoint accepts valid input
- Prediction endpoint returns correct format
- Prediction endpoint rejects invalid input
"""

import os
import pytest
from fastapi.testclient import TestClient

# Set model path for testing before importing the app
os.environ["MODEL_PATH"] = "models/model.pkl"

from src.main import api


# ==============================================================================
# TEST FIXTURES
# ==============================================================================

@pytest.fixture
def client():
    """
    Creates a test client for making requests to the API.
    TestClient simulates HTTP requests without starting a real server.
    """
    # Mock the model instead of loading the real one (50x faster)
    import src.main
    from unittest.mock import Mock

    mock_model = Mock()
    mock_model.predict.return_value = [15000.0]  # Fake prediction
    src.main.model = mock_model

    return TestClient(api)


@pytest.fixture
def sample_car_data():
    """
    Provides valid car data for testing predictions.
    """
    return {
        "Manufacturer": "Toyota",
        "Model": "Corolla",
        "Fuel type": "Petrol",
        "Engine size": 1.8,
        "Year of manufacture": 2018,
        "Mileage": 45000
    }


# ==============================================================================
# HEALTH CHECK TESTS
# ==============================================================================

def test_health_endpoint_returns_200(client):
    """
    Test: Health endpoint should return HTTP 200 (success).
    This is the most basic test - can we reach the server?
    """
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_returns_correct_structure(client):
    """
    Test: Health endpoint should return expected JSON structure.
    """
    response = client.get("/health")
    data = response.json()

    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] == "healthy"


# ==============================================================================
# PREDICTION ENDPOINT TESTS
# ==============================================================================

def test_predict_endpoint_accepts_valid_data(client, sample_car_data):
    """
    Test: Prediction endpoint should accept valid car data.
    This verifies the happy path (everything works correctly).
    """
    response = client.post("/predict", json=sample_car_data)
    assert response.status_code == 200


def test_predict_endpoint_returns_price(client, sample_car_data):
    """
    Test: Prediction should return a price in the expected format.
    """
    response = client.post("/predict", json=sample_car_data)
    data = response.json()

    # Check the response has the predicted_price_gbp field
    assert "predicted_price_gbp" in data

    # Check that the price is a number
    price = data["predicted_price_gbp"]
    assert isinstance(price, (int, float))

    # Sanity check: price should be positive
    assert price > 0


def test_predict_endpoint_rejects_missing_fields(client):
    """
    Test: Prediction endpoint should reject incomplete data.
    This validates that the API properly handles bad requests.
    """
    incomplete_data = {
        "Manufacturer": "Toyota",
        # Missing all other required fields!
    }

    response = client.post("/predict", json=incomplete_data)

    # Should return 422 (Unprocessable Entity - validation error)
    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_types(client):
    """
    Test: Prediction endpoint should reject wrong data types.
    """
    invalid_data = {
        "Manufacturer": "Toyota",
        "Model": "Corolla",
        "Fuel type": "Petrol",
        "Engine size": "not-a-number",  # Should be float!
        "Year of manufacture": 2018,
        "Mileage": 45000
    }

    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422


def test_predict_handles_edge_cases(client):
    """
    Test: Prediction should handle edge cases gracefully.
    Tests boundary conditions - very old cars, high mileage, etc.
    """
    edge_case_data = {
        "Manufacturer": "Ford",
        "Model": "Model T",
        "Fuel type": "Petrol",
        "Engine size": 2.9,
        "Year of manufacture": 1920,  # Very old car!
        "Mileage": 999999  # Very high mileage
    }

    response = client.post("/predict", json=edge_case_data)

    # Should still return 200 (even if prediction is weird)
    assert response.status_code == 200

    data = response.json()
    assert "predicted_price_gbp" in data


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

def test_full_prediction_workflow(client, sample_car_data):
    """
    Integration test: Complete workflow from health check to prediction.
    This simulates what a real user/frontend would do.
    """
    # Step 1: Check service is healthy
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    # Step 2: Make a prediction
    prediction = client.post("/predict", json=sample_car_data)
    assert prediction.status_code == 200

    # Step 3: Verify prediction format
    result = prediction.json()
    assert "predicted_price_gbp" in result
    assert isinstance(result["predicted_price_gbp"], (int, float))
    assert result["predicted_price_gbp"] > 0
