"""
test_api.py - Unit tests for FastAPI web endpoints
"""

from fastapi.testclient import TestClient
from web.app import app

client = TestClient(app)


def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Data Classification Using AI" in response.text
    assert "DecodeLabs" in response.text


def test_api_dataset_info():
    response = client.get("/api/dataset-info")
    assert response.status_code == 200
    data = response.json()
    assert "profile" in data
    assert data["profile"]["num_samples"] == 150
    assert "sample_data" in data


def test_api_elbow_analysis():
    response = client.get("/api/elbow-analysis?max_k=10")
    assert response.status_code == 200
    data = response.json()
    assert "optimal_k" in data
    assert "test_errors" in data


def test_api_train_evaluate():
    response = client.post("/api/train-evaluate", json={"k_neighbors": 5, "test_size": 0.20})
    assert response.status_code == 200
    data = response.json()
    assert data["k_used"] == 5
    assert "metrics" in data
    assert data["metrics"]["accuracy"] >= 0.90
    assert "benchmarks" in data


def test_api_predict():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_species"] == "setosa"
    assert data["confidence"] > 0.8
