import pytest
from fastapi.testclient import TestClient
from app.main import app  

client = TestClient(app)

# Tests prédiction 1-correct, 2-incorrect, 3-invalid json

def test_predict_success():
    payload = {"features": [1.0, 2.0, 3.0]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == {"prediction": 6.0}

def test_predict_incorrect():
    payload = {"features": [1.0, 2.0, 3.0]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] != 999.0

def test_predict_invalid_json():
    payload = {"invalid_field": [3.5, 1.2, 4.9]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422