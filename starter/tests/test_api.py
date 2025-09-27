# tests/test_api.py
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import app


client = TestClient(app)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Income Prediction API 🚀"}


def test_post_predict_low_income():
    sample = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "11th",
        "education-num": 7,
        "marital-status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]
    assert response.json()["prediction"] == "<=50K"


def test_post_predict_high_income():
    sample = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 99999,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }

    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]
    assert response.json()["prediction"] == ">50K"
