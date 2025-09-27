# live_test.py
import json
import traceback
from typing import Dict

import requests


URL = "https://census-ml-deploy.onrender.com/predict"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
TIMEOUT = 20


def post_predict(sample: Dict) -> None:
    print("\n=== POST /predict ===")
    print("Payload:", json.dumps(sample, indent=2))
    try:
        resp = requests.post(URL, headers=HEADERS, json=sample, timeout=TIMEOUT)
        print("Status code:", resp.status_code)

        # Success path
        if resp.ok:
            try:
                print("JSON:", resp.json())
            except json.JSONDecodeError:
                print("JSON decode failed; raw text follows:")
                print(resp.text)
            return

        # Error path: try to show JSON error if present, else raw text
        try:
            err = resp.json()
            print("Error JSON:", json.dumps(err, indent=2))
        except json.JSONDecodeError:
            print("Error (non-JSON) response body:")
            print(resp.text)

    except requests.RequestException as e:
        print("Request failed:", repr(e))
        traceback.print_exc()


if __name__ == "__main__":
    # Low-income-ish example
    sample_low = {
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
        "native-country": "United-States",
    }

    # High-income-ish example
    sample_high = {
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
        "native-country": "United-States",
    }

    print("Target URL:", URL)
    post_predict(sample_low)
    post_predict(sample_high)
