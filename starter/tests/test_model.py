# tests/test_model.py
from pathlib import Path
import numpy as np
import pandas as pd

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


# Helper to find the CSV robustly from tests/
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "census.csv"


def _tiny_dataset(df: pd.DataFrame, n: int = 400) -> pd.DataFrame:
    """Take a small, stratified-ish slice to keep tests fast."""
    # keep both classes if possible
    pos = df[df["salary"] == ">50K"].head(n // 2)
    neg = df[df["salary"] == "<=50K"].head(n // 2)
    out = pd.concat([pos, neg]).dropna()
    return out if len(out) >= 40 else df.head(n)


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def test_train_model_returns_estimator():
    df = pd.read_csv(DATA_PATH, skipinitialspace=True)
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip()
    df = _tiny_dataset(df)

    X, y, encoder, lb = process_data(
        df, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    model = train_model(X, y)

    # should have predict method and be fitted
    assert hasattr(model, "predict")
    assert hasattr(model, "coef_") or hasattr(model, "classes_")


def test_inference_output_is_numpy_array():
    df = pd.read_csv(DATA_PATH, skipinitialspace=True)
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip()
    df = _tiny_dataset(df)

    X, y, encoder, lb = process_data(
        df, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]


def test_compute_model_metrics_return_types():
    df = pd.read_csv(DATA_PATH, skipinitialspace=True)
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip()
    df = _tiny_dataset(df)

    X, y, encoder, lb = process_data(
        df, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)

    precision, recall, f1 = compute_model_metrics(y, preds)
    for val in (precision, recall, f1):
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0
