# model.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: you could swap in RandomForestClassifier here if you prefer.
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Trains a machine learning model and returns it.
    """
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Run model inferences and return the predictions."""
    return model.predict(X)


# ---- Convenience: save/load artifacts (model, encoder, label binarizer) ----
def save_artifacts(model, encoder, lb, out_dir: str = "model") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(out_dir) / "model.pkl")
    joblib.dump(encoder, Path(out_dir) / "encoder.pkl")
    joblib.dump(lb, Path(out_dir) / "lb.pkl")


def load_artifacts(out_dir: str = "model"):
    model = joblib.load(Path(out_dir) / "model.pkl")
    encoder = joblib.load(Path(out_dir) / "encoder.pkl")
    lb = joblib.load(Path(out_dir) / "lb.pkl")
    return model, encoder, lb
