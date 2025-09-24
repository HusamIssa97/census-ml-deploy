# train_model.py
# Script to train machine learning model.
from __future__ import annotations
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, save_artifacts


DATA_PATH = "data/census.csv"   # make sure the file is here
ARTIFACT_DIR = "model"
SLICE_FILE = "slice_output.txt"

CAT_FEATURES: List[str] = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def load_clean_data(path: str) -> pd.DataFrame:
    """
    Load the census data and do light cleaning so we don't have to edit the CSV by hand.
    (The README suggests removing spaces; this does the equivalent safely.)
    """
    df = pd.read_csv(path, skipinitialspace=True)
    # strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
    return df


def write_slice_metrics(
    data: pd.DataFrame,
    feature: str,
    model,
    encoder,
    lb,
    cat_features: List[str],
    out_file: str = SLICE_FILE,
) -> None:
    """
    Compute and write metrics for every value of a single categorical feature.
    """
    lines = []
    for value in sorted(data[feature].unique()):
        df_slice = data[data[feature] == value]
        X_slice, y_slice, _, _ = process_data(
            df_slice,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X_slice)
        precision, recall, f1 = compute_model_metrics(y_slice, preds)
        lines.append(
            f"{feature} == {value!r} -> precision: {precision:.4f}, "
            f"recall: {recall:.4f}, f1: {f1:.4f}, n={len(df_slice)}"
        )

    Path(out_file).write_text("\n".join(lines) + "\n")
    print(f"[saved] slice metrics -> {out_file}")


def main() -> None:
    # 1) Load data
    df = load_clean_data(DATA_PATH)

    # 2) Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["salary"])

    # 3) Process data
    X_train, y_train, encoder, lb = process_data(
        train_df, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # 4) Train
    model = train_model(X_train, y_train)

    # 5) Evaluate on test
    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)
    print(f"Test metrics -> precision: {precision:.4f}  recall: {recall:.4f}  f1: {f1:.4f}")

    # 6) Save artifacts
    save_artifacts(model, encoder, lb, out_dir=ARTIFACT_DIR)
    print(f"[saved] model artifacts -> {ARTIFACT_DIR}/")

    # 7) Slice metrics (required by rubric) – pick one categorical feature, e.g. education
    write_slice_metrics(df, "education", model, encoder, lb, CAT_FEATURES, out_file=SLICE_FILE)


if __name__ == "__main__":
    main()
