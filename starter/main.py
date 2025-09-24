# main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from ml.model import load_artifacts, inference
from ml.data import process_data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "model"


app = FastAPI(
    title="Census Income Prediction API",
    description="Predict whether a person earns >50K or <=50K based on census data.",
    version="1.0.0",
)

# Load model and encoders at startup
model, encoder, lb = load_artifacts(str(ARTIFACT_DIR))


# Pydantic model for POST body
class CensusInput(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

    class Config:
        allow_population_by_field_name = True


@app.get("/")
def welcome():
    return {"message": "Welcome to the Census Income Prediction API 🚀"}


@app.post("/predict")
def predict(data: CensusInput):
    df = pd.DataFrame([data.dict(by_alias=True)])

    X, _, _, _ = process_data(
        df,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X)
    label = lb.inverse_transform(preds)[0]

    return {"prediction": label}
