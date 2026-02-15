import os
import sys

from fastapi import FastAPI
import pandas as pd
import mlflow.pyfunc

from src.api.pydantic_models import CustomerFeatures, PredictionResponse


# Fix Python import path (must come AFTER imports according to flake8)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# Load MLflow model
MODEL_PATH = os.path.join("models", "best_model")
model = mlflow.pyfunc.load_model(MODEL_PATH)

app = FastAPI(title="Credit Risk Prediction API")


@app.get("/")
def root():
    return {"message": "Credit Risk API is running!"}


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(payload: CustomerFeatures):
    """
    Receives customer features, predicts probability of high risk.
    """

    # Convert input Pydantic model → DataFrame
    data = pd.DataFrame([payload.model_dump()])

    # Model outputs probability directly (because MLflow pyfunc)
    proba = float(model.predict(data)[0])

    # Classification threshold
    label = 1 if proba >= 0.5 else 0

    return PredictionResponse(
        probability=proba,
        is_high_risk=label
    )
