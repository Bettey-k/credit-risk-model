import sys
import os

# Add project root so Python can find src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, PredictionResponse


import mlflow.pyfunc
from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
import pandas as pd
import os

# Load MLflow best model
MODEL_PATH = "models/best_model"
model = mlflow.pyfunc.load_model(MODEL_PATH)

app = FastAPI(title="Credit Risk Prediction API")


@app.get("/")
def root():
    return {"message": "Credit Risk API is running!"}


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(payload: CustomerFeatures):

    # Convert input to dataframe
    data = pd.DataFrame([payload.dict()])

    # Predict probability
    proba = model.predict(data)[0]

    # binary classification with threshold 0.5
    label = 1 if proba >= 0.5 else 0

    return PredictionResponse(
        risk_probability=float(proba),
        is_high_risk=label
    )
