import os
import mlflow.pyfunc
import pandas as pd


MODEL_PATH = os.path.join("models", "best_model")


def load_model():
    """Load the trained MLflow model."""
    return mlflow.pyfunc.load_model(MODEL_PATH)


def predict(input_dict):
    """
    Predict risk probability from a dictionary of input features.

    Parameters:
        input_dict (dict): One sample of features

    Returns:
        float: probability of high risk
    """
    model = load_model()
    df = pd.DataFrame([input_dict])
    proba = model.predict(df)[0]
    return float(proba)
