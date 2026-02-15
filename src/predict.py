import os
import mlflow.pyfunc
import pandas as pd


MODEL_PATH = os.path.join("models", "best_model")


def load_model():
    """Load the trained MLflow model."""
    return mlflow.pyfunc.load_model(MODEL_PATH)


def predict_single(input_dict):
    """
    Predict risk probability from a dictionary of input features.
    Used by the FastAPI endpoint.
    """
    model = load_model()
    df = pd.DataFrame([input_dict])
    proba = model.predict(df)[0]
    return float(proba)


def predict(model, X):
    """
    Predict function required for unit tests.
    Signature: predict(model, X)

    Parameters:
        model: sklearn model with .predict()
        X (pd.DataFrame): DataFrame of features

    Returns:
        np.ndarray: predictions
    """
    return model.predict(X)
