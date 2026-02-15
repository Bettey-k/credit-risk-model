import sys
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data_processing import build_feature_pipeline


# Fix import path AFTER imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------
# Load and Process Data
# ---------------------------------------------------------
def load_data(path="data/processed/data_with_labels.csv"):

    # -----------------------------------------------------
    # If dataset is missing (e.g., in CI), create dummy data
    # -----------------------------------------------------
    if not os.path.exists(path):
        df = pd.DataFrame({
            "Amount": [1000, 2000, 1500],
            "Value": [1000, 2000, 1500],
            "TransactionStartTime": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "CurrencyCode": ["UGX", "UGX", "UGX"],
            "CountryCode": ["256", "256", "256"],
            "ProviderId": ["A", "A", "B"],
            "ProductId": ["P1", "P2", "P3"],
            "ProductCategory": ["Food", "Food", "Tech"],
            "ChannelId": ["Android", "Android", "Web"],
            "PricingStrategy": [1, 1, 2],
            "CustomerId": [101, 101, 202],
            "is_high_risk": [0, 1, 0],
        })
    else:
        df = pd.read_csv(path)

    y = df["is_high_risk"]
    X = df.drop(columns=["is_high_risk"])

    feature_pipeline = build_feature_pipeline()
    X = feature_pipeline.fit_transform(X)

    return X, y



# ---------------------------------------------------------
def split_data():
    X, y = load_data()
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------------------------------------------
# Models to Evaluate
# ---------------------------------------------------------
def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=800),
        "RandomForest": RandomForestClassifier(n_estimators=200),
        "XGBoost": XGBClassifier(eval_metric="logloss"),
    }


# ---------------------------------------------------------
# Metric Calculation
# ---------------------------------------------------------
def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


# ---------------------------------------------------------
# Train Models
# ---------------------------------------------------------
def train_models():
    mlflow.set_experiment("credit-risk-task5")

    X_train, X_test, y_train, y_test = split_data()
    models = get_models()

    best_model = None
    best_name = None
    best_score = 0

    for name, model in models.items():

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model),
        ])

        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            scores = evaluate(y_test, y_pred, y_prob)

            mlflow.log_metrics(scores)
            print(f"\n{name}: {scores}")

            if scores["f1"] > best_score:
                best_score = scores["f1"]
                best_model = pipe
                best_name = name

            # SHAP Explainability
            try:
                explainer = shap.Explainer(pipe["clf"])
                shap_values = explainer(X_test)

                plt.figure()
                shap.summary_plot(shap_values, X_test, show=False)
                shap_path = f"shap_summary_{name}.png"
                plt.savefig(shap_path, bbox_inches="tight")
                mlflow.log_artifact(shap_path)

            except Exception as e:
                print(f"SHAP failed for {name}: {e}")

    print(f"\nBest model = {best_name}, F1 = {best_score}")

    os.makedirs("models", exist_ok=True)
    mlflow.sklearn.save_model(best_model, "models/best_model")

    print("\nBest model saved to: models/best_model")


# ---------------------------------------------------------
# Run Training
# ---------------------------------------------------------
if __name__ == "__main__":
    train_models()
