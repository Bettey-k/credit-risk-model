import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data_processing import build_feature_pipeline


def load_data(path="data/processed/data_with_labels.csv"):
    df = pd.read_csv(path)
    y = df["is_high_risk"]
    X = df.drop(columns=["is_high_risk"])

    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(X)

    return X, y


def split_data():
    X, y = load_data()
    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_models():
    return {
        "LR": LogisticRegression(max_iter=500),
        "RF": RandomForestClassifier(),
        "XGB": XGBClassifier(eval_metric="logloss")
    }


def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }


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
            ("clf", model)
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

    # FORCE SAVE
    print(f"\nBest model = {best_name}, F1 = {best_score}")

    os.makedirs("models", exist_ok=True)
    mlflow.sklearn.save_model(best_model, "models/best_model")

    print("\nSaved best model to models/best_model")


if __name__ == "__main__":
    train_models()
