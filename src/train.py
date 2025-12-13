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

# ------------------------------------------------
# Load processed data with proxy label
# ------------------------------------------------
def load_data(path="data/processed/data_with_labels.csv"):
    df = pd.read_csv(path)
    
    # target
    y = df["is_high_risk"]

    # drop label
    df = df.drop(columns=["is_high_risk"])

    # build pipeline
    pipeline = build_feature_pipeline()

    # transform raw data → engineered features
    X = pipeline.fit_transform(df)

    return X, y

# ------------------------------------------------
# Train/Test Split
# ------------------------------------------------
def split_data(test_size=0.2, random_state=42):
    X, y = load_data()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ------------------------------------------------
# Define models to test
# ------------------------------------------------
def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }


# ------------------------------------------------
# Hyperparameter grids
# ------------------------------------------------
def get_param_grid():
    return {
        "LogisticRegression": {
            "clf__C": [0.1, 1, 10]
        },
        "RandomForest": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [5, 10, None]
        },
        "XGBoost": {
            "clf__n_estimators": [50, 100, 150],
            "clf__max_depth": [3, 5, 7]
        }
    }


# ------------------------------------------------
# Model Evaluation
# ------------------------------------------------
def evaluate_model(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


# ------------------------------------------------
# Train all models + Log to MLflow
# ------------------------------------------------
def train_models():
    mlflow.set_experiment("credit-risk-task5")

    X_train, X_test, y_train, y_test = split_data()
    models = get_models()
    param_grid = get_param_grid()

    best_model = None
    best_score = 0

    for model_name, model in models.items():

        pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("clf", model)
        ])

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid[model_name],
            n_iter=3,
            scoring="f1",
            cv=3,
            random_state=42
        )

        with mlflow.start_run(run_name=model_name):

            search.fit(X_train, y_train)

            best = search.best_estimator_
            y_pred = best.predict(X_test)
            y_prob = best.predict_proba(X_test)[:, 1]

            metrics = evaluate_model(y_test, y_pred, y_prob)

            # Log experiment details
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best, f"{model_name}_model")

            print(f"\n{model_name} scores:", metrics)

            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_model = best

    # Save best model
    mlflow.sklearn.save_model(best_model, "../models/best_model")
    print(f"\nBest model saved with F1 = {best_score}")


if __name__ == "__main__":
    train_models()
