import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# -------------------------------------------------------------
# 0. Column Validator (makes test_pipeline_missing_columns_fails pass)
# -------------------------------------------------------------
class ColumnValidator(BaseEstimator, TransformerMixin):
    """Ensure required columns exist before running the pipeline."""

    def __init__(self, required_columns):
        self.required_columns = required_columns

    def fit(self, X, y=None):
        missing = [col for col in self.required_columns if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return self

    def transform(self, X):
        # Fit already validated; no need to re-check
        return X


# -------------------------------------------------------------
# 1. Date Feature Extractor
# -------------------------------------------------------------
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract hour, day, month, year from TransactionStartTime."""

    def __init__(self, datetime_col="TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], errors="coerce")
        df["hour"] = df[self.datetime_col].dt.hour
        df["day"] = df[self.datetime_col].dt.day
        df["month"] = df[self.datetime_col].dt.month
        df["year"] = df[self.datetime_col].dt.year
        return df


# -------------------------------------------------------------
# 2. Transaction Behavior Aggregator
# -------------------------------------------------------------
class TransactionAggregator(BaseEstimator, TransformerMixin):
    """Aggregate customer behavior features."""

    def __init__(self, customer_col="CustomerId"):
        self.customer_col = customer_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        agg = (
            df.groupby(self.customer_col)["Amount"]
            .agg(
                total_amount="sum",
                avg_amount="mean",
                std_amount="std",
                txn_count="count",
            )
            .reset_index()
        )
        df = df.merge(agg, on=self.customer_col, how="left")
        return df


# -------------------------------------------------------------
# 3. Build Full Pipeline
# -------------------------------------------------------------
def build_feature_pipeline():
    numeric_features = [
        "Amount", "Value",
        "hour", "day", "month", "year",
        "total_amount", "avg_amount", "std_amount", "txn_count"
    ]

    categorical_features = [
        "CurrencyCode", "CountryCode", "ProviderId", "ProductId",
        "ProductCategory", "ChannelId", "PricingStrategy"
    ]

    required_columns = [
        "Amount", "Value", "TransactionStartTime", "CurrencyCode", "CountryCode",
        "ProviderId", "ProductId", "ProductCategory", "ChannelId",
        "PricingStrategy", "CustomerId"
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    pipeline = Pipeline(
        steps=[
            ("validate", ColumnValidator(required_columns)),
            ("datetime", DateFeatureExtractor()),
            ("aggregations", TransactionAggregator()),
            ("preprocessing", preprocessor),
        ]
    )

    return pipeline
