import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


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
# 2. Customer Transaction Aggregator
# -------------------------------------------------------------
class TransactionAggregator(BaseEstimator, TransformerMixin):
    """
    Create total_amount, avg_amount, std_amount, txn_count
    aggregated per CustomerId.
    """

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
# 3. Build Full Feature Engineering Pipeline
# -------------------------------------------------------------
def build_feature_pipeline():
    """Full preprocessing pipeline for Task 3."""

    # Numerical features (including engineered ones)
    numeric_features = [
        "Amount",
        "Value",
        "hour",
        "day",
        "month",
        "year",
        "total_amount",
        "avg_amount",
        "std_amount",
        "txn_count",
    ]

    # Categorical features
    categorical_features = [
        "CurrencyCode",
        "CountryCode",
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
        "PricingStrategy",
    ]

    # Numeric transformer
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical transformer
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    # Full pipeline
    pipeline = Pipeline(
        steps=[
            ("datetime", DateFeatureExtractor()),
            ("aggregations", TransactionAggregator()),
            ("preprocessing", preprocessor),
        ]
    )

    return pipeline
