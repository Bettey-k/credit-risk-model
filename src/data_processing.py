import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# --------------------------------------------
# 1. Extract Date Features
# --------------------------------------------

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


# --------------------------------------------
# 2. Aggregate Customer Transaction Behavior
# --------------------------------------------

class TransactionAggregator(BaseEstimator, TransformerMixin):
    """Create total, avg, std, txn_count per CustomerId."""

    def __init__(self, customer_col="CustomerId"):
        self.customer_col = customer_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        agg = df.groupby(self.customer_col)["Amount"].agg(
            total_amount="sum",
            avg_amount="mean",
            std_amount="std",
            txn_count="count"
        ).reset_index()

        df = df.merge(agg, on=self.customer_col, how="left")

        return df


# --------------------------------------------
# 3. Label Encoding for Categorical Features
# --------------------------------------------

class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    """Apply LabelEncoder to multiple categorical columns."""

    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            self.encoders[col] = le.fit(X[col].astype(str))
        return self

    def transform(self, X):
        df = X.copy()
        for col in df.columns:
            df[col] = self.encoders[col].transform(df[col].astype(str))
        return df


# --------------------------------------------
# 4. Build Full Feature Pipeline
# --------------------------------------------

def build_feature_pipeline():
    """Full preprocessing pipeline for Task 3."""

    numeric_features = [
        "Amount", "Value",
        "hour", "day", "month", "year",
        "total_amount", "avg_amount", "std_amount", "txn_count"
    ]

    categorical_features = [
        "CurrencyCode", "CountryCode", "ProviderId", "ProductId",
        "ProductCategory", "ChannelId", "PricingStrategy"
    ]

    from sklearn.impute import SimpleImputer

    numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("labelencoder", LabelEncoderWrapper())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    pipeline = Pipeline(steps=[
        ("datetime", DateFeatureExtractor()),
        ("aggregations", TransactionAggregator()),
        ("preprocessing", preprocessor)
    ])

    return pipeline
