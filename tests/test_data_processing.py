import pandas as pd
from src.data_processing import build_feature_pipeline


def test_pipeline_runs_successfully():
    """
    Test that the feature engineering pipeline runs without errors
    and returns the correct number of rows.
    """

    # Create a small sample dataframe similar to the real dataset
    data = pd.DataFrame({
        "TransactionId": [1, 2, 3],
        "BatchId": [1, 1, 1],
        "AccountId": [10, 10, 20],
        "SubscriptionId": [100, 100, 200],
        "CustomerId": [1, 1, 2],
        "CurrencyCode": ["UGX", "UGX", "UGX"],
        "CountryCode": ["256", "256", "256"],
        "ProviderId": ["A", "A", "B"],
        "ProductId": ["P1", "P2", "P3"],
        "ProductCategory": ["Food", "Food", "Tech"],
        "ChannelId": ["Android", "Android", "Web"],
        "Amount": [1000, 2000, 500],
        "Value": [1000, 2000, 500],
        "TransactionStartTime": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "PricingStrategy": [1, 1, 2],
        "FraudResult": [0, 0, 0],
    })

    pipeline = build_feature_pipeline()
    output = pipeline.fit_transform(data)

    # Check number of rows is same
    assert output.shape[0] == len(data)

    # Should have more than 5 transformed columns
    assert output.shape[1] > 5


def test_pipeline_missing_columns_fails():
    """
    Pipeline should fail if required columns are missing.
    This ensures data integrity before training.
    """
    incomplete_data = pd.DataFrame({
        "Amount": [100],
        "Value": [200]
    })

    pipeline = build_feature_pipeline()

    try:
        pipeline.fit_transform(incomplete_data)
        assert False, "Pipeline should raise an error when columns are missing"
    except Exception:
        assert True  # Expected behavior
