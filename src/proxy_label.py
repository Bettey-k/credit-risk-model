import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ----------------------------------------------------
# STEP 1: Compute RFM Features
# ----------------------------------------------------

def compute_rfm(df):
    """Compute Recency, Frequency, Monetary metrics for each customer."""

    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Snapshot date = max transaction date
    snapshot_date = df["TransactionStartTime"].max()

    # Recency = days since last transaction
    recency = df.groupby("CustomerId")["TransactionStartTime"].max().apply(
        lambda x: (snapshot_date - x).days
    )

    # Frequency = number of transactions
    frequency = df.groupby("CustomerId")["TransactionId"].count()

    # Monetary = total spending (sum of Amount)
    monetary = df.groupby("CustomerId")["Amount"].sum()

    # Combine into a single RFM dataframe
    rfm = pd.DataFrame({
        "CustomerId": recency.index,
        "Recency": recency.values,
        "Frequency": frequency.values,
        "Monetary": monetary.values
    })

    return rfm


# ----------------------------------------------------
# STEP 2 & 3: Cluster and Identify High-Risk Customers
# ----------------------------------------------------

def create_proxy_label(df):
    """Create high-risk proxy target using RFM + KMeans clustering."""

    # Compute RFM features
    rfm = compute_rfm(df)

    # Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    # KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    # Identify which cluster is highest-risk:
    # low frequency + low monetary + high recency = disengaged customers
    cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

    # Sort clusters: Recency DESC, Frequency ASC, Monetary ASC
    risk_order = cluster_summary.sort_values(
        by=["Frequency", "Monetary", "Recency"],
        ascending=[True, True, False]
    )

    # The first row is the highest-risk cluster
    high_risk_cluster = risk_order.index[0]

    # Create binary label
    rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)

    return rfm[["CustomerId", "is_high_risk"]]
