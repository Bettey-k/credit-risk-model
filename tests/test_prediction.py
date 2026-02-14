from src.predict import predict  # adjust if needed
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def test_prediction_shape():
    # Dummy trained model
    model = LogisticRegression()
    X = pd.DataFrame({
        "age": [25, 40, 35],
        "income": [50000, 80000, 60000]
    })
    y = np.array([0, 1, 0])
    model.fit(X, y)

    preds = predict(model, X)

    assert len(preds) == len(X)
