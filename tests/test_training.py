from src.train import load_data, split_data

def test_load_data():
    X, y = load_data()

    # X and y should have same number of rows
    assert X.shape[0] == len(y)

    # X must be a 2D numeric array
    assert X.ndim == 2

def test_split_data():
    X_train, X_test, y_train, y_test = split_data()
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
