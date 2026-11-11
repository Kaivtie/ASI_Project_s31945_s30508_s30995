import numpy as np
import pandas as pd

from asi_project.pipelines.data_science.nodes import (
    evaluate_model,
    train_baseline,
    train_model,
)


def sample_data():
    X = pd.DataFrame(
        {
            "cap-diameter": [2.1, 3.4, 5.2, 6.1, 2.8, 4.7, 5.5, 3.2],
            "cap-shape": ["x", "b", "x", "b", "x", "b", "x", "b"],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    return X, y


def test_train_model():
    X, y = sample_data()
    model = train_model(X, y, ["cap-diameter"], ["cap-shape"])

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_train_baseline():
    X, y = sample_data()
    baseline = train_baseline(X, y)

    predictions = baseline.predict(X)

    assert len(predictions) == len(y)


def test_evaluate_model():
    X, y = sample_data()
    model = train_model(X, y, ["cap-diameter"], ["cap-shape"])

    metrics = evaluate_model(model, X, y)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert all(0 <= value <= 1 for value in metrics.values())


def test_model_reproducibility():
    X, y = sample_data()

    model1 = train_model(X, y, ["cap-diameter"], ["cap-shape"])
    model2 = train_model(X, y, ["cap-diameter"], ["cap-shape"])

    predictions1 = model1.predict(X)
    predictions2 = model2.predict(X)

    np.testing.assert_array_equal(predictions1, predictions2)
    np.testing.assert_allclose(
        model1.predict_proba(X),
        model2.predict_proba(X),
    )


def test_missing_values_and_unknown_categories():
    X, y = sample_data()

    model = train_model(
        X,
        y,
        ["cap-diameter"],
        ["cap-shape"],
    )

    new_data = pd.DataFrame(
        {
            "cap-diameter": [np.nan, 4.5],
            "cap-shape": ["unknown_category", np.nan],
        }
    )

    predictions = model.predict(new_data)

    assert len(predictions) == 2
    assert set(np.unique(predictions)).issubset({0, 1})
