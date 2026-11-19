import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from asi_project.model_package import load_model_package, save_model_package


def test_model_package_roundtrip(tmp_path):
    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])

    from sklearn.pipeline import Pipeline

    model = Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))])
    model.fit(X, y)

    metadata = {
        "features": ["feature"],
        "target": "class",
        "positive_class": "p",
        "metrics": {"accuracy": 0.5},
        "git_sha": "test",
        "data_fingerprint": "test",
        "library_versions": {"scikit-learn": "test"},
    }

    package_path = tmp_path / "model_package"

    save_model_package(model, metadata, package_path)
    loaded_model, loaded_metadata = load_model_package(package_path)

    np.testing.assert_array_equal(model.predict(X), loaded_model.predict(X))
    assert loaded_metadata == metadata
    assert (package_path / "model.pkl").exists()
    assert (package_path / "metadata.json").exists()