import json
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

from asi_project.model_package import (
    create_model_metadata,
    save_model_package,
)
from asi_project.pipelines.data_science.nodes import (
    evaluate_model,
    prepare_mushroom_data,
)

model_directory = Path("data/06_models/s3-catboost")
model_paths = list(model_directory.glob("*/predictor.pkl"))
model_path = max(model_paths, key=lambda path: path.stat().st_mtime).parent

model = TabularPredictor.load(str(model_path))

data = pd.read_csv("data/02_intermediate/mushrooms.csv")
_, X_test, _, y_test = prepare_mushroom_data(data)

metrics = evaluate_model(model, X_test, y_test)
metadata = create_model_metadata("CatBoost", metrics)

package_path = save_model_package(
    model,
    metadata,
    "data/06_models/production",
)

print("Pakiet:", package_path)
print("Metryki:", json.dumps(metrics, indent=2))
print("Metadata:", package_path / "metadata.json")
