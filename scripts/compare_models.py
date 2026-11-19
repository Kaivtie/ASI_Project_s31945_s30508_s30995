
from pathlib import Path
import time
import tracemalloc
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from asi_project.pipelines.data_science.nodes import prepare_mushroom_data

data = pd.read_csv("data/02_intermediate/mushrooms.csv")
_, X_test, _, y_test = prepare_mushroom_data(data)

for name in ["s3-catboost", "s3-lightgbm"]:
    paths = list((Path("data/06_models") / name).glob("*/predictor.pkl"))
    model_path = max(paths, key=lambda path: path.stat().st_mtime).parent
    model = TabularPredictor.load(str(model_path))
    predictions = model.predict(X_test)

    print(name)
    print("accuracy:", accuracy_score(y_test, predictions))
    print("precision:", precision_score(y_test, predictions))
    print("recall:", recall_score(y_test, predictions))
    print("f1:", f1_score(y_test, predictions))

    print("f1:", f1_score(y_test, predictions))
    tracemalloc.start()
    start = time.perf_counter()
    model.predict(X_test)
    prediction_time = time.perf_counter() - start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("prediction_time_seconds:", round(prediction_time, 4))
    print("peak_python_memory_mb:", round(peak_memory / 1024**2, 2))