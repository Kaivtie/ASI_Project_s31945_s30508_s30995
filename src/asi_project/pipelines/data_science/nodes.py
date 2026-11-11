import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from asi_project.config import DatasetContract, get_contract, load_project_config
from asi_project.data import make_xy, split_train_test
from asi_project.model import build_baseline_pipeline, make_dummy


def prepare_mushroom_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    config = load_project_config()
    contract = get_contract(config)

    return prepare_train_test(
        data=data,
        contract=contract,
        features=list(config.model_features),
        test_size=config.test_size,
        random_state=config.random_state,
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
):

    model = build_baseline_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    model.fit(X_train, y_train)
    return model


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series):
    baseline = make_dummy()
    baseline.fit(X_train, y_train)
    return baseline


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }


def prepare_train_test(
    data: pd.DataFrame,
    contract: DatasetContract,
    features: list[str],
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = make_xy(data, contract, features)
    return split_train_test(
        X,
        y,
        contract,
        test_size=test_size,
        random_state=random_state,
    )
