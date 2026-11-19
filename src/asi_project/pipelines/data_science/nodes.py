import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import pandas as pd
import wandb
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from asi_project.config import DatasetContract, get_contract, load_project_config
from asi_project.data import make_xy, split_train_test
from asi_project.model import build_baseline_pipeline, make_dummy

WANDB_ENTITY = "bonsai-better-than-fable"
WANDB_PROJECT = "Shrooms"


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


def log_sprint2_results(
    baseline_model,
    trained_model,
    baseline_metrics: dict,
    model_metrics: dict,
) -> None:
    if os.environ.get("WANDB_MODE") == "disabled":
        return

    project_config = load_project_config()

    common_parameters = {
        "dataset": project_config.dataset_id,
        "test_size": project_config.test_size,
        "random_state": project_config.random_state,
        "features": list(project_config.model_features),
    }

    log(
        name="s2-dummy",
        parameters={
            **common_parameters,
            "model_type": "DummyClassifier",
            "model_parameters": baseline_model.get_params(),
        },
        metrics=baseline_metrics,
    )

    log(
        name="s2-logistic-regression",
        parameters={
            **common_parameters,
            "model_type": "LogisticRegression",
            "model_parameters": trained_model.named_steps["classifier"].get_params(),
        },
        metrics=model_metrics,
    )


def log(
    name: str,
    parameters: dict,
    metrics: dict,
) -> None:
    git_sha, data_fingerprint = get_metadata()

    config = {
        **parameters,
        "git_sha": git_sha,
        "data_fingerprint": data_fingerprint,
    }

    with wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=name,
        config=config,
        job_type="evaluation",
    ) as run:
        run.log(metrics)


def get_metadata() -> tuple[str, str]:
    repo_root = Path(__file__).resolve().parents[4]

    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()

    fingerprint_path = repo_root / "data" / "01_raw" / "secondary_mushrooms" / "fingerprint.json"

    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))

    return git_sha, fingerprint["sha256"]


def train_autogluon(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    time_limit: int,
    model_path: str,
) -> TabularPredictor:
    train_data = X_train.copy()
    train_data["target"] = y_train

    predictor = TabularPredictor(
        label="target",
        problem_type="binary",
        positive_class=1,
        eval_metric="recall",
        path=model_path,
    )

    predictor.fit(
        train_data=train_data,
        hyperparameters={model_type: {}},
        time_limit=time_limit,
    )

    return predictor


def run_autogluon(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    experiment: dict,
) -> dict:
    project_config = load_project_config()

    model_path = project_config.root / "data" / "06_models" / experiment["name"] / uuid4().hex

    predictor = train_autogluon(
        X_train=X_train,
        y_train=y_train,
        model_type=experiment["model_type"],
        time_limit=experiment["time_limit"],
        model_path=str(model_path),
    )

    metrics = evaluate_model(predictor, X_test, y_test)

    if os.environ.get("WANDB_MODE") != "disabled":
        log(
            name=experiment["name"],
            parameters={
                **experiment,
                "dataset": project_config.dataset_id,
                "test_size": project_config.test_size,
                "random_state": project_config.random_state,
                "features": list(project_config.model_features),
                "eval_metric": "recall",
                "positive_class": 1,
                "hyperparameters": {
                    experiment["model_type"]: {},
                },
                "model_path": str(model_path),
            },
            metrics=metrics,
        )

    return metrics
