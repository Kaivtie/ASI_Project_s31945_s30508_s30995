from functools import partial

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_model,
    log_sprint2_results,
    prepare_mushroom_data,
    run_autogluon,
    train_baseline,
    train_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_mushroom_data,
                inputs="model_input",
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="prepare_mushroom_data_node",
            ),
            node(
                func=train_baseline,
                inputs=["X_train", "y_train"],
                outputs="baseline_model",
                name="train_baseline_node",
            ),
            node(
                func=train_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:numeric_features",
                    "params:categorical_features",
                ],
                outputs="trained_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["baseline_model", "X_test", "y_test"],
                outputs="baseline_metrics",
                name="evaluate_baseline_node",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "X_test", "y_test"],
                outputs="model_metrics",
                name="evaluate_model_node",
            ),
            node(
                func=log_sprint2_results,
                inputs=[
                    "baseline_model",
                    "trained_model",
                    "baseline_metrics",
                    "model_metrics",
                ],
                outputs=None,
                name="log_sprint2_results_node",
            ),
        ]
    )


def create_autogluon_pipeline() -> Pipeline:
    experiments = [
        {
            "name": "s3-lightgbm",
            "model_type": "GBM",
            "time_limit": 120,
        },
        {
            "name": "s3-catboost",
            "model_type": "CAT",
            "time_limit": 120,
        },
    ]

    return pipeline(
        [
            node(
                func=partial(
                    run_autogluon,
                    experiment=experiment,
                ),
                inputs=["X_train", "X_test", "y_train", "y_test"],
                outputs=f"{experiment['name']}_metrics",
                name=f"{experiment['name']}_node",
            )
            for experiment in experiments
        ]
    )
