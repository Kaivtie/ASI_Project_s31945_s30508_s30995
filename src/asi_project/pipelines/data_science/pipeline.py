from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_model,
    prepare_mushroom_data,
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
        ]
    )
