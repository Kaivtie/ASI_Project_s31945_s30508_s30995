"""Rejestrowanie pipeline'ów Kedro (Sprint 2).

Po przygotowaniu ``create_pipeline()`` w podpakietach
``data_processing`` oraz ``data_science`` Kedro wykrywa je automatycznie.
Ten plik NIE zawiera implementacji nodów ani gotowego pipeline'u.
"""

from kedro.pipeline import Pipeline

from asi_project.pipelines.data_processing.pipeline import (
    create_pipeline as create_data_processing_pipeline,
)
from asi_project.pipelines.data_science.pipeline import (
    create_autogluon_pipeline,
)
from asi_project.pipelines.data_science.pipeline import (
    create_pipeline as create_data_science_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    data_processing = create_data_processing_pipeline()
    data_science = create_data_science_pipeline()

    default = data_processing + data_science
    experiments = default + create_autogluon_pipeline()

    return {
        "__default__": default,
        "data_processing": data_processing,
        "data_science": data_science,
        "experiments": experiments,
    }
