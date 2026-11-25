
from pathlib import Path

import wandb

WANDB_ENTITY = "bonsai-better-than-fable"
WANDB_PROJECT = "Shrooms"
ARTIFACT_NAME = "mushroom-model"
ARTIFACT_ALIAS = "production"


def publish_model_artifact(
    package_path: str | Path,
    entity: str = WANDB_ENTITY,
    project: str = WANDB_PROJECT,
) -> str:
    """Publikuje gotowy pakiet modelu w W&B."""

    package_path = Path(package_path)

    if not package_path.is_dir():
        raise FileNotFoundError(package_path)

    if not (package_path / "metadata.json").is_file():
        raise FileNotFoundError("Brak metadata.json w pakiecie modelu")

    if not (
        (package_path / "model.pkl").is_file()
        or (package_path / "autogluon_model" / "predictor.pkl").is_file()
    ):
        raise FileNotFoundError("Brak zapisanego modelu w pakiecie")

    with wandb.init(
        entity=entity,
        project=project,
        job_type="model-publication",
        mode="online",
    ) as run:
        artifact = wandb.Artifact(
            name=ARTIFACT_NAME,
            type="model",
            description="Pakiet modelu produkcyjnego klasyfikacji grzybów",
        )

        artifact.add_dir(str(package_path))

        logged_artifact = run.log_artifact(
            artifact,
            aliases=["latest", ARTIFACT_ALIAS],
        )
        logged_artifact.wait()

    return f"{entity}/{project}/{ARTIFACT_NAME}:{ARTIFACT_ALIAS}"


def download_model_artifact(
    download_path: str | Path,
    entity: str = WANDB_ENTITY,
    project: str = WANDB_PROJECT,
) -> Path:
    """Pobiera pakiet modelu oznaczony aliasem production."""

    artifact_reference = f"{entity}/{project}/{ARTIFACT_NAME}:{ARTIFACT_ALIAS}"

    api = wandb.Api()
    artifact = api.artifact(
        artifact_reference,
        type="model",
    )

    path = artifact.download(root=str(download_path))

    return Path(path)
