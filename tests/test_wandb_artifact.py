
from unittest.mock import MagicMock, patch

import pytest

from asi_project.wandb_artifact import (
    download_model_artifact,
    publish_model_artifact,
)


def test_publish_model_artifact(tmp_path):
    (tmp_path / "metadata.json").write_text(
        "{}", encoding="utf-8"
    )
    (tmp_path / "model.pkl").write_bytes(b"test-model")

    with (
        patch("asi_project.wandb_artifact.wandb.init") as mock_init,
        patch("asi_project.wandb_artifact.wandb.Artifact") as mock_artifact,
    ):
        run = mock_init.return_value.__enter__.return_value
        artifact = mock_artifact.return_value

        result = publish_model_artifact(tmp_path)

        artifact.add_dir.assert_called_once_with(str(tmp_path))
        run.log_artifact.assert_called_once_with(
            artifact,
            aliases=["latest", "production"],
        )
        run.log_artifact.return_value.wait.assert_called_once()

        assert result.endswith("mushroom-model:production")


def test_download_model_artifact(tmp_path):
    with patch("asi_project.wandb_artifact.wandb.Api") as mock_api:
        artifact = MagicMock()
        mock_api.return_value.artifact.return_value = artifact
        artifact.download.return_value = str(tmp_path)

        result = download_model_artifact(tmp_path)

        mock_api.return_value.artifact.assert_called_once_with(
            "bonsai-better-than-fable/Shrooms/mushroom-model:production",
            type="model",
        )
        artifact.download.assert_called_once_with(root=str(tmp_path))

        assert result == tmp_path


def test_publish_missing_metadata(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        publish_model_artifact(tmp_path)


def test_publish_missing_model(tmp_path):
    (tmp_path / "metadata.json").write_text(
        "{}", encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError, match="Brak zapisanego modelu"):
        publish_model_artifact(tmp_path)
