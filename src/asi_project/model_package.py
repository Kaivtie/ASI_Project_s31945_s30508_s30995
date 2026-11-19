import json
import pickle
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sklearn.pipeline import Pipeline

from asi_project.config import get_contract, load_project_config


def save_model_package(
    model,
    metadata: dict,
    package_path: str | Path,
) -> Path:
    package_path = Path(package_path)
    package_path.mkdir(parents=True, exist_ok=True)

    metadata_path = package_path / "metadata.json"

    if isinstance(model, Pipeline):
        with (package_path / "model.pkl").open("wb") as file:
            pickle.dump(model, file)
    else:
        source_path = Path(model.path)
        destination_path = package_path / "autogluon_model"
        if source_path.resolve() != destination_path.resolve():
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return package_path


def load_model_package(package_path: str | Path):
    package_path = Path(package_path)

    metadata = json.loads((package_path / "metadata.json").read_text(encoding="utf-8"))

    if (package_path / "model.pkl").exists():
        with (package_path / "model.pkl").open("rb") as file:
            model = pickle.load(file)
    else:
        from autogluon.tabular import TabularPredictor

        model = TabularPredictor.load(str(package_path / "autogluon_model"))

    return model, metadata


def create_model_metadata(model_name: str, metrics: dict) -> dict:
    config = load_project_config()
    contract = get_contract(config)

    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=config.root,
        text=True,
    ).strip()

    fingerprint_path = config.root / "data" / "01_raw" / "secondary_mushrooms" / "fingerprint.json"

    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))

    libraries = {}

    for package in ["scikit-learn", "pandas", "autogluon.tabular"]:
        try:
            libraries[package] = version(package)
        except PackageNotFoundError:
            libraries[package] = None

    return {
        "model_name": model_name,
        "dataset": config.dataset_id,
        "features": list(config.model_features),
        "target": contract.target,
        "positive_class": contract.positive_class,
        "metrics": metrics,
        "git_sha": git_sha,
        "data_fingerprint": fingerprint["sha256"],
        "python_version": sys.version.split()[0],
        "library_versions": libraries,
    }
