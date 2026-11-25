
import json
from tempfile import TemporaryDirectory

from asi_project.config import get_contract, load_project_config
from asi_project.data import load_dataset, normalize_input
from asi_project.model_package import load_model_package
from asi_project.wandb_artifact import download_model_artifact


def main() -> None:
    config = load_project_config()
    contract = get_contract(config)

    with TemporaryDirectory() as directory:
        package_path = download_model_artifact(directory)

        print("Pobrano artefakt production z W&B.")

        model, metadata = load_model_package(package_path)

        print("Wczytano model:", metadata["model_name"])

        if metadata["features"] != list(config.model_features):
            raise ValueError("Niezgodna lista cech modelu")

        data = load_dataset(config, contract, mode="sample")
        X = normalize_input(data[metadata["features"]].head(5))

        if len(X) != 5:
            raise ValueError("Oczekiwano 5 wierszy wejściowych")

        predictions = model.predict(X)

        if predictions.shape != (5,):
            raise ValueError("Nieprawidłowy schemat predykcji")

        if not predictions.isin([0, 1]).all():
            raise ValueError("Nieprawidłowe klasy predykcji")

        result = {
            "predictions": [int(value) for value in predictions]
        }

        print(json.dumps(result, indent=2))
        print("Smoke test: OK")


if __name__ == "__main__":
    main()
