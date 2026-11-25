
import argparse
from pathlib import Path

from asi_project.wandb_artifact import publish_model_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publikacja pakietu modelu produkcyjnego w W&B."
    )
    parser.add_argument(
        "--package-path",
        type=Path,
        default=Path("data/06_models/production"),
        help="Ścieżka do gotowego pakietu modelu.",
    )
    args = parser.parse_args()

    artifact_reference = publish_model_artifact(args.package_path)

    print(f"Opublikowano artefakt: {artifact_reference}")


if __name__ == "__main__":
    main()
