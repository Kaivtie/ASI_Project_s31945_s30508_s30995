"""Sprawdza plik metrics.json względem kontraktu (schemat, zakresy, dataset, cechy).

Użycie:
    python scripts/check_metrics.py metrics.json [--data-mode full|sample]
"""

import argparse
import math
import sys
from pathlib import Path

from asi_project.config import get_contract, load_project_config
from asi_project.metrics import METRIC_NAMES, load_metrics, validate_metrics_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walidacja metrics.json.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--data-mode", choices=["full", "sample"])
    parser.add_argument("--kedro", action="store_true")
    args = parser.parse_args(argv)

    if args.kedro and args.data_mode:
        parser.error("--data-mode dotyczy tylko formatu metryk notebooka.")

    record = load_metrics(args.path)
    if args.kedro:
        problems = []
        for name in METRIC_NAMES:
            value = record.get(name) if isinstance(record, dict) else None
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                problems.append(f"{name}: oczekiwana liczba z zakresu [0, 1].")
        if problems:
            for problem in problems:
                print(f"{args.path}: {problem}")
            return 1
        print(f"{args.path}: OK")
        return 0

    config = load_project_config()
    contract = get_contract(config)
    problems = validate_metrics_record(record, contract)
    if args.data_mode and record.get("data_mode") != args.data_mode:
        problems.append(f"data_mode: oczekiwano {args.data_mode!r}.")
    if problems:
        print(f"{args.path}: NIEZGODNY z kontraktem")
        for problem in problems:
            print(f"- {problem}")
        return 1
    models = record["models"]
    for name, values in models.items():
        print(f"{name:<20} f1={values['f1']:.3f} recall={values['recall']:.3f}")
    print(f"{args.path}: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
