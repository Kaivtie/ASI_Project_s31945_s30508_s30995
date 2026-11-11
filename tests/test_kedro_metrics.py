import json
import runpy

import pytest


@pytest.mark.parametrize("value", [-0.1, 1.1, True, "0.5", None, float("nan"), float("inf")])
def test_kedro_metrics_reject_invalid_values(repo_root, tmp_path, value):
    main = runpy.run_path(str(repo_root / "scripts/check_metrics.py"))["main"]
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps({"accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": value}),
        encoding="utf-8",
    )
    assert main([str(path), "--kedro"]) == 1


def test_kedro_metrics_accept_valid_values(repo_root, tmp_path):
    main = runpy.run_path(str(repo_root / "scripts/check_metrics.py"))["main"]
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps({"accuracy": 1, "precision": 0, "recall": 0.6, "f1": 0.5}), encoding="utf-8"
    )
    assert main([str(path), "--kedro"]) == 0


def test_kedro_metrics_reject_missing_metric(repo_root, tmp_path):
    main = runpy.run_path(str(repo_root / "scripts/check_metrics.py"))["main"]
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"accuracy": 0.8}), encoding="utf-8")
    assert main([str(path), "--kedro"]) == 1
