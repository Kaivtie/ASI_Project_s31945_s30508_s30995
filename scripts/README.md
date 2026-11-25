# scripts/

## Skrypty bazowe (od Sprintu 1)

| Skrypt | Rola |
|---|---|
| `download_data.py` | pobranie pełnych danych wybranego datasetu i odcisk SHA-256 |
| `make_ci_sample.py` | próbka CI `data/01_raw/sample.csv` |
| `check_env.py` | kontrola środowiska; `--autogluon` to pre-flight gotowości do Sprintu 3 |
| `check_dod.py` | automatyczny raport tylko dla Sprintu 1; w S2–S6 wskazuje dokumentację i nie ocenia projektu |
| `check_metrics.py`, `project_status.py`, `run_notebook.py` | kroki CI |
| `build_issues.py` | generowanie `github_issues/` z `docs/sprints/` |
| `smoke_datasets.py` | narzędzie prowadzącej: smoke prawdziwych źródeł wszystkich datasetów |

## Kolejne etapy (Sprinty 3–6)

> Opis kontraktów. Implementacje przygotowują zespoły w odpowiednim sprincie.

| Skrypt | Sprint | Kontrakt |
|---|---|---|
| smoke modelu | S3 | pobranie pakietu `production` z W&B → wczytanie w czystym procesie → predykcja 5 wierszy → sprawdzenie schematu odpowiedzi |
| wspólny smoke API | S4 → S6 | `/healthz` → tryb persistence → `/predict` z tokenem → kontrakt odpowiedzi → w trybie `sql` zapisany rekord, w trybie `disabled` brak wymogu zapisu; ten sam skrypt na kontenerze, na compose i na Cloud Run |

Skrypt wdrożenia (`deploy.sh`, Sprint 6) znajdzie się w katalogu `deploy/`.


## Model produkcyjny (Sprint 3)

Model produkcyjny jest przechowywany w W&B jako Artifact:

`bonsai-better-than-fable/Shrooms/mushroom-model:production`

### Publikacja pakietu modelu

```bash
wandb login
python scripts/publish_model.py --package-path data/06_models/production_clean
```

Publikacja wymaga gotowego pakietu modelu oraz dostępu do projektu W&B.

### Smoke test

```bash
python scripts/smoke_model.py
```

Skrypt pobiera artefakt `production`, wczytuje model bez ponownego
treningu, wykonuje predykcję dla 5 wierszy i sprawdza schemat wyniku.

Smoke test wymaga dostępu do W&B i jest wykonywany poza CI.

### Testy automatyczne

```bash
WANDB_MODE=disabled python -m pytest -q
pre-commit run --all-files
```

Testy automatyczne nie wymagają połączenia z W&B.
