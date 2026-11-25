# Model Card

> Scaffold for Sprint 3 — do not implement before this sprint.
> Minimalną Model Card wypełnia zespół w Sprincie 3, a aktualizuje w Sprincie 6.

## Model

- Model: AutoGluon CatBoost.
- W&B Artifact: `bonsai-better-than-fable/Shrooms/mushroom-model:production`.
- Data przygotowania: 19.11.2026.
- Data publikacji w W&B: 25.11.2026.
- Wybór i pakowanie modelu: Ewelina.
- Publikacja i smoke test: Kasia.

## Przeznaczenie

Model służy do klasyfikacji grzybów na podstawie ich cech.

- Target: `class`.
- Klasa pozytywna: `p` (1) — grzyby trujące lub o nieznanej jadalności.
- Klasa negatywna: `e` (0) — grzyby jadalne.

Model został przygotowany w celach edukacyjnych.

## Dane

- Dataset: Secondary Mushroom (`secondary_mushrooms`).
- Źródło: UCI Machine Learning Repository.
- Liczba rekordów: 61 069.
- Liczba cech: 20.
- Podział danych: 80/20, `stratified_random`, `random_state=42`.
- Liczba rekordów testowych: 12 214.
- Fingerprint: `c0eb333df5747171cfc4356c966434b4e9ba1f099c4a0aa2c27f545853e6d203`.

## Metryki

| Metryka | DummyClassifier | CatBoost |
|---|---:|---:|
| Accuracy | 0.5549 | 0.9984 |
| Precision | 0.5549 | 0.9972 |
| Recall | 1.0000 | 0.9999 |
| F1 | 0.7138 | 0.9985 |

Główną metryką jest recall. Wybraliśmy CatBoost, ponieważ przy bardzo wysokim recall osiąga znacznie lepsze precision i F1 niż DummyClassifier.

Macierz pomyłek nie została zapisana w metadanych pakietu.

## Ograniczenia i ryzyka

- Dane są symulowane.
- Niektóre cechy zawierają dużo brakujących wartości.
- Model może błędnie klasyfikować grzyby.
- Model nie powinien być używany do podejmowania decyzji o spożyciu rzeczywistych grzybów.

## Wersje

- Python: 3.11.18
- scikit-learn: 1.7.2
- pandas: 2.3.3
- AutoGluon: 1.6.3
- CatBoost: 1.2.10 (środowisko smoke testu)
- W&B: 0.30.0
- Git SHA z metadanych: `985b6d420a8a7b76df0d35491f0f512c47f575d4`
