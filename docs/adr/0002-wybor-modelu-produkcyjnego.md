# ADR-2: Wybór modelu produkcyjnego

- **Status:** proponowany
- **Sprint:** S3
- **Autor:** Ewelina
- **Recenzent:** Kasia

## Kontekst

Celem Sprintu 3 jest porównanie modelu bazowego oraz co najmniej dwóch konfiguracji AutoGluon i wybór modelu produkcyjnego dla klasyfikacji grzybów.

Klasa pozytywna oznacza grzyb trujący (`p`, kodowany jako `1`). Dlatego szczególnie ważna jest wysoka wartość recall, ograniczająca ryzyko zaklasyfikowania grzyba trującego jako nietrującego. Jednocześnie model powinien osiągać dobrą precision i F1, a jego czas predykcji, rozmiar oraz zużycie pamięci muszą być akceptowalne.

Porównanie wykonano lokalnie, bez dostępu do W&B, na zbiorze testowym obejmującym 12 214 obserwacji i 20 cech wejściowych.

## Rozważone opcje

1. **DummyClassifier** — model referencyjny przewidujący najczęstszą klasę.
2. **LogisticRegression** — model bazowy z preprocessingiem sklearn.
3. **AutoGluon LightGBM** — konfiguracja `GBM`, limit treningu 120 sekund.
4. **AutoGluon CatBoost** — konfiguracja `CAT`, limit treningu 120 sekund.

## Decyzja

Wybieramy **AutoGluon CatBoost** jako model produkcyjny.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| DummyClassifier | 0,5549 | 0,5549 | 1,0000 | 0,7138 |
| LogisticRegression | 0,8476 | 0,8692 | 0,8539 | 0,8615 |
| LightGBM | 0,5549 | 0,5549 | 1,0000 | 0,7138 |
| **CatBoost** | **0,9984** | **0,9972** | **0,9999** | **0,9985** |

Dodatkowe pomiary:

| Wielkość | CatBoost | LightGBM | Skąd pochodzi |
|---|---:|---:|---|
| Czas predykcji 12 214 wierszy | 0,1668 s | 0,1200 s | Lokalny skrypt `compare_models.py` |
| Rozmiar artefaktu | 3,5 MB | 3,3 MB | Polecenie `du -sh` |
| Przyrost pamięci RSS przy wczytaniu | 11,48 MB | 11,48 MB | Lokalny pomiar `psutil` |

CatBoost ma nieco dłuższy czas predykcji i większy artefakt niż LightGBM, ale osiąga zdecydowanie wyższą precision i F1 przy recall bliskim 100%. Ten kompromis jest korzystny w zadaniu rozpoznawania grzybów trujących.

Pomiary pamięci są przybliżone i odnoszą się do procesu Pythona, a nie wyłącznie do pamięci zajmowanej przez sam model.

## Odrzucone alternatywy

- **DummyClassifier:** recall wynosi 1,0, ale model przewiduje klasę pozytywną dla wszystkich obserwacji, przez co ma niską precision.
- **LogisticRegression:** osiąga lepsze wyniki niż DummyClassifier, ale recall wynosi jedynie 0,8539.
- **LightGBM:** jest nieco szybszy i mniejszy niż CatBoost, jednak w lokalnym eksperymencie osiągnął precision 0,5549 i F1 0,7138, co odpowiada przewidywaniu wyłącznie klasy pozytywnej.

## Konsekwencje

Wybrany CatBoost zapewnia bardzo wysoką jakość klasyfikacji na zbiorze testowym. Kosztem tej decyzji jest nieco większy artefakt i dłuższy czas predykcji względem LightGBM.

Model jest zapisywany jako pakiet zawierający katalog AutoGluon oraz `metadata.json` z wymaganymi informacjami o cechach, klasie pozytywnej, metrykach, wersjach bibliotek, Git SHA i fingerprintem danych.

Pakiet można wczytać i wykorzystać do predykcji bez ponownego treningu. W kolejnych etapach będzie wykorzystywany przez smoke test oraz API.

Publikacja artefaktu w W&B z aliasem `production` i smoke test pobierający ten artefakt wymagają dostępności usługi oraz odpowiednich uprawnień.

## Historia

| Data | Zmiana |
|---|---|
| 2026-11-19 | Utworzenie ADR-2 i udokumentowanie wyboru CatBoost |
