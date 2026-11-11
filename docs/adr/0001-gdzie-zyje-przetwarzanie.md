# ADR-1: Podział przetwarzania danych

- **Status:** proponowany, do review
- **Sprint:** S2
- **Autor:** Kacper Wierzba
- **Recenzent:** Ewelina Paczkowska

## Kontekst

Przenosimy baseline Mushroom z notebooka do Kedro. Chcemy używać tych samych
funkcji podczas treningu i późniejszej predykcji, bez uczenia preprocessingu na teście.

## Rozważone opcje

1. Funkcje przygotowania danych w `data.py`, a preprocessing wymagający uczenia w modelu.
2. Całe przetwarzanie, łącznie z kodowaniem i imputacją, w nodach przed splitem.

## Decyzja

Wybieramy opcję 1:

- `normalize_input` w `data.py` przycina tekst i zamienia puste napisy na braki.
  Działa na pojedynczym rekordzie i będzie używane również przez API.
- `filter_training_rows` w `data.py` usuwa rekordy bez targetu tylko przy treningu.
  Nie uruchamiamy tego kroku dla pojedynczej predykcji.
- Imputacja, skalowanie i OneHotEncoder pozostają w sklearn Pipeline z `model.py`.
  `fit` wykonujemy wyłącznie na zbiorze treningowym i zapisujemy cały model w pickle.

| Wielkość | Wartość | Źródło |
|---|---|---|
| Liczba cech numerycznych | 3 | `project.yaml`, `features.numeric` |
| Liczba cech kategorycznych | 17 | `project.yaml`, `features.categorical` |

## Odrzucone alternatywy

Odrzucamy uczenie imputacji i kodowania przed splitem, ponieważ statystyki testu
wpływałyby na model.

## Konsekwencje

Model powinien zawierać nauczony preprocessing. API będzie musiało wywołać `normalize_input`
przed `predict`; nie będzie usuwać rekordów bez targetu.

## Historia

| Data | Zmiana |
|---|---|
| 2026-11-09 | Propozycja decyzji do review Eweliny |
