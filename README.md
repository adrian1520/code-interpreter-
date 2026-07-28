# PDF self-engine backend dla ChatGPT Code Interpreter

Repo zawiera produkcyjny silnik analizy PDF oraz prosty launcher zgodny z pracą w `/mnt/data` w ChatGPT Code Interpreter.

## Pliki

- `pdf_self_engine.py` — finalny pipeline engine, loader backendu przez `importlib`, walidacja i eksport artefaktów.
- `backend/prepare/prepare_pdf.py` — etap `PreparePDF`, który klasyfikuje każdą stronę i dzieli PDF bez rasteryzacji oraz bez OCR.
- `backend/prepare/classifier.py` — heurystyczna, odporna na polskie etykiety klasyfikacja stron (`TEXT`, `DRAWING`, `TABLE`, `LEGEND`, `SINGLE_LINE`, `PANEL_SCHEDULE`, `MIXED`, `UNKNOWN`).
- `backend/prepare/pdf_splitter.py` — zapis przygotowanych PDF-ów kategorii z zachowaniem oryginalnych stron PDF.
- `backend/prepare/manifest.py` — kontrakt manifestu przygotowania.
- `backend/processors/` — lekkie adaptery procesorów kategorii gotowe do przyszłego zrównoleglenia.
- `backend/pipeline.py`, `backend/merge.py` — modułowa orkiestracja i scalanie artefaktów.
- `pdf_backend_launcher.py` — minimalny bootstrap do automatycznego uruchamiania analizy z poziomu Code Interpreter.
- `bootstrap.xml`, `config.json` — deklaracje runtime i discovery backendu.
- `pipeline.py`, `pipeline_02.py` — wcześniejsze wersje robocze pozostawione dla kompatybilności.

## Kontrakt wejścia

- `input_pdf`: absolutna ścieżka PDF w `/mnt/data`, np. `/mnt/data/projekt.pdf`.
- `backend_module`: absolutna ścieżka backendu Python w `/mnt/data`, np. `/mnt/data/pdf_self_engine.py`; jeśli nie zostanie podana, launcher wybiera pierwszy plik zgodny z konwencją `*backend*.py`, `*engine*.py`, `pipeline*.py`, `pdf_*.py`.
- `output_dir`: absolutna ścieżka wyniku w `/mnt/data/output/<nazwa_pdf>`.

## Kontrakt wyjścia

Po jednym uruchomieniu powstaje jeden katalog wyniku z artefaktami:

```text
/mnt/data/output/<nazwa_pdf>/
├── prepare/
│   ├── manifest.json
│   ├── text/text.pdf
│   ├── drawings/drawings.pdf
│   ├── tables/tables.pdf
│   ├── legends/legends.pdf
│   ├── single_line/single_line.pdf
│   ├── panel_schedules/panel_schedules.pdf
│   ├── mixed/mixed.pdf
│   └── unknown/unknown.pdf
├── summary.json
├── graph.json
├── messages.json
├── tables.json
├── pages/
│   └── page_001.md
├── tables/
│   └── page_001_table_00.csv
├── ocr/
│   └── page_001.json
├── legend.json
├── symbols.json
├── devices.json
├── rooms.json
└── bom.json
```

`prepare/manifest.json` mapuje oryginalne numery stron na kategorię, przygotowany PDF oraz indeks strony w przygotowanym PDF-ie. `ocr/*.json` pojawia się tylko dla stron, dla których tekst wektorowy jest pusty lub zbyt słaby. JSON i Markdown są zapisywane w UTF-8 z zachowaniem polskich znaków oraz oryginalnych oznaczeń technicznych.

## Dwustopniowy pipeline

Silnik wykonuje etapy w stałej kolejności:

1. `prepare_pdf` — jednorazowa klasyfikacja i podział PDF do `prepare/`.
2. `load_pdf` — ładowanie wyłącznie manifestu i przygotowanych PDF-ów.
3. `extract_vector_text`
4. `render_pages`
5. `extract_tables`
6. `preprocess`
7. `ocr_pages`
8. `detect_legend`
9. `extract_bom_tables`
10. `extract_circuit_numbers`
11. `extract_rooms`
12. `detect_symbols`
13. `detect_devices`
14. `map_symbols_to_rooms`
15. `build_graph`
16. `validate`
17. `export`

Po etapie `prepare_pdf` downstream nie analizuje ponownie pełnego dokumentu źródłowego. Procesory korzystają z przygotowanych PDF-ów kategorii i `manifest.json`, zachowując oryginalną numerację stron w wynikach.

## Uruchomienie w Code Interpreterze

```python
from pdf_backend_launcher import run_pdf_analysis
result = run_pdf_analysis("/mnt/data/projekt.pdf")
result
```

Alternatywnie z jawnym backendiem i katalogiem wyjściowym:

```python
from pdf_backend_launcher import run_pdf_analysis
result = run_pdf_analysis(
    "/mnt/data/projekt.pdf",
    "/mnt/data/output/projekt",
    "/mnt/data/pdf_self_engine.py",
)
result
```

## Przykład CLI

```bash
python /mnt/data/pdf_backend_launcher.py /mnt/data/projekt.pdf --backend-name /mnt/data/pdf_self_engine.py --output-dir /mnt/data/output/projekt
```

## Zasady bezpieczeństwa i jakości

- Runtime wymaga ścieżek absolutnych w `/mnt/data` i nie zapisuje artefaktów poza `/mnt/data`.
- Etap przygotowania nie rasteryzuje stron i nie uruchamia OCR.
- Braki danych są raportowane jawnie w `warnings` albo `errors` w `summary.json` oraz `messages.json`.
- Tabele są ekstrahowane przez `pdfplumber`, strony renderowane przez PyMuPDF, preprocessing obrazu przez OpenCV/NumPy, eksport tabel przez Pandas, dopasowania tekstowe przez RapidFuzz, a graf dokumentu przez NetworkX.
