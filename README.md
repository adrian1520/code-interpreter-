# PDF self-engine backend dla ChatGPT Code Interpreter

Repo zawiera produkcyjny, jednoplikowy silnik analizy PDF oraz prosty launcher zgodny z pracą w `/mnt/data` w ChatGPT Code Interpreter.

## Pliki

- `pdf_self_engine.py` — finalny pipeline engine, loader backendu przez `importlib`, walidacja i eksport artefaktów.
- `pdf_backend_launcher.py` — minimalny bootstrap do automatycznego uruchamiania analizy z poziomu Code Interpreter.
- `pipeline.py`, `pipeline_02.py` — wcześniejsze wersje robocze pozostawione dla kompatybilności.

## Kontrakt wejścia

- `input_pdf`: absolutna ścieżka PDF w `/mnt/data`, np. `/mnt/data/projekt.pdf`.
- `backend_module`: absolutna ścieżka backendu Python w `/mnt/data`, np. `/mnt/data/pdf_self_engine.py`; jeśli nie zostanie podana, launcher wybiera pierwszy plik zgodny z konwencją `*backend*.py`, `*engine*.py`, `pipeline*.py`, `pdf_*.py`.
- `output_dir`: absolutna ścieżka wyniku w `/mnt/data/output/<nazwa_pdf>`.

## Kontrakt wyjścia

Po jednym uruchomieniu powstaje jeden katalog wyniku z artefaktami:

```text
/mnt/data/output/<nazwa_pdf>/
├── summary.json
├── graph.json
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

`ocr/*.json` pojawia się tylko dla stron, dla których tekst wektorowy jest pusty lub zbyt słaby. JSON i Markdown są zapisywane w UTF-8 z zachowaniem polskich znaków oraz oryginalnych oznaczeń technicznych.

## Deterministyczny pipeline

Silnik wykonuje etapy w stałej kolejności:

1. `load_pdf`
2. `extract_vector_text`
3. `render_pages`
4. `extract_tables`
5. `preprocess`
6. `ocr_pages`
7. `detect_legend`
8. `extract_bom_tables`
9. `extract_circuit_numbers`
10. `extract_rooms`
11. `detect_symbols`
12. `detect_devices`
13. `map_symbols_to_rooms`
14. `build_graph`
15. `validate`
16. `export`

Silnik używa tekstu wektorowego jako pierwszego źródła. OCR przez PaddleOCR jest wywoływany wyłącznie jako fallback i nie jest zastępowany własnym algorytmem OCR. Tabele są ekstrahowane przez `pdfplumber`, strony renderowane przez PyMuPDF, preprocessing obrazu przez OpenCV/NumPy, eksport tabel przez Pandas, dopasowania tekstowe przez RapidFuzz, a graf dokumentu przez NetworkX.

## Uruchomienie w Code Interpreterze

1. Wgraj PDF do `/mnt/data`, np. `/mnt/data/projekt.pdf`.
2. Wgraj lub skopiuj backend do `/mnt/data`, np. `/mnt/data/pdf_self_engine.py` i opcjonalnie `/mnt/data/pdf_backend_launcher.py`.
3. Uruchom single-shot bez pytań interaktywnych:

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
- Braki danych są raportowane jawnie w `warnings` albo `errors` w `summary.json`.
- Brak pseudokodu i brak interakcji sieciowej.
- Loader obsługuje `main(pdf_path, output_dir)`, `run(pdf_path, output_dir)` albo klasę `Pipeline` z metodą `run(ctx)`.
