from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pdf_self_engine import MNT_DATA, find_backend, load_module_from_path, main, run_backend_entry


def run_pdf_analysis(input_pdf: str, output_dir: str | None = None, backend_module: str | None = None) -> dict[str, Any]:
    """Uruchamia jedną deterministyczną analizę PDF i zwraca summary.json jako dict."""
    return main(input_pdf, output_dir, backend_module)


def auto_run(input_pdf: str, backend_name: str | None = None) -> dict[str, Any]:
    pdf = Path(input_pdf)
    if not pdf.is_absolute():
        pdf = MNT_DATA / pdf
    output = MNT_DATA / "output" / pdf.stem
    return main(str(pdf), str(output), backend_name)


__all__ = ["auto_run", "run_pdf_analysis", "load_module_from_path", "find_backend", "run_backend_entry"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap uruchamiany w ChatGPT Code Interpreter bez pytań interaktywnych.")
    parser.add_argument("input_pdf")
    parser.add_argument("--backend-name", default=None)
    parser.add_argument("--output-dir", default=None)
    ns = parser.parse_args()
    result = run_pdf_analysis(ns.input_pdf, ns.output_dir, ns.backend_name)
    print(json.dumps(result, ensure_ascii=False, indent=2))
