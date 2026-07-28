from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz
import pdfplumber

from .classifier import PDFPageClassifier, PAGE_CATEGORIES
from .manifest import CATEGORY_TO_DIR, PrepareManifest
from .pdf_splitter import PDFSplitter


class PreparePDF:
    """One-time PDF preparation: classify pages and split without rasterizing or OCR."""

    def __init__(self, classifier: PDFPageClassifier | None = None, splitter: PDFSplitter | None = None) -> None:
        self.classifier = classifier or PDFPageClassifier()
        self.splitter = splitter or PDFSplitter()

    def run(self, input_pdf: str | Path, output_dir: str | Path) -> PrepareManifest:
        source = Path(input_pdf)
        prepare_dir = Path(output_dir) / "prepare"
        prepare_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = prepare_dir / "manifest.json"
        if manifest_path.is_file():
            return PrepareManifest.from_path(manifest_path)

        with fitz.open(source) as doc:
            page_count = len(doc)
            tables_by_page = self._extract_table_signals(source, page_count)
            assignments: dict[str, list[int]] = {category: [] for category in PAGE_CATEGORIES}
            page_categories: dict[int, str] = {}
            for index in range(page_count):
                category = self.classifier.classify(doc[index], tables_by_page.get(index, []))
                assignments[category].append(index + 1)
                page_categories[index + 1] = category

        output_paths = self.splitter.split(source, prepare_dir, assignments)
        per_category_position: dict[str, int] = {category: 0 for category in PAGE_CATEGORIES}
        page_map: dict[str, dict[str, Any]] = {}
        for page_number in range(1, page_count + 1):
            category = page_categories[page_number]
            per_category_position[category] += 1
            page_map[str(page_number)] = {
                "category": category,
                "prepared_pdf": str(output_paths[category]),
                "prepared_page_index": per_category_position[category] - 1,
            }
        manifest = PrepareManifest(
            pages=page_count,
            categories={CATEGORY_TO_DIR[k]: v for k, v in assignments.items() if v},
            page_map=page_map,
        )
        manifest.write(manifest_path)
        return manifest

    def _extract_table_signals(self, source: Path, page_count: int) -> dict[int, list[list[list[Any]]]]:
        tables: dict[int, list[list[list[Any]]]] = {i: [] for i in range(page_count)}
        try:
            with pdfplumber.open(source) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        tables[i] = page.extract_tables() or []
                    except Exception:
                        tables[i] = []
        except Exception:
            pass
        return tables
