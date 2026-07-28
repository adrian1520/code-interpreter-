from __future__ import annotations

from pathlib import Path

import fitz

from .manifest import CATEGORY_TO_DIR


class PDFSplitter:
    def split(self, source_pdf: Path, output_prepare_dir: Path, assignments: dict[str, list[int]]) -> dict[str, Path]:
        output_prepare_dir.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, Path] = {}
        with fitz.open(source_pdf) as src:
            for category, dirname in CATEGORY_TO_DIR.items():
                category_dir = output_prepare_dir / dirname
                category_dir.mkdir(parents=True, exist_ok=True)
                out_path = category_dir / f"{dirname}.pdf"
                pages = assignments.get(category, [])
                if pages:
                    doc = fitz.open()
                    for page_number in pages:
                        doc.insert_pdf(src, from_page=page_number - 1, to_page=page_number - 1)
                    doc.save(out_path)
                    doc.close()
                else:
                    self._write_empty_pdf(out_path)
                output_paths[category] = out_path
        return output_paths

    def _write_empty_pdf(self, path: Path) -> None:
        path.write_bytes(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 0/Kids[]>>endobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer<</Size 3/Root 1 0 R>>\nstartxref\n108\n%%EOF\n")
