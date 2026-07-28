from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.prepare.manifest import PrepareManifest


class CategoryProcessor:
    category_dir: str = ""

    def pages_for_category(self, manifest: PrepareManifest) -> list[int]:
        return manifest.categories.get(self.category_dir, [])

    def pdf_for_category(self, prepare_dir: Path) -> Path:
        return prepare_dir / self.category_dir / f"{self.category_dir}.pdf"

    def run(self, prepare_dir: str | Path, manifest: PrepareManifest) -> dict[str, Any]:
        pdf = self.pdf_for_category(Path(prepare_dir))
        return {"category": self.category_dir, "pdf": str(pdf), "pages": self.pages_for_category(manifest)}
