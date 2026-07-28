from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CATEGORY_TO_DIR = {
    "TEXT": "text",
    "DRAWING": "drawings",
    "TABLE": "tables",
    "LEGEND": "legends",
    "SINGLE_LINE": "single_line",
    "PANEL_SCHEDULE": "panel_schedules",
    "MIXED": "mixed",
    "UNKNOWN": "unknown",
}

DIR_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_DIR.items()}


@dataclass
class PreparedPage:
    page_number: int
    category: str
    prepared_pdf: str
    prepared_page_index: int


@dataclass
class PrepareManifest:
    pages: int
    categories: dict[str, list[int]] = field(default_factory=dict)
    page_map: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages": self.pages,
            "categories": {k: v for k, v in self.categories.items() if v},
            "page_map": self.page_map,
        }

    @classmethod
    def from_path(cls, path: Path) -> "PrepareManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(pages=int(data.get("pages", 0)), categories=data.get("categories", {}), page_map=data.get("page_map", {}))

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
