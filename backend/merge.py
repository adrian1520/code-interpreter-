from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MergeResults:
    """Collects stable result files produced by category processors."""

    RESULT_FILES = ("summary.json", "devices.json", "tables.json", "symbols.json", "messages.json", "graph.json")

    def run(self, output_dir: str | Path) -> dict[str, Any]:
        root = Path(output_dir)
        merged: dict[str, Any] = {}
        for name in self.RESULT_FILES:
            path = root / name
            if path.is_file():
                merged[name] = json.loads(path.read_text(encoding="utf-8"))
        return merged
