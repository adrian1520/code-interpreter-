from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.merge import MergeResults
from backend.prepare.prepare_pdf import PreparePDF
from pdf_self_engine import Context, Pipeline as EnginePipeline


class PreparedPipeline:
    """Two-stage orchestration: prepare once, then process only prepared PDFs."""

    def run(self, input_pdf: str | Path, output_dir: str | Path) -> dict[str, Any]:
        manifest = PreparePDF().run(input_pdf, output_dir)
        ctx = EnginePipeline().run(Context(input_pdf=Path(input_pdf), output_dir=Path(output_dir)))
        merged = MergeResults().run(output_dir)
        merged.setdefault("manifest", manifest.to_dict())
        return merged or ctx.artifacts
