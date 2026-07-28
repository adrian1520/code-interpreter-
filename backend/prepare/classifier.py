from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import fitz


PAGE_CATEGORIES = (
    "TEXT",
    "DRAWING",
    "TABLE",
    "LEGEND",
    "SINGLE_LINE",
    "PANEL_SCHEDULE",
    "MIXED",
    "UNKNOWN",
)


@dataclass(frozen=True)
class PageSignals:
    text_chars: int = 0
    line_count: int = 0
    drawing_count: int = 0
    table_count: int = 0
    legend_hits: int = 0
    panel_hits: int = 0
    single_line_hits: int = 0


class PDFPageClassifier:
    """Heuristic, Polish-aware page classifier used before expensive extraction."""

    LEGEND_TERMS = (
        "legenda", "oznaczenia", "symbole", "opis symboli", "wykaz symboli",
        "objaśnienia", "znaczenie symboli", "skrót", "ozn.",
    )
    TABLE_TERMS = (
        "tabela", "zestawienie", "wykaz", "lista", "ilość", "jedn.", "lp.",
        "materiał", "materiały", "opis", "typ", "producent",
    )
    PANEL_TERMS = (
        "rozdzielnica", "tablica", "szafa", "panel", "obwód", "zabezpieczenie",
        "schemat rozdzielnicy", "wykaz obwodów", "aparatura", "prąd", "moc",
    )
    SINGLE_LINE_TERMS = (
        "schemat jednokreskowy", "jednokreskowy", "single line", "schemat ideowy",
        "zasilanie", "wlz", "kabel", "odpływ", "dopływ", "l1", "l2", "l3", "pe", "n",
    )
    DRAWING_TERMS = (
        "rzut", "plan", "instalacja", "trasa", "widok", "przekrój", "detal",
        "rysunek", "skala", "kondygnacja", "poziom",
    )

    def classify(self, page: fitz.Page, tables: list[list[list[Any]]] | None = None) -> str:
        text = self._normalize(page.get_text("text") or "")
        lower = text.lower()
        drawings = len(page.get_drawings() or [])
        signals = PageSignals(
            text_chars=len(re.sub(r"\s+", "", text)),
            line_count=text.count("\n") + 1 if text else 0,
            drawing_count=drawings,
            table_count=len(tables or []),
            legend_hits=self._hits(lower, self.LEGEND_TERMS),
            panel_hits=self._hits(lower, self.PANEL_TERMS),
            single_line_hits=self._hits(lower, self.SINGLE_LINE_TERMS),
        )
        table_hits = self._hits(lower, self.TABLE_TERMS)
        drawing_hits = self._hits(lower, self.DRAWING_TERMS)

        if signals.legend_hits >= 2 or (signals.legend_hits and table_hits):
            return "LEGEND"
        if signals.panel_hits >= 2 and (signals.table_count or table_hits >= 2):
            return "PANEL_SCHEDULE"
        if signals.single_line_hits >= 2:
            return "SINGLE_LINE"
        if signals.table_count >= 2 or (signals.table_count and signals.text_chars > 80) or table_hits >= 3:
            return "TABLE"
        text_heavy = signals.text_chars >= 700 and signals.drawing_count < 30
        drawing_heavy = signals.drawing_count >= 50 or drawing_hits >= 2
        if text_heavy and drawing_heavy:
            return "MIXED"
        if drawing_heavy:
            return "DRAWING"
        if text_heavy or signals.text_chars >= 120:
            return "TEXT"
        return "UNKNOWN"

    def _hits(self, text: str, terms: tuple[str, ...]) -> int:
        return sum(1 for term in terms if term in text)

    def _normalize(self, text: str) -> str:
        return re.sub(r"[ \t]+", " ", (text or "").replace("\x00", " ")).strip()
