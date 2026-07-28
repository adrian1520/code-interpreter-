from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import json
import re

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pdfplumber
import networkx as nx
from paddleocr import PaddleOCR
from rapidfuzz import process, fuzz


# -----------------------------
# Data model
# -----------------------------

@dataclass
class DetectedItem:
    kind: str
    label: str
    page: int
    source: str = "unknown"
    confidence: float = 0.0
    bbox: Optional[list[float]] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageRecord:
    index: int
    width: int = 0
    height: int = 0

    image_raw: Optional[np.ndarray] = None
    image_preprocessed: Optional[np.ndarray] = None

    vector_text: str = ""
    tables: list[list[list[Any]]] = field(default_factory=list)
    ocr_result: list[Any] = field(default_factory=list)

    symbols: list[DetectedItem] = field(default_factory=list)
    devices: list[DetectedItem] = field(default_factory=list)


@dataclass
class Context:
    pdf: Path
    output: Path

    document: Optional[fitz.Document] = None
    pages: list[int] = field(default_factory=list)

    page_data: dict[int, PageRecord] = field(default_factory=dict)

    # Backward-compatible fields
    images: dict[int, np.ndarray] = field(default_factory=dict)
    processed_images: dict[int, np.ndarray] = field(default_factory=dict)
    text: dict[int, str] = field(default_factory=dict)
    tables: dict[int, list[list[list[Any]]]] = field(default_factory=dict)
    ocr: dict[int, list[Any]] = field(default_factory=dict)
    symbols: dict[int, list[DetectedItem]] = field(default_factory=dict)
    devices: dict[int, list[DetectedItem]] = field(default_factory=dict)

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# -----------------------------
# Pipeline
# -----------------------------

class Pipeline:
    """
    Deterministyczny pipeline do analizy dokumentacji technicznej.
    - najpierw tekst wektorowy,
    - potem render,
    - OCR tylko dla stron z pustym / słabym tekstem,
    - na końcu detekcje heurystyczne i eksport.
    """

    OCR_LANG = "en"
    OCR_USE_ANGLE = True

    DEVICE_KEYWORDS = [
        "rozdzielnica", "rz", "rk", "rsk", "rs1", "rs2", "rs3", "rs4",
        "stycznik", "przekaźnik", "wyłącznik", "bezpiecznik", "rccb",
        "rcd", "spd", "lampka", "gniazdo", "uz", "apf", "aw", "ups",
        "patch panel", "rack", "switch", "lan", "uziemienie", "szyna",
    ]

    SYMBOL_KEYWORDS = [
        "F", "Q", "H", "K", "T", "U", "X", "Y", "PE", "N", "L1", "L2", "L3",
        "AW", "APF", "SPD", "RS", "RK", "RSK", "FT", "PP1", "PP2",
    ]

    def __init__(self) -> None:
        self.ocr_engine = PaddleOCR(
            use_angle_cls=self.OCR_USE_ANGLE,
            lang=self.OCR_LANG,
            show_log=False,
        )

    # ---- lifecycle -------------------------------------------------

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # nic specjalnego; dokument zamykany jest w close()
        self.close()

    def close(self) -> None:
        if self._ctx_document is not None:
            try:
                self._ctx_document.close()
            except Exception:
                pass
            self._ctx_document = None

    @property
    def _ctx_document(self) -> Optional[fitz.Document]:
        return getattr(self, "_document_ref", None)

    @_ctx_document.setter
    def _ctx_document(self, value: Optional[fitz.Document]) -> None:
        setattr(self, "_document_ref", value)

    # ---- main ------------------------------------------------------

    def run(self, ctx: Context) -> Context:
        try:
            ctx = self.load_pdf(ctx)
            ctx = self.extract_vector_text(ctx)
            ctx = self.render_pages(ctx)
            ctx = self.extract_tables(ctx)
            ctx = self.preprocess(ctx)
            ctx = self.ocr_pages(ctx)
            ctx = self.detect_symbols(ctx)
            ctx = self.detect_devices(ctx)
            ctx = self.build_graph(ctx)
            ctx = self.validate(ctx)
            ctx = self.export(ctx)
            return ctx
        finally:
            self.close()

    # ---- helpers ---------------------------------------------------

    def _ensure_page(self, ctx: Context, page_index: int) -> PageRecord:
        if page_index not in ctx.page_data:
            ctx.page_data[page_index] = PageRecord(index=page_index)
        return ctx.page_data[page_index]

    def _normalize_text(self, text: str) -> str:
        text = text or ""
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _ocr_to_lines(self, ocr_result: Any) -> list[dict[str, Any]]:
        """
        Zamienia wynik PaddleOCR na listę linii z bbox + tekst.
        Oczekiwany format: [[ [bbox], (text, score) ], ...]
        """
        lines: list[dict[str, Any]] = []
        if not ocr_result:
            return lines

        # PaddleOCR zwraca zwykle listę stron; tutaj przepuszczamy pojedynczą stronę
        page_result = ocr_result[0] if isinstance(ocr_result, list) and ocr_result and isinstance(ocr_result[0], list) else ocr_result

        for item in page_result or []:
            try:
                bbox = item[0]
                text, score = item[1]
                lines.append({
                    "text": text,
                    "score": float(score),
                    "bbox": bbox,
                })
            except Exception:
                continue
        return lines

    def _page_needs_ocr(self, text: str) -> bool:
        """
        OCR uruchamiamy tylko gdy tekst wektorowy jest słaby lub pusty.
        Heurystyka: < 40 znaków lub mało alfanumeryków.
        """
        if not text:
            return True
        compact = re.sub(r"\s+", "", text)
        alpha_num = sum(ch.isalnum() for ch in compact)
        return len(compact) < 40 or alpha_num < 20

    def _find_keywords(self, text: str, keywords: list[str]) -> list[DetectedItem]:
        """
        Prosta detekcja heurystyczna po tekście.
        Zwraca znalezione słowa-klucze jako itemy.
        """
        found: list[DetectedItem] = []
        if not text:
            return found

        lower = text.lower()
        for kw in keywords:
            if kw.lower() in lower:
                found.append(
                    DetectedItem(
                        kind="keyword",
                        label=kw,
                        page=-1,
                        source="text",
                        confidence=0.80,
                    )
                )
        return found

    # ---- stages ----------------------------------------------------

    def load_pdf(self, ctx: Context) -> Context:
        if not ctx.pdf.exists():
            raise FileNotFoundError(f"Brak pliku PDF: {ctx.pdf}")

        ctx.document = fitz.open(ctx.pdf)
        self._ctx_document = ctx.document

        ctx.pages = list(range(len(ctx.document)))
        for i in ctx.pages:
            page = ctx.document[i]
            record = self._ensure_page(ctx, i)
            rect = page.rect
            record.width = int(rect.width)
            record.height = int(rect.height)

        return ctx

    def extract_vector_text(self, ctx: Context) -> Context:
        """
        Najpierw próbujemy wyciągnąć tekst wektorowy z PyMuPDF.
        """
        if ctx.document is None:
            raise RuntimeError("Dokument nie został załadowany.")

        for i in ctx.pages:
            page = ctx.document[i]
            text = page.get_text("text") or ""
            text = self._normalize_text(text)

            ctx.text[i] = text
            ctx.page_data[i].vector_text = text

        return ctx

    def render_pages(self, ctx: Context, dpi: int = 600) -> Context:
        """
        Render stron do obrazu. Potrzebne dla OCR oraz detekcji symboli.
        """
        if ctx.document is None:
            raise RuntimeError("Dokument nie został załadowany.")

        for i in ctx.pages:
            page = ctx.document[i]
            pix = page.get_pixmap(dpi=dpi, alpha=False)

            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape(pix.height, pix.width, pix.n)

            # PyMuPDF daje zwykle RGB; zachowujemy raw i page record
            ctx.images[i] = img
            ctx.page_data[i].image_raw = img

        return ctx

    def extract_tables(self, ctx: Context) -> Context:
        """
        Ekstrakcja tabel z użyciem pdfplumber.
        """
        with pdfplumber.open(ctx.pdf) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables() or []
                except Exception as exc:
                    ctx.warnings.append(f"Strona {i}: extract_tables failed: {exc}")
                    tables = []

                ctx.tables[i] = tables
                ctx.page_data[i].tables = tables

        return ctx

    def preprocess(self, ctx: Context) -> Context:
        """
        Wstępne przetwarzanie obrazu pod OCR / analizę symboli.
        """
        for i, img in ctx.images.items():
            try:
                if img.ndim == 3 and img.shape[2] == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif img.ndim == 3 and img.shape[2] == 4:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                else:
                    gray = img.copy()

                # lekkie odszumianie + progowanie Otsu
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                ctx.processed_images[i] = thr
                ctx.page_data[i].image_preprocessed = thr
            except Exception as exc:
                ctx.warnings.append(f"Strona {i}: preprocess failed: {exc}")
                ctx.processed_images[i] = img
                ctx.page_data[i].image_preprocessed = img

        return ctx

    def ocr_pages(self, ctx: Context) -> Context:
        """
        OCR tylko dla stron, które nie mają wystarczającego tekstu wektorowego.
        """
        for i, img in ctx.processed_images.items():
            vector_text = ctx.text.get(i, "")
            if not self._page_needs_ocr(vector_text):
                ctx.ocr[i] = []
                ctx.page_data[i].ocr_result = []
                continue

            try:
                result = self.ocr_engine.ocr(img, cls=self.OCR_USE_ANGLE)
                ctx.ocr[i] = result or []
                ctx.page_data[i].ocr_result = result or []
            except Exception as exc:
                ctx.warnings.append(f"Strona {i}: OCR failed: {exc}")
                ctx.ocr[i] = []
                ctx.page_data[i].ocr_result = []

        return ctx

    def detect_symbols(self, ctx: Context) -> Context:
        """
        Heurystyczne wykrywanie symboli.
        W tej wersji: bazujemy na tekście wektorowym i OCR.
        """
        for i in ctx.pages:
            record = ctx.page_data[i]
            items: list[DetectedItem] = []

            # 1) z tekstu wektorowego
            vector_matches = self._find_keywords(record.vector_text, self.SYMBOL_KEYWORDS)
            for m in vector_matches:
                m.page = i
                m.source = "vector_text"
            items.extend(vector_matches)

            # 2) z OCR
            ocr_lines = self._ocr_to_lines(ctx.ocr.get(i, []))
            for line in ocr_lines:
                txt = str(line.get("text", "")).strip()
                if not txt:
                    continue

                # wyszukaj krótkie oznaczenia techniczne: F1, Q100, H03, etc.
                tech_ids = re.findall(r"\b(?:F\d+|Q\d+|H\d+|K\d+|T\d+|U\d+|X\d+|Y\d+|AW\d+|APF\d+|PP\d+|RS\d+|RK|RSK|FT)\b", txt)
                for tid in tech_ids:
                    items.append(
                        DetectedItem(
                            kind="symbol",
                            label=tid,
                            page=i,
                            source="ocr",
                            confidence=float(line.get("score", 0.5)),
                            bbox=line.get("bbox"),
                            meta={"text": txt},
                        )
                    )

            # deduplikacja po etykiecie i źródle
            dedup: dict[tuple[str, str], DetectedItem] = {}
            for it in items:
                key = (it.label, it.source)
                if key not in dedup or it.confidence > dedup[key].confidence:
                    dedup[key] = it

            final_items = list(dedup.values())
            ctx.symbols[i] = final_items
            record.symbols = final_items

        return ctx

    def detect_devices(self, ctx: Context) -> Context:
        """
        Heurystyka urządzeń: szukamy słów-kluczy w tekstach, tabelach i OCR.
        """
        for i in ctx.pages:
            record = ctx.page_data[i]
            items: list[DetectedItem] = []

            text_sources = [
                ("vector_text", record.vector_text),
            ]

            for source_name, text in text_sources:
                lower = text.lower()
                for kw in self.DEVICE_KEYWORDS:
                    if kw.lower() in lower:
                        items.append(
                            DetectedItem(
                                kind="device",
                                label=kw,
                                page=i,
                                source=source_name,
                                confidence=0.75,
                            )
                        )

            # Detekcja z OCR
            for line in self._ocr_to_lines(ctx.ocr.get(i, [])):
                txt = str(line.get("text", "")).strip()
                if not txt:
                    continue

                # fuzzy match do słownika urządzeń
                best = process.extractOne(
                    txt,
                    self.DEVICE_KEYWORDS,
                    scorer=fuzz.partial_ratio,
                )
                if best and best[1] >= 85:
                    items.append(
                        DetectedItem(
                            kind="device",
                            label=best[0],
                            page=i,
                            source="ocr",
                            confidence=float(line.get("score", 0.5)),
                            bbox=line.get("bbox"),
                            meta={"raw_text": txt, "match_score": best[1]},
                        )
                    )

            # deduplikacja
            dedup: dict[tuple[str, str], DetectedItem] = {}
            for it in items:
                key = (it.label.lower(), it.source)
                if key not in dedup or it.confidence > dedup[key].confidence:
                    dedup[key] = it

            final_items = list(dedup.values())
            ctx.devices[i] = final_items
            record.devices = final_items

        return ctx

    def build_graph(self, ctx: Context) -> Context:
        """
        Graf zależności:
        - węzeł dokumentu,
        - węzły stron,
        - węzły wykrytych symboli i urządzeń,
        - relacje 'contains' oraz 'mentions'.
        """
        g = nx.DiGraph()
        g.add_node("document", kind="document", path=str(ctx.pdf))

        for i in ctx.pages:
            page_id = f"page:{i}"
            record = ctx.page_data[i]

            g.add_node(
                page_id,
                kind="page",
                index=i,
                width=record.width,
                height=record.height,
            )
            g.add_edge("document", page_id, relation="contains")

            for item in record.symbols:
                node_id = f"symbol:{i}:{item.label}:{item.source}"
                g.add_node(
                    node_id,
                    kind="symbol",
                    label=item.label,
                    page=i,
                    source=item.source,
                    confidence=item.confidence,
                )
                g.add_edge(page_id, node_id, relation="contains")

            for item in record.devices:
                node_id = f"device:{i}:{item.label}:{item.source}"
                g.add_node(
                    node_id,
                    kind="device",
                    label=item.label,
                    page=i,
                    source=item.source,
                    confidence=item.confidence,
                )
                g.add_edge(page_id, node_id, relation="contains")

            # tekst / tabele
            if record.vector_text:
                g.add_node(
                    f"text:{i}",
                    kind="text",
                    length=len(record.vector_text),
                )
                g.add_edge(page_id, f"text:{i}", relation="has_text")

            if record.tables:
                g.add_node(
                    f"tables:{i}",
                    kind="tables",
                    count=len(record.tables),
                )
                g.add_edge(page_id, f"tables:{i}", relation="has_tables")

        ctx.graph = g
        return ctx

    def validate(self, ctx: Context) -> Context:
        """
        Walidacja podstawowa.
        """
        if not ctx.pages:
            ctx.errors.append("Dokument nie zawiera stron.")

        missing_text_pages = [i for i in ctx.pages if not ctx.text.get(i)]
        if len(missing_text_pages) == len(ctx.pages):
            ctx.warnings.append("Brak tekstu wektorowego na wszystkich stronach.")

        if not ctx.images:
            ctx.warnings.append("Brak renderów stron.")

        if ctx.graph.number_of_nodes() == 0:
            ctx.errors.append("Graf nie został zbudowany.")

        return ctx

    def export(self, ctx: Context) -> Context:
        """
        Eksport:
        - page_XXX.md dla stron,
        - summary.json z pełnym wynikiem,
        - tables_page_XXX.csv dla tabel,
        - graph.json z grafem.
        """
        ctx.output.mkdir(parents=True, exist_ok=True)

        # Markdown per page
        for i, text in ctx.text.items():
            md_path = ctx.output / f"page_{i:03d}.md"
            md_path.write_text(text or "", encoding="utf-8")

        # Tabele per page
        for i, tables in ctx.tables.items():
            if not tables:
                continue

            for t_idx, table in enumerate(tables):
                try:
                    # normalizacja tabeli do DataFrame
                    df = pd.DataFrame(table)
                    csv_path = ctx.output / f"page_{i:03d}_table_{t_idx:02d}.csv"
                    df.to_csv(csv_path, index=False, header=False, encoding="utf-8-sig")
                except Exception as exc:
                    ctx.warnings.append(f"Strona {i}, tabela {t_idx}: export CSV failed: {exc}")

        # Graph export
        graph_data = {
            "nodes": [
                {"id": n, **data}
                for n, data in ctx.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **data}
                for u, v, data in ctx.graph.edges(data=True)
            ],
        }
        (ctx.output / "graph.json").write_text(
            json.dumps(graph_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Full summary
        summary = {
            "pdf": str(ctx.pdf),
            "pages": ctx.pages,
            "warnings": ctx.warnings,
            "errors": ctx.errors,
            "text": ctx.text,
            "tables_count": {str(k): len(v) for k, v in ctx.tables.items()},
            "symbols": {
                str(k): [asdict(x) for x in v]
                for k, v in ctx.symbols.items()
            },
            "devices": {
                str(k): [asdict(x) for x in v]
                for k, v in ctx.devices.items()
            },
            "graph_stats": {
                "nodes": ctx.graph.number_of_nodes(),
                "edges": ctx.graph.number_of_edges(),
            },
        }

        (ctx.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return ctx


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    ctx = Context(
        pdf=Path("input.pdf"),
        output=Path("output"),
    )

    with Pipeline() as pipeline:
        result = pipeline.run(ctx)

    print("OK")
    print(f"Stron: {len(result.pages)}")
    print(f"Błędy: {result.errors}")
    print(f"Ostrzeżenia: {result.warnings}")