from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import json
import re

import cv2
import fitz
import numpy as np
import pandas as pd
import pdfplumber
import networkx as nx
from paddleocr import PaddleOCR
from rapidfuzz import process, fuzz


# -----------------------------
# Additional data structures
# -----------------------------

@dataclass
class LegendEntry:
    code: str
    label: str
    page: int
    source: str = "unknown"
    confidence: float = 0.0
    bbox: Optional[list[float]] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoomEntry:
    room_no: str
    room_name: str
    page: int
    lx: Optional[int] = None
    source: str = "vector"
    confidence: float = 0.0
    bbox: Optional[list[float]] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitEntry:
    circuit_no: str
    label: str
    page: int
    source: str = "unknown"
    confidence: float = 0.0
    bbox: Optional[list[float]] = None
    meta: dict[str, Any] = field(default_factory=dict)


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

    legend: list[LegendEntry] = field(default_factory=list)
    rooms: list[RoomEntry] = field(default_factory=list)
    circuits: list[CircuitEntry] = field(default_factory=list)

    symbols: list[DetectedItem] = field(default_factory=list)
    devices: list[DetectedItem] = field(default_factory=list)


@dataclass
class Context:
    pdf: Path
    output: Path

    document: Optional[fitz.Document] = None
    pages: list[int] = field(default_factory=list)

    page_data: dict[int, PageRecord] = field(default_factory=dict)

    images: dict[int, np.ndarray] = field(default_factory=dict)
    processed_images: dict[int, np.ndarray] = field(default_factory=dict)
    text: dict[int, str] = field(default_factory=dict)
    tables: dict[int, list[list[list[Any]]]] = field(default_factory=dict)
    ocr: dict[int, list[Any]] = field(default_factory=dict)

    legends: dict[int, list[LegendEntry]] = field(default_factory=dict)
    rooms: dict[int, list[RoomEntry]] = field(default_factory=dict)
    circuits: dict[int, list[CircuitEntry]] = field(default_factory=dict)

    symbols: dict[int, list[DetectedItem]] = field(default_factory=dict)
    devices: dict[int, list[DetectedItem]] = field(default_factory=dict)

    room_map: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    bom: dict[int, list[dict[str, Any]]] = field(default_factory=dict)

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Pipeline:
    LEGEND_HINTS = [
        "legenda", "legend", "oznaczenia", "symbole", "opis symboli",
        "wykaz symboli", "objaśnienia", "znaczenie symboli"
    ]

    ROOM_PATTERN = re.compile(r"(?<!\d)(-?\d+\.\d{2})(?!\d)")
    CIRCUIT_PATTERN = re.compile(
        r"\b(?:F|Q|H|K|T|U|X|Y|AW|APF|PP|RS|RK|RSK)\d*[A-Z]?\d*\b"
    )

    BOM_HINTS = [
        "zestawienie materiałów", "bom", "wykaz materiałów",
        "podstawowe zestawienie materiałów", "lista materiałów"
    ]

    ROOM_LABEL_STOPWORDS = {
        "nr", "pom", "pom.", "nazwa", "klasyfikacja", "min", "wg", "normy"
    }

    def __init__(self) -> None:
        self.ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
        )

    # -----------------------------
    # orchestration
    # -----------------------------

    def run(self, ctx: Context) -> Context:
        try:
            ctx = self.load_pdf(ctx)
            ctx = self.extract_vector_text(ctx)
            ctx = self.render_pages(ctx)
            ctx = self.extract_tables(ctx)
            ctx = self.preprocess(ctx)
            ctx = self.ocr_pages(ctx)
            ctx = self.detect_legend(ctx)
            ctx = self.extract_bom_tables(ctx)
            ctx = self.extract_circuit_numbers(ctx)
            ctx = self.extract_rooms(ctx)
            ctx = self.detect_symbols(ctx)
            ctx = self.detect_devices(ctx)
            ctx = self.map_symbols_to_rooms(ctx)
            ctx = self.build_graph(ctx)
            ctx = self.validate(ctx)
            ctx = self.export(ctx)
            return ctx
        finally:
            self.close()

    def close(self) -> None:
        if self.document is not None:
            try:
                self.document.close()
            except Exception:
                pass
            self.document = None

    # -----------------------------
    # utilities
    # -----------------------------

    def _normalize_text(self, text: str) -> str:
        text = text or ""
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _page_text(self, ctx: Context, page_index: int) -> str:
        return ctx.text.get(page_index, "") or ""

    def _page_needs_ocr(self, text: str) -> bool:
        if not text:
            return True
        compact = re.sub(r"\s+", "", text)
        alpha_num = sum(ch.isalnum() for ch in compact)
        return len(compact) < 40 or alpha_num < 20

    def _ocr_to_lines(self, ocr_result: Any) -> list[dict[str, Any]]:
        lines: list[dict[str, Any]] = []
        if not ocr_result:
            return lines

        page_result = ocr_result[0] if isinstance(ocr_result, list) and ocr_result and isinstance(ocr_result[0], list) else ocr_result

        for item in page_result or []:
            try:
                bbox = item[0]
                text, score = item[1]
                lines.append({"text": text, "score": float(score), "bbox": bbox})
            except Exception:
                continue
        return lines

    def _best_page_for_hint(self, ctx: Context, hints: list[str]) -> list[int]:
        pages: list[int] = []
        for i in ctx.pages:
            txt = self._page_text(ctx, i).lower()
            if any(h in txt for h in hints):
                pages.append(i)
        return pages

    def _collect_table_text(self, table: list[list[Any]]) -> str:
        parts: list[str] = []
        for row in table or []:
            for cell in row or []:
                if cell is None:
                    continue
                s = str(cell).strip()
                if s:
                    parts.append(s)
        return " ".join(parts)

    # -----------------------------
    # core stages
    # -----------------------------

    def load_pdf(self, ctx: Context) -> Context:
        if not ctx.pdf.exists():
            raise FileNotFoundError(f"Brak pliku PDF: {ctx.pdf}")
        ctx.document = fitz.open(ctx.pdf)
        ctx.pages = list(range(len(ctx.document)))

        for i in ctx.pages:
            page = ctx.document[i]
            record = ctx.page_data.setdefault(i, PageRecord(index=i))
            rect = page.rect
            record.width = int(rect.width)
            record.height = int(rect.height)
        return ctx

    def extract_vector_text(self, ctx: Context) -> Context:
        for i in ctx.pages:
            page = ctx.document[i]
            text = self._normalize_text(page.get_text("text") or "")
            ctx.text[i] = text
            ctx.page_data[i].vector_text = text
        return ctx

    def render_pages(self, ctx: Context, dpi: int = 600) -> Context:
        for i in ctx.pages:
            page = ctx.document[i]
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            ctx.images[i] = img
            ctx.page_data[i].image_raw = img
        return ctx

    def extract_tables(self, ctx: Context) -> Context:
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
        for i, img in ctx.images.items():
            try:
                if img.ndim == 3 and img.shape[2] == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif img.ndim == 3 and img.shape[2] == 4:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                else:
                    gray = img.copy()
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
        for i, img in ctx.processed_images.items():
            if not self._page_needs_ocr(ctx.text.get(i, "")):
                ctx.ocr[i] = []
                ctx.page_data[i].ocr_result = []
                continue
            try:
                result = self.ocr_engine.ocr(img, cls=True)
                ctx.ocr[i] = result or []
                ctx.page_data[i].ocr_result = result or []
            except Exception as exc:
                ctx.warnings.append(f"Strona {i}: OCR failed: {exc}")
                ctx.ocr[i] = []
                ctx.page_data[i].ocr_result = []
        return ctx

    # -----------------------------
    # legend
    # -----------------------------

    def detect_legend(self, ctx: Context) -> Context:
        legend_pages = self._best_page_for_hint(ctx, self.LEGEND_HINTS)

        # fallback: jeśli nie ma jawnej etykiety, przeszukaj pierwsze strony schematów
        if not legend_pages:
            legend_pages = [i for i in ctx.pages if i <= 4]

        for i in legend_pages:
            text = self._page_text(ctx, i)
            entries: list[LegendEntry] = []

            # bardzo ostrożna ekstrakcja: szukamy par "kod + opis"
            # np. F0 SPD, Q1 RCD, AW1 oprawa awaryjna itd.
            pattern = re.compile(
                r"\b(?P<code>(?:F|Q|H|K|T|U|X|Y|AW|APF|PP|RS|RK|RSK|FT)\d*[A-Z]?)\b"
                r"(?:\s*[-:/]?\s*)"
                r"(?P<label>[A-Za-zÀ-ÿ0-9żźćńółęąśŻŹĆĄŚĘŁÓŃ\.\-\/\(\) ]{2,80})"
            )

            for m in pattern.finditer(text):
                code = m.group("code").strip()
                label = m.group("label").strip()
                if len(label) < 3:
                    continue
                entries.append(
                    LegendEntry(
                        code=code,
                        label=label,
                        page=i,
                        source="vector_text",
                        confidence=0.72,
                        meta={"matched_text": m.group(0)},
                    )
                )

            # OCR fallback dla stron graficznych
            for line in self._ocr_to_lines(ctx.ocr.get(i, [])):
                txt = str(line.get("text", "")).strip()
                if not txt:
                    continue
                m = re.search(
                    r"\b(?P<code>(?:F|Q|H|K|T|U|X|Y|AW|APF|PP|RS|RK|RSK|FT)\d*[A-Z]?)\b\s*(?P<label>.+)$",
                    txt
                )
                if m:
                    entries.append(
                        LegendEntry(
                            code=m.group("code").strip(),
                            label=m.group("label").strip(),
                            page=i,
                            source="ocr",
                            confidence=float(line.get("score", 0.5)),
                            bbox=line.get("bbox"),
                            meta={"raw_text": txt},
                        )
                    )

            # dedupe
            dedup: dict[tuple[str, str], LegendEntry] = {}
            for e in entries:
                key = (e.code, e.label.lower())
                if key not in dedup or e.confidence > dedup[key].confidence:
                    dedup[key] = e

            final = list(dedup.values())
            ctx.legends[i] = final
            ctx.page_data[i].legend = final

        return ctx

    # -----------------------------
    # BOM
    # -----------------------------

    def extract_bom_tables(self, ctx: Context) -> Context:
        """
        BOM / zestawienie materiałów:
        - strony z nagłówkiem "Podstawowe zestawienie materiałów",
        - tabele z wieloma kolumnami,
        - normalizacja do listy rekordów.
        """
        bom_pages = self._best_page_for_hint(ctx, self.BOM_HINTS)

        for i in bom_pages:
            page_tables = ctx.tables.get(i, [])
            extracted: list[dict[str, Any]] = []

            for t_idx, table in enumerate(page_tables):
                if not table or len(table) < 2:
                    continue

                header = [str(c).strip() if c is not None else "" for c in table[0]]
                header_join = " | ".join(header).lower()

                # heurystyka: BOM zwykle ma kolumny Lp / Wyszczególnienie / Jedn / Ilość / Oznaczenie
                if not any(k in header_join for k in ["lp", "wyszczególnienie", "ilość", "oznaczenie", "jedn"]):
                    # nadal może być tabela materiałów, spróbuj czy wiersze wyglądają jak pozycje
                    pass

                for row in table[1:]:
                    if not row:
                        continue
                    cells = [str(c).strip() if c is not None else "" for c in row]
                    if not any(cells):
                        continue

                    # Najprostsza normalizacja "pozycja = pierwszy sensowny wiersz"
                    item = {
                        "page": i,
                        "table_index": t_idx,
                        "raw": cells,
                    }

                    # spróbuj wyciągnąć typowe pola
                    if len(cells) >= 5:
                        item.update({
                            "lp": cells[0],
                            "description": cells[1],
                            "unit": cells[2],
                            "quantity": cells[3],
                            "designation": cells[4],
                        })
                    elif len(cells) == 4:
                        item.update({
                            "lp": cells[0],
                            "description": cells[1],
                            "unit": cells[2],
                            "quantity": cells[3],
                        })
                    else:
                        item["description"] = " ".join(cells)

                    extracted.append(item)

            ctx.bom[i] = extracted
        return ctx

    # -----------------------------
    # circuit numbers
    # -----------------------------

    def extract_circuit_numbers(self, ctx: Context) -> Context:
        """
        Wydobywa oznaczenia obwodów i aparatów:
        F1, F101, Q0, Q1, H0, AW1, APF1, PP1, RS1 itd.
        """
        for i in ctx.pages:
            text_sources = [
                ("vector_text", ctx.text.get(i, "")),
            ]

            items: list[CircuitEntry] = []

            for source_name, text in text_sources:
                for m in self.CIRCUIT_PATTERN.finditer(text):
                    code = m.group(0).strip()
                    items.append(
                        CircuitEntry(
                            circuit_no=code,
                            label=code,
                            page=i,
                            source=source_name,
                            confidence=0.8,
                            meta={"matched_text": m.group(0)},
                        )
                    )

            for line in self._ocr_to_lines(ctx.ocr.get(i, [])):
                txt = str(line.get("text", "")).strip()
                if not txt:
                    continue
                for m in self.CIRCUIT_PATTERN.finditer(txt):
                    code = m.group(0).strip()
                    items.append(
                        CircuitEntry(
                            circuit_no=code,
                            label=code,
                            page=i,
                            source="ocr",
                            confidence=float(line.get("score", 0.5)),
                            bbox=line.get("bbox"),
                            meta={"raw_text": txt},
                        )
                    )

            # dedupe
            dedup: dict[tuple[str, str], CircuitEntry] = {}
            for c in items:
                key = (c.circuit_no, c.source)
                if key not in dedup or c.confidence > dedup[key].confidence:
                    dedup[key] = c

            final = list(dedup.values())
            ctx.circuits[i] = final
            ctx.page_data[i].circuits = final

        return ctx

    # -----------------------------
    # rooms
    # -----------------------------

    def extract_rooms(self, ctx: Context) -> Context:
        """
        Ekstrakcja pomieszczeń z tabeli oświetlenia.
        W tym dokumencie to kluczowe źródło numerów pomieszczeń.
        """
        for i in ctx.pages:
            text = ctx.text.get(i, "")
            if "tabela 1 - natężenie oświetlenia" not in text.lower():
                continue

            rooms: list[RoomEntry] = []
            lines = text.splitlines()

            # wykrycie wierszy typu:
            # -1.01 Magazyn dzienny Składy i magazyny 100
            line_pattern = re.compile(
                r"^(?P<room_no>-?\d+\.\d{2})\s+(?P<name>.+?)\s+(?P<classification>.+?)\s+(?P<lx>\d{2,4})$"
            )

            for raw in lines:
                row = raw.strip()
                if not row:
                    continue
                m = line_pattern.match(row)
                if m:
                    room_no = m.group("room_no").strip()
                    name = m.group("name").strip()
                    lx = int(m.group("lx"))
                    rooms.append(
                        RoomEntry(
                            room_no=room_no,
                            room_name=name,
                            page=i,
                            lx=lx,
                            source="vector_text",
                            confidence=0.9,
                            meta={"classification": m.group("classification").strip()},
                        )
                    )

            # fallback: jeżeli PDF nie daje wierszy line-by-line, użyj regex globalnie
            if not rooms:
                room_no_matches = list(self.ROOM_PATTERN.finditer(text))
                for m in room_no_matches:
                    room_no = m.group(1)
                    # znajdź kawałek tekstu wokół numeru
                    start = max(0, m.start() - 80)
                    end = min(len(text), m.end() + 120)
                    snippet = text[start:end]
                    # wyciągnij nazwę jako fragment między numerem a klasą / wartością lx
                    name = self._guess_room_name(snippet, room_no)
                    if name:
                        rooms.append(
                            RoomEntry(
                                room_no=room_no,
                                room_name=name,
                                page=i,
                                source="vector_text",
                                confidence=0.65,
                                meta={"snippet": snippet},
                            )
                        )

            ctx.rooms[i] = rooms
            ctx.page_data[i].rooms = rooms

        return ctx

    def _guess_room_name(self, snippet: str, room_no: str) -> Optional[str]:
        s = snippet.replace(room_no, " ")
        s = re.sub(r"\s+", " ", s).strip()
        tokens = s.split(" ")

        # odrzuć nagłówki tabeli
        cleaned = []
        for t in tokens:
            low = t.lower().strip(".,:;()")
            if low in self.ROOM_LABEL_STOPWORDS:
                continue
            if re.fullmatch(r"\d{2,4}", low):
                continue
            cleaned.append(t)

        if not cleaned:
            return None

        # najkrótszy sensowny fragment
        return " ".join(cleaned[:5]).strip()

    # -----------------------------
    # symbols / devices
    # -----------------------------

    def detect_symbols(self, ctx: Context) -> Context:
        for i in ctx.pages:
            record = ctx.page_data[i]
            items: list[DetectedItem] = []

            for m in self.CIRCUIT_PATTERN.finditer(record.vector_text):
                items.append(
                    DetectedItem(
                        kind="symbol",
                        label=m.group(0),
                        page=i,
                        source="vector_text",
                        confidence=0.8,
                    )
                )

            for line in self._ocr_to_lines(ctx.ocr.get(i, [])):
                txt = str(line.get("text", "")).strip()
                if not txt:
                    continue
                for m in self.CIRCUIT_PATTERN.finditer(txt):
                    items.append(
                        DetectedItem(
                            kind="symbol",
                            label=m.group(0),
                            page=i,
                            source="ocr",
                            confidence=float(line.get("score", 0.5)),
                            bbox=line.get("bbox"),
                            meta={"raw_text": txt},
                        )
                    )

            dedup: dict[tuple[str, str], DetectedItem] = {}
            for it in items:
                key = (it.label, it.source)
                if key not in dedup or it.confidence > dedup[key].confidence:
                    dedup[key] = it

            final = list(dedup.values())
            ctx.symbols[i] = final
            record.symbols = final
        return ctx

    def detect_devices(self, ctx: Context) -> Context:
        device_keywords = [
            "rozdzielnica", "stycznik", "przekaźnik", "wyłącznik", "bezpiecznik",
            "spd", "rcd", "lampka", "gniazdo", "ups", "szafa rack", "switch",
            "patch panel", "szyna", "uziemienie", "terminal"
        ]

        for i in ctx.pages:
            record = ctx.page_data[i]
            items: list[DetectedItem] = []

            lower = record.vector_text.lower()
            for kw in device_keywords:
                if kw in lower:
                    items.append(
                        DetectedItem(
                            kind="device",
                            label=kw,
                            page=i,
                            source="vector_text",
                            confidence=0.75,
                        )
                    )

            for line in self._ocr_to_lines(ctx.ocr.get(i, [])):
                txt = str(line.get("text", "")).strip()
                if not txt:
                    continue
                best = process.extractOne(txt, device_keywords, scorer=fuzz.partial_ratio)
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

            dedup: dict[tuple[str, str], DetectedItem] = {}
            for it in items:
                key = (it.label.lower(), it.source)
                if key not in dedup or it.confidence > dedup[key].confidence:
                    dedup[key] = it

            final = list(dedup.values())
            ctx.devices[i] = final
            record.devices = final

        return ctx

    # -----------------------------
    # room mapping
    # -----------------------------

    def map_symbols_to_rooms(self, ctx: Context) -> Context:
        """
        Mapowanie:
        - bierze numer pomieszczenia z tabeli pomieszczeń,
        - bierze oznaczenia symboli / urządzeń,
        - przypisuje po wspólnej konwencji oznaczeń oraz stronie.
        W tym projekcie częściowo działa po numerach typu 0.01, -1.13, PELxx, Fxx.
        """
        # budujemy indeks pomieszczeń po stronie
        room_index: dict[str, RoomEntry] = {}
        for page_rooms in ctx.rooms.values():
            for room in page_rooms:
                room_index[room.room_no] = room

        mapped: dict[int, list[dict[str, Any]]] = {}

        for i in ctx.pages:
            page_mappings: list[dict[str, Any]] = []
            page_text = ctx.text.get(i, "")

            # 1) mapowanie przez tekst strony: jeśli numer pomieszczenia występuje na stronie
            for room_no, room in room_index.items():
                if room_no in page_text:
                    page_mappings.append(
                        {
                            "room_no": room_no,
                            "room_name": room.room_name,
                            "page": i,
                            "reason": "room_no_present_in_text",
                            "symbols": [s.label for s in ctx.symbols.get(i, [])],
                            "devices": [d.label for d in ctx.devices.get(i, [])],
                        }
                    )

            # 2) mapowanie PEL/obwodów do pomieszczeń komputerowych / sali
            if any(x in page_text.lower() for x in ["sala komputerowa", "pel", "rack"]):
                for room_no, room in room_index.items():
                    if room_no == "-1.13":
                        page_mappings.append(
                            {
                                "room_no": room_no,
                                "room_name": room.room_name,
                                "page": i,
                                "reason": "computer_room_keywords",
                                "symbols": [s.label for s in ctx.symbols.get(i, [])],
                                "devices": [d.label for d in ctx.devices.get(i, [])],
                            }
                        )

            # 3) pomieszczenia kuchenne 0.01 / 0.02
            if any(x in page_text.lower() for x in ["kuchnia", "rozdzielnia", "zmywalnia", "jadalnia", "przygotowalnia"]):
                for room_no, room in room_index.items():
                    if room_no in {"0.01", "0.02"}:
                        page_mappings.append(
                            {
                                "room_no": room_no,
                                "room_name": room.room_name,
                                "page": i,
                                "reason": "kitchen_area_keywords",
                                "symbols": [s.label for s in ctx.symbols.get(i, [])],
                                "devices": [d.label for d in ctx.devices.get(i, [])],
                            }
                        )

            # 4) dedup
            seen = set()
            unique = []
            for item in page_mappings:
                key = (item["room_no"], item["page"], item["reason"])
                if key not in seen:
                    seen.add(key)
                    unique.append(item)

            mapped[i] = unique

        ctx.room_map = mapped
        return ctx

    # -----------------------------
    # graph / validation / export
    # -----------------------------

    def build_graph(self, ctx: Context) -> Context:
        g = nx.DiGraph()
        g.add_node("document", kind="document", path=str(ctx.pdf))

        for i in ctx.pages:
            page_id = f"page:{i}"
            record = ctx.page_data[i]
            g.add_node(page_id, kind="page", index=i, width=record.width, height=record.height)
            g.add_edge("document", page_id, relation="contains")

            for room in record.rooms:
                node_id = f"room:{i}:{room.room_no}"
                g.add_node(node_id, kind="room", room_no=room.room_no, room_name=room.room_name, page=i)
                g.add_edge(page_id, node_id, relation="contains")

            for item in record.legend:
                node_id = f"legend:{i}:{item.code}"
                g.add_node(node_id, kind="legend", code=item.code, label=item.label, page=i)
                g.add_edge(page_id, node_id, relation="contains")

            for item in record.circuits:
                node_id = f"circuit:{i}:{item.circuit_no}"
                g.add_node(node_id, kind="circuit", circuit_no=item.circuit_no, page=i)
                g.add_edge(page_id, node_id, relation="contains")

            for item in record.symbols:
                node_id = f"symbol:{i}:{item.label}:{item.source}"
                g.add_node(node_id, kind="symbol", label=item.label, page=i, source=item.source)
                g.add_edge(page_id, node_id, relation="contains")

            for item in record.devices:
                node_id = f"device:{i}:{item.label}:{item.source}"
                g.add_node(node_id, kind="device", label=item.label, page=i, source=item.source)
                g.add_edge(page_id, node_id, relation="contains")

            if record.tables:
                g.add_node(f"tables:{i}", kind="tables", count=len(record.tables))
                g.add_edge(page_id, f"tables:{i}", relation="has_tables")

            if ctx.bom.get(i):
                g.add_node(f"bom:{i}", kind="bom", count=len(ctx.bom[i]))
                g.add_edge(page_id, f"bom:{i}", relation="has_bom")

            if ctx.room_map.get(i):
                g.add_node(f"room_map:{i}", kind="room_map", count=len(ctx.room_map[i]))
                g.add_edge(page_id, f"room_map:{i}", relation="has_room_map")

        ctx.graph = g
        return ctx

    def validate(self, ctx: Context) -> Context:
        if not ctx.pages:
            ctx.errors.append("Dokument nie zawiera stron.")
        if not ctx.images:
            ctx.warnings.append("Brak renderów stron.")
        if ctx.graph.number_of_nodes() == 0:
            ctx.errors.append("Graf nie został zbudowany.")
        return ctx

    def export(self, ctx: Context) -> Context:
        ctx.output.mkdir(parents=True, exist_ok=True)

        # page text
        for i, text in ctx.text.items():
            (ctx.output / f"page_{i:03d}.md").write_text(text or "", encoding="utf-8")

        # legend
        (ctx.output / "legend.json").write_text(
            json.dumps(
                {str(k): [asdict(x) for x in v] for k, v in ctx.legends.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # rooms
        (ctx.output / "rooms.json").write_text(
            json.dumps(
                {str(k): [asdict(x) for x in v] for k, v in ctx.rooms.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # BOM
        (ctx.output / "bom.json").write_text(
            json.dumps(ctx.bom, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # room map
        (ctx.output / "room_map.json").write_text(
            json.dumps(ctx.room_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # graph
        graph_data = {
            "nodes": [{"id": n, **data} for n, data in ctx.graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **data} for u, v, data in ctx.graph.edges(data=True)],
        }
        (ctx.output / "graph.json").write_text(
            json.dumps(graph_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # summary
        summary = {
            "pdf": str(ctx.pdf),
            "pages": ctx.pages,
            "warnings": ctx.warnings,
            "errors": ctx.errors,
            "graph_stats": {
                "nodes": ctx.graph.number_of_nodes(),
                "edges": ctx.graph.number_of_edges(),
            },
            "counts": {
                "legend_pages": len(ctx.legends),
                "room_pages": len(ctx.rooms),
                "bom_pages": len(ctx.bom),
                "circuit_pages": len(ctx.circuits),
            },
        }

        (ctx.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return ctx