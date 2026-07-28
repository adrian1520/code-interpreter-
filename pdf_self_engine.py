from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import cv2
import fitz  # PyMuPDF
import networkx as nx
import numpy as np
import pandas as pd
import pdfplumber
from rapidfuzz import fuzz, process


MNT_DATA = Path("/mnt/data")
BACKEND_NAME_PATTERNS = ("*backend*.py", "*engine*.py", "pipeline*.py", "pdf_*.py")


def log(message: str) -> None:
    print(f"[pdf-self-engine] {message}", flush=True)


@dataclass
class LegendEntry:
    code: str
    label: str
    page: int
    source: str
    confidence: float = 0.0
    bbox: list[float] | list[list[float]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoomEntry:
    room_no: str
    room_name: str
    page: int
    source: str
    confidence: float = 0.0
    lx: int | None = None
    bbox: list[float] | list[list[float]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitEntry:
    circuit_no: str
    label: str
    page: int
    source: str
    confidence: float = 0.0
    bbox: list[float] | list[list[float]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectedItem:
    kind: str
    label: str
    page: int
    source: str
    confidence: float = 0.0
    bbox: list[float] | list[list[float]] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageRecord:
    index: int
    page_number: int
    width: float = 0.0
    height: float = 0.0
    vector_text: str = ""
    markdown_path: str | None = None
    tables: list[list[list[Any]]] = field(default_factory=list)
    table_paths: list[str] = field(default_factory=list)
    ocr_result: list[dict[str, Any]] = field(default_factory=list)
    ocr_path: str | None = None
    legend: list[LegendEntry] = field(default_factory=list)
    rooms: list[RoomEntry] = field(default_factory=list)
    circuits: list[CircuitEntry] = field(default_factory=list)
    symbols: list[DetectedItem] = field(default_factory=list)
    devices: list[DetectedItem] = field(default_factory=list)


@dataclass
class Context:
    input_pdf: Path
    output_dir: Path
    backend_module: Path | None = None
    document: Any = None
    pages: list[int] = field(default_factory=list)
    page_records: dict[int, PageRecord] = field(default_factory=dict)
    images: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    processed_images: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    graph: nx.DiGraph = field(default_factory=nx.DiGraph, repr=False)
    bom: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    room_map: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    LEGEND_HINTS = ("legenda", "oznaczenia", "symbole", "opis symboli", "objaśnienia")
    BOM_HINTS = ("zestawienie materiałów", "wykaz materiałów", "bom", "ilość", "jedn")
    DEVICE_KEYWORDS = (
        "rozdzielnica", "stycznik", "przekaźnik", "wyłącznik", "bezpiecznik", "rcd", "spd",
        "oprawa", "gniazdo", "łącznik", "switch", "patch panel", "rack", "ups", "uziemienie",
    )
    CIRCUIT_PATTERN = re.compile(r"\b(?:F|Q|H|K|T|U|X|Y|AW|APF|PP|RS|RK|RSK|FT|PE|N|L[123])\d*[A-Z]?\d*\b")
    ROOM_PATTERN = re.compile(r"(?<!\d)(-?\d+\.\d{2})(?!\d)")

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi
        self.ocr_engine: Any = None

    def run(self, ctx: Context) -> Context:
        for stage in (
            self.load_pdf, self.extract_vector_text, self.render_pages, self.extract_tables, self.preprocess,
            self.ocr_pages, self.detect_legend, self.extract_bom_tables, self.extract_circuit_numbers,
            self.extract_rooms, self.detect_symbols, self.detect_devices, self.map_symbols_to_rooms,
            self.build_graph, self.validate, self.export,
        ):
            log(stage.__name__)
            ctx = stage(ctx)
        return ctx

    def _record(self, ctx: Context, i: int) -> PageRecord:
        return ctx.page_records.setdefault(i, PageRecord(index=i, page_number=i + 1))

    def _text(self, ctx: Context, i: int) -> str:
        return ctx.page_records.get(i, PageRecord(i, i + 1)).vector_text or ""

    def _normalize(self, text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", (text or "").replace("\x00", " "))).strip()

    def _needs_ocr(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        return len(compact) < 40 or sum(ch.isalnum() for ch in compact) < 20

    def _ocr(self) -> Any:
        if self.ocr_engine is None:
            if importlib.util.find_spec("paddleocr") is None:
                return None
            module = importlib.import_module("paddleocr")
            self.ocr_engine = module.PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        return self.ocr_engine

    def load_pdf(self, ctx: Context) -> Context:
        if not ctx.input_pdf.is_absolute() or not str(ctx.input_pdf).startswith(str(MNT_DATA) + "/"):
            raise ValueError(f"PDF musi mieć ścieżkę absolutną w /mnt/data: {ctx.input_pdf}")
        if not ctx.input_pdf.is_file():
            raise FileNotFoundError(f"Brak pliku PDF: {ctx.input_pdf}")
        if ctx.output_dir.exists() and not ctx.output_dir.is_dir():
            raise NotADirectoryError(f"output_dir nie jest katalogiem: {ctx.output_dir}")
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        ctx.document = fitz.open(ctx.input_pdf)
        ctx.pages = list(range(len(ctx.document)))
        for i in ctx.pages:
            rect = ctx.document[i].rect
            rec = self._record(ctx, i)
            rec.width, rec.height = float(rect.width), float(rect.height)
        return ctx

    def extract_vector_text(self, ctx: Context) -> Context:
        for i in ctx.pages:
            self._record(ctx, i).vector_text = self._normalize(ctx.document[i].get_text("text") or "")
        return ctx

    def render_pages(self, ctx: Context) -> Context:
        for i in ctx.pages:
            pix = ctx.document[i].get_pixmap(dpi=self.dpi, alpha=False)
            ctx.images[i] = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return ctx

    def extract_tables(self, ctx: Context) -> Context:
        with pdfplumber.open(ctx.input_pdf) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    self._record(ctx, i).tables = page.extract_tables() or []
                except Exception as exc:
                    ctx.warnings.append(f"Strona {i + 1}: nie udało się wyodrębnić tabel: {exc}")
        return ctx

    def preprocess(self, ctx: Context) -> Context:
        for i, img in ctx.images.items():
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            ctx.processed_images[i] = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return ctx

    def ocr_pages(self, ctx: Context) -> Context:
        engine = None
        for i, img in ctx.processed_images.items():
            rec = self._record(ctx, i)
            if not self._needs_ocr(rec.vector_text):
                continue
            if engine is None:
                engine = self._ocr()
            if engine is None:
                ctx.warnings.append("PaddleOCR nie jest dostępny; OCR fallback pominięty.")
                break
            try:
                raw = engine.ocr(img, cls=True) or []
                rows = raw[0] if raw and isinstance(raw[0], list) else raw
                for item in rows or []:
                    bbox, pair = item[0], item[1]
                    rec.ocr_result.append({"text": str(pair[0]), "score": float(pair[1]), "bbox": bbox})
            except Exception as exc:
                ctx.warnings.append(f"Strona {i + 1}: OCR nie powiódł się: {exc}")
        return ctx

    def detect_legend(self, ctx: Context) -> Context:
        for i in ctx.pages:
            text = self._text(ctx, i)
            if not any(h in text.lower() for h in self.LEGEND_HINTS):
                continue
            entries = []
            for m in re.finditer(r"\b(?P<code>[A-Z]{1,4}\d*[A-Z]?)\b\s*[-–:]\s*(?P<label>[^\n]{2,100})", text):
                entries.append(LegendEntry(m.group("code"), m.group("label").strip(), i + 1, "vector_text", 0.72))
            self._record(ctx, i).legend = self._dedupe(entries, lambda e: (e.code, e.label.lower()))
        return ctx

    def extract_bom_tables(self, ctx: Context) -> Context:
        for i in ctx.pages:
            if not any(h in self._text(ctx, i).lower() for h in self.BOM_HINTS):
                continue
            rows = []
            for t_idx, table in enumerate(self._record(ctx, i).tables):
                for r_idx, row in enumerate(table[1:] if len(table) > 1 else table):
                    cells = [str(c).strip() if c is not None else "" for c in (row or [])]
                    if any(cells):
                        rows.append({"page": i + 1, "table_index": t_idx, "row_index": r_idx, "raw": cells})
            ctx.bom[i] = rows
        return ctx

    def extract_circuit_numbers(self, ctx: Context) -> Context:
        for i in ctx.pages:
            items = [CircuitEntry(m.group(0), m.group(0), i + 1, "vector_text", 0.8) for m in self.CIRCUIT_PATTERN.finditer(self._text(ctx, i))]
            for line in self._record(ctx, i).ocr_result:
                for m in self.CIRCUIT_PATTERN.finditer(line["text"]):
                    items.append(CircuitEntry(m.group(0), m.group(0), i + 1, "ocr", line.get("score", 0.5), line.get("bbox")))
            self._record(ctx, i).circuits = self._dedupe(items, lambda e: (e.circuit_no, e.source))
        return ctx

    def extract_rooms(self, ctx: Context) -> Context:
        for i in ctx.pages:
            rooms = []
            for m in self.ROOM_PATTERN.finditer(self._text(ctx, i)):
                snippet = self._text(ctx, i)[m.end():m.end() + 80]
                name = re.split(r"\s{2,}|\n|\d{2,4}\b", snippet.strip())[0].strip(" -–:;")
                rooms.append(RoomEntry(m.group(1), name or "brak jawnej nazwy", i + 1, "vector_text", 0.62, meta={"name_inferred_from_adjacent_text": bool(name)}))
            self._record(ctx, i).rooms = self._dedupe(rooms, lambda e: e.room_no)
        return ctx

    def detect_symbols(self, ctx: Context) -> Context:
        for i in ctx.pages:
            self._record(ctx, i).symbols = [DetectedItem("symbol", c.circuit_no, i + 1, c.source, c.confidence, c.bbox) for c in self._record(ctx, i).circuits]
        return ctx

    def detect_devices(self, ctx: Context) -> Context:
        for i in ctx.pages:
            items = []
            lower = self._text(ctx, i).lower()
            for kw in self.DEVICE_KEYWORDS:
                if kw in lower:
                    items.append(DetectedItem("device", kw, i + 1, "vector_text", 0.75))
            for line in self._record(ctx, i).ocr_result:
                best = process.extractOne(line["text"], self.DEVICE_KEYWORDS, scorer=fuzz.partial_ratio)
                if best and best[1] >= 85:
                    items.append(DetectedItem("device", best[0], i + 1, "ocr", line.get("score", 0.5), line.get("bbox"), {"raw_text": line["text"]}))
            self._record(ctx, i).devices = self._dedupe(items, lambda e: (e.label, e.source))
        return ctx

    def map_symbols_to_rooms(self, ctx: Context) -> Context:
        all_rooms = [r for rec in ctx.page_records.values() for r in rec.rooms]
        for i in ctx.pages:
            text = self._text(ctx, i)
            mappings = []
            for room in all_rooms:
                if room.room_no in text:
                    mappings.append({"page": i + 1, "room_no": room.room_no, "room_name": room.room_name, "reason": "room_no_present_in_vector_text", "symbols": [s.label for s in self._record(ctx, i).symbols]})
            ctx.room_map[i] = mappings
        return ctx

    def build_graph(self, ctx: Context) -> Context:
        g = nx.DiGraph()
        g.add_node("document", kind="document", path=str(ctx.input_pdf))
        for i, rec in ctx.page_records.items():
            pid = f"page:{rec.page_number}"
            g.add_node(pid, kind="page", page_number=rec.page_number, width=rec.width, height=rec.height)
            g.add_edge("document", pid, relation="contains")
            for collection_name in ("legend", "rooms", "circuits", "symbols", "devices"):
                for idx, item in enumerate(getattr(rec, collection_name)):
                    node_id = f"{collection_name}:{rec.page_number}:{idx}"
                    g.add_node(node_id, kind=collection_name[:-1], **asdict(item))
                    g.add_edge(pid, node_id, relation="contains")
        ctx.graph = g
        return ctx

    def validate(self, ctx: Context) -> Context:
        if not ctx.pages:
            ctx.errors.append("PDF nie zawiera stron.")
        if not ctx.output_dir.is_dir():
            ctx.errors.append(f"Nie można utworzyć output_dir: {ctx.output_dir}")
        if ctx.graph.number_of_nodes() == 0:
            ctx.errors.append("graph.json byłby pusty; graf nie został zbudowany.")
        if not any(rec.vector_text or rec.ocr_result for rec in ctx.page_records.values()):
            ctx.warnings.append("Nie znaleziono tekstu wektorowego ani OCR w dokumencie.")
        return ctx

    def export(self, ctx: Context) -> Context:
        pages_dir, tables_dir, ocr_dir = ctx.output_dir / "pages", ctx.output_dir / "tables", ctx.output_dir / "ocr"
        pages_dir.mkdir(parents=True, exist_ok=True); tables_dir.mkdir(parents=True, exist_ok=True); ocr_dir.mkdir(parents=True, exist_ok=True)
        for i, rec in sorted(ctx.page_records.items()):
            md = pages_dir / f"page_{rec.page_number:03d}.md"
            md.write_text(rec.vector_text or "", encoding="utf-8")
            rec.markdown_path = str(md)
            for t_idx, table in enumerate(rec.tables):
                csv_path = tables_dir / f"page_{rec.page_number:03d}_table_{t_idx:02d}.csv"
                pd.DataFrame(table).to_csv(csv_path, index=False, header=False, encoding="utf-8-sig")
                rec.table_paths.append(str(csv_path))
            if rec.ocr_result:
                ocr_path = ocr_dir / f"page_{rec.page_number:03d}.json"
                ocr_path.write_text(json.dumps(rec.ocr_result, ensure_ascii=False, indent=2), encoding="utf-8")
                rec.ocr_path = str(ocr_path)
        graph_data = {"nodes": [{"id": n, **d} for n, d in ctx.graph.nodes(data=True)], "edges": [{"source": u, "target": v, **d} for u, v, d in ctx.graph.edges(data=True)]}
        self._write_json(ctx.output_dir / "graph.json", graph_data)
        self._write_json(ctx.output_dir / "legend.json", self._collect(ctx, "legend"))
        self._write_json(ctx.output_dir / "rooms.json", self._collect(ctx, "rooms"))
        self._write_json(ctx.output_dir / "devices.json", self._collect(ctx, "devices"))
        self._write_json(ctx.output_dir / "symbols.json", self._collect(ctx, "symbols"))
        self._write_json(ctx.output_dir / "bom.json", ctx.bom)
        summary = {"input_pdf": str(ctx.input_pdf), "backend_module": str(ctx.backend_module) if ctx.backend_module else None, "output_dir": str(ctx.output_dir), "pages": len(ctx.pages), "artifacts": {"pages": str(pages_dir), "tables": str(tables_dir), "ocr": str(ocr_dir), "graph": str(ctx.output_dir / "graph.json")}, "counts": {"tables": sum(len(r.tables) for r in ctx.page_records.values()), "ocr_pages": sum(1 for r in ctx.page_records.values() if r.ocr_result), "legend_entries": sum(len(r.legend) for r in ctx.page_records.values()), "rooms": sum(len(r.rooms) for r in ctx.page_records.values()), "circuits": sum(len(r.circuits) for r in ctx.page_records.values()), "symbols": sum(len(r.symbols) for r in ctx.page_records.values()), "devices": sum(len(r.devices) for r in ctx.page_records.values())}, "warnings": ctx.warnings, "errors": ctx.errors}
        self._write_json(ctx.output_dir / "summary.json", summary)
        if not (ctx.output_dir / "summary.json").is_file():
            raise RuntimeError("summary.json nie został zapisany")
        ctx.artifacts = summary["artifacts"]
        if ctx.document is not None:
            ctx.document.close(); ctx.document = None
        return ctx

    def _collect(self, ctx: Context, attr: str) -> dict[str, list[dict[str, Any]]]:
        return {str(rec.page_number): [asdict(x) for x in getattr(rec, attr)] for rec in ctx.page_records.values() if getattr(rec, attr)}

    def _write_json(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _dedupe(self, items: list[Any], key_fn: Callable[[Any], Any]) -> list[Any]:
        seen = set(); out = []
        for item in items:
            key = key_fn(item)
            if key not in seen:
                seen.add(key); out.append(item)
        return out


def load_module_from_path(path: str | Path) -> ModuleType:
    module_path = Path(path)
    if not module_path.is_absolute() or not str(module_path).startswith(str(MNT_DATA) + "/"):
        raise ValueError(f"Backend musi być plikiem .py w /mnt/data: {module_path}")
    if not module_path.is_file():
        raise FileNotFoundError(f"Brak backendu: {module_path}")
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nie można przygotować importu backendu: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path.stem] = module
    spec.loader.exec_module(module)
    return module


def find_backend(pdf_path: Path, explicit_backend: str | Path | None = None) -> Path:
    if explicit_backend:
        candidate = Path(explicit_backend)
        return candidate if candidate.is_absolute() else MNT_DATA / candidate
    candidates: list[Path] = []
    for pattern in BACKEND_NAME_PATTERNS:
        candidates.extend(sorted(MNT_DATA.glob(pattern)))
    candidates = [p for p in dict.fromkeys(candidates) if p.is_file() and p.suffix == ".py" and p.name != Path(__file__).name]
    if not candidates:
        return Path(__file__).resolve()
    preferred = [p for p in candidates if p.stem.startswith(pdf_path.stem)]
    return (preferred or candidates)[0]


def run_backend_entry(module: ModuleType, pdf_path: Path, output_dir: Path, backend_path: Path) -> Context | dict[str, Any]:
    if hasattr(module, "main") and callable(module.main):
        return module.main(str(pdf_path), str(output_dir))
    if hasattr(module, "run") and callable(module.run):
        return module.run(str(pdf_path), str(output_dir))
    pipeline_cls = getattr(module, "Pipeline", None)
    context_cls = getattr(module, "Context", Context)
    if pipeline_cls is not None:
        try:
            ctx = context_cls(input_pdf=pdf_path, output_dir=output_dir, backend_module=backend_path)
        except TypeError:
            try:
                ctx = context_cls(pdf=pdf_path, output=output_dir)
            except TypeError:
                ctx = Context(input_pdf=pdf_path, output_dir=output_dir, backend_module=backend_path)
        return pipeline_cls().run(ctx)
    raise AttributeError("Backend musi eksportować main(pdf_path, output_dir), run(pdf_path, output_dir) albo klasę Pipeline z metodą run(ctx).")


def main(pdf_path: str, output_dir: str | None = None, backend_module: str | None = None) -> dict[str, Any]:
    pdf = Path(pdf_path)
    if not pdf.is_absolute():
        pdf = MNT_DATA / pdf
    out = Path(output_dir) if output_dir else MNT_DATA / "output" / pdf.stem
    if not out.is_absolute():
        out = MNT_DATA / out
    if not str(out).startswith(str(MNT_DATA) + "/"):
        raise ValueError(f"output_dir musi być w /mnt/data: {out}")
    backend = find_backend(pdf, backend_module)
    log(f"input_pdf={pdf}")
    log(f"backend_module={backend}")
    log(f"output_dir={out}")
    if backend.resolve() == Path(__file__).resolve():
        ctx = Pipeline().run(Context(input_pdf=pdf, output_dir=out, backend_module=backend))
    else:
        module = load_module_from_path(backend)
        result = run_backend_entry(module, pdf, out, backend)
        if isinstance(result, Context):
            ctx = result
        elif isinstance(result, dict) and (out / "summary.json").is_file():
            return result
        else:
            raise TypeError("Backend zwrócił niespójny wynik; oczekiwano Context albo dict z zapisanym summary.json.")
    summary_path = out / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Brak wymaganego artefaktu: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-shot PDF self-engine launcher for /mnt/data.")
    parser.add_argument("pdf_path")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--backend-module", default=None)
    args = parser.parse_args()
    print(json.dumps(main(args.pdf_path, args.output_dir, args.backend_module), ensure_ascii=False, indent=2))
