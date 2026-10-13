"""Versioned, JSON-serializable document structure. Coordinates are one-based except PDF bbox."""
from typing import Literal, Protocol
from pydantic import BaseModel, Field, ConfigDict, model_validator


class SourceLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["pdf", "docx", "xlsx"]
    page: int | None = Field(default=None, ge=1)
    bbox: list[float] | None = None  # PDF points, top-left origin, unrotated page
    paragraph: int | None = Field(default=None, ge=1)
    heading_path: list[str] = Field(default_factory=list)
    table: int | None = Field(default=None, ge=1)
    sheet: str | None = None
    cell_range: str | None = None
    row_start: int | None = Field(default=None, ge=1)
    row_end: int | None = Field(default=None, ge=1)
    column_start: int | None = Field(default=None, ge=1)
    column_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def check_location(self):
        if self.format == "pdf" and self.page is None:
            raise ValueError("PDF source requires physical page")
        if self.format != "pdf" and self.page is not None:
            raise ValueError("Only PDF has a stable physical page")
        if self.format == "docx" and self.paragraph is None and self.table is None:
            raise ValueError("Word source requires paragraph or table")
        if self.format == "xlsx" and (not self.sheet or not self.cell_range):
            raise ValueError("Excel source requires sheet and cell range")
        if self.bbox is not None and (len(self.bbox) != 4 or self.bbox[2] < self.bbox[0] or self.bbox[3] < self.bbox[1]):
            raise ValueError("Invalid bbox")
        for start, end in ((self.row_start, self.row_end), (self.column_start, self.column_end)):
            if start is not None and end is not None and end < start:
                raise ValueError("Invalid source range")
        return self


class Cell(BaseModel):
    row: int = Field(ge=1)
    column: int = Field(ge=1)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    text: str
    value: str | int | float | bool | None = None
    value_type: str = "text"
    formula: str | None = None
    cached_value: str | int | float | bool | None = None
    cache_status: Literal["not_formula", "missing", "unverified"] = "not_formula"
    number_format: str | None = None
    source: SourceLocation


class Table(BaseModel):
    table_id: str
    rows: int = Field(ge=1)
    columns: int = Field(ge=1)
    cells: list[Cell]
    header_rows: list[int] = Field(default_factory=list)
    header_origin: Literal["explicit", "heuristic", "unknown"] = "unknown"
    column_names: list[str] = Field(default_factory=list)
    column_header_sources: list[SourceLocation] = Field(default_factory=list)
    source: SourceLocation

    @model_validator(mode="after")
    def check_cells(self):
        if any(c.row+c.row_span-1>self.rows or c.column+c.column_span-1>self.columns for c in self.cells):
            raise ValueError("Cell extends beyond table grid")
        if len({(c.row,c.column) for c in self.cells})!=len(self.cells):
            raise ValueError("Duplicate table cell anchor")
        if any(c.source.format!=self.source.format for c in self.cells):
            raise ValueError("Cell source format differs from table")
        return self


class Element(BaseModel):
    element_id: str
    kind: Literal["paragraph", "heading", "table"]
    text: str = ""
    source: SourceLocation
    table_id: str | None = None
    extraction: Literal["native", "ocr"] = "native"


class OCRRequest(BaseModel):
    document_id: str
    page: int = Field(ge=1)
    bbox: list[float] | None = None
    reason: str
    status: Literal["needs_ocr"] = "needs_ocr"


class OCRResult(BaseModel):
    elements: list[Element]
    tables: list[Table] = Field(default_factory=list)
    engine: str
    revision: str
    warnings: list[str] = Field(default_factory=list)


class OCRBackend(Protocol):
    """Future adapter; image bytes supplied explicitly, never download a model implicitly."""
    def recognize(self, image: bytes, request: OCRRequest) -> OCRResult: ...


class Document(BaseModel):
    schema_version: str = "1.0"
    parser_version: str = "native-v1"
    document_id: str  # SHA256 of original bytes: immutable source version
    filename: str
    format: Literal["pdf", "docx", "xlsx"]
    elements: list[Element] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    ocr_requests: list[OCRRequest] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    page_count: int | None = None
    sheets: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_references(self):
        tables = {t.table_id for t in self.tables}
        if len(tables) != len(self.tables):
            raise ValueError("Duplicate table id")
        if len({e.element_id for e in self.elements}) != len(self.elements):
            raise ValueError("Duplicate element id")
        if any(e.kind == "table" and e.table_id not in tables for e in self.elements):
            raise ValueError("Missing table reference")
        if any(e.source.format != self.format for e in self.elements) or any(t.source.format != self.format for t in self.tables):
            raise ValueError("Source format differs from document")
        return self


def location_label(source: dict) -> str:
    source = SourceLocation.model_validate(source)
    if source.format == "pdf":
        label = f"PDF 第 {source.page} 页"
    elif source.format == "xlsx":
        label = f"'{source.sheet}'!{source.cell_range}"
    else:
        label = " / ".join(source.heading_path)
        label += (" · " if label else "") + (f"表 {source.table}" if source.table else f"段落 {source.paragraph}")
    if source.table and source.format == "pdf":
        label += f" · 表 {source.table}"
    if source.row_start and source.format != "xlsx":
        label += f" · 行 {source.row_start}–{source.row_end or source.row_start}"
    return label


PROVENANCE_FIELDS = ("source", "source_label", "element_id", "table_id", "cell_refs", "context_refs", "header_cell_refs", "text_span", "text_spans", "ocr_pending_pages", "schema_version", "parser_version")


def provenance(item):
    return {key: item[key] for key in PROVENANCE_FIELDS if key in item}
