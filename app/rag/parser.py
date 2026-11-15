from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from app.rag.types import ParsedBlock, ParsedDocument

WHITESPACE = re.compile(r"[ \t]+")
MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = WHITESPACE.sub(" ", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def table_to_markdown(rows: Iterable[Iterable[object]]) -> str:
    rows = [[str(cell).replace("|", "\\|").strip() if cell is not None else "" for cell in row] for row in rows]
    rows = [row for row in rows if any(row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


class DocumentParser:
    """把不同格式的文件解析成带页码/章节的文本块。"""

    def parse(self, path: str | Path, doc_id: str, title: str | None = None) -> ParsedDocument:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            doc = self._parse_pdf(path, doc_id, title)
        elif suffix in (".docx", ".doc"):
            doc = self._parse_docx(path, doc_id, title)
        elif suffix in (".xlsx", ".xls"):
            doc = self._parse_xlsx(path, doc_id, title)
        elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
            doc = self._parse_image(path, doc_id, title)
        else:
            doc = self._parse_text(path, doc_id, title)
        doc.metadata.setdefault("suffix", suffix)
        return doc

    def _parse_pdf(self, path: Path, doc_id: str, title: str | None) -> ParsedDocument:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        blocks: list[ParsedBlock] = []
        for page_index, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if not text:
                blocks.append(ParsedBlock(text="[扫描件页面，需要 OCR 兜底]", page=page_index, kind="image_text"))
            else:
                blocks.append(ParsedBlock(text=text, page=page_index))
        return ParsedDocument(
            doc_id=doc_id,
            title=title or path.stem,
            source_type="pdf",
            blocks=blocks,
            metadata={"pages": len(reader.pages)},
        )

    def _parse_docx(self, path: Path, doc_id: str, title: str | None) -> ParsedDocument:
        from docx import Document as WordDocument

        doc = WordDocument(str(path))
        blocks: list[ParsedBlock] = []
        current_section: str | None = None
        for paragraph in doc.paragraphs:
            text = clean_text(paragraph.text)
            if not text:
                continue
            style_name = (paragraph.style.name or "").lower()
            if style_name.startswith("heading") or style_name.startswith("标题"):
                current_section = text
            blocks.append(ParsedBlock(text=text, section=current_section))
        for table in doc.tables:
            markdown = table_to_markdown([[cell.text for cell in row.cells] for row in table.rows])
            if markdown:
                blocks.append(ParsedBlock(text=markdown, section=current_section, kind="table"))
        return ParsedDocument(doc_id=doc_id, title=title or path.stem, source_type="docx", blocks=blocks)

    def _parse_xlsx(self, path: Path, doc_id: str, title: str | None) -> ParsedDocument:
        from openpyxl import load_workbook

        workbook = load_workbook(str(path), read_only=True, data_only=True)
        blocks: list[ParsedBlock] = []
        for sheet in workbook.worksheets:
            rows = []
            for row_index, row in enumerate(sheet.iter_rows(values_only=True)):
                if row_index > 500:
                    break
                rows.append(row)
            markdown = table_to_markdown(rows)
            if markdown:
                blocks.append(ParsedBlock(text=markdown, section=sheet.title, kind="table"))
        workbook.close()
        return ParsedDocument(doc_id=doc_id, title=title or path.stem, source_type="xlsx", blocks=blocks)

    def _parse_image(self, path: Path, doc_id: str, title: str | None) -> ParsedDocument:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - depends on local OCR
            raise RuntimeError("OCR dependencies are not installed") from exc
        text = clean_text(pytesseract.image_to_string(Image.open(path), lang="chi_sim+eng"))
        return ParsedDocument(
            doc_id=doc_id,
            title=title or path.stem,
            source_type="image",
            blocks=[ParsedBlock(text=text, kind="image_text")],
            metadata={"ocr": True},
        )

    def _parse_text(self, path: Path, doc_id: str, title: str | None) -> ParsedDocument:
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        return ParsedDocument(doc_id=doc_id, title=title or path.stem, source_type="text", blocks=[ParsedBlock(text=text)])
