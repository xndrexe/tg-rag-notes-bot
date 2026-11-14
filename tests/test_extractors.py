"""Тесты на извлечение и чанкование текста."""
from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
from docx import Document as DocxDocument
from pypdf import PdfWriter

from tg_rag_bot.extractors import ExtractError, chunk_text, extract_text


def _make_docx(path: Path, paragraphs: list[str]) -> None:
    doc = DocxDocument()
    for para in paragraphs:
        doc.add_paragraph(para)
    doc.save(str(path))


def _make_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as fh:
        writer.write(fh)


def test_extract_markdown_passthrough(tmp_path: Path) -> None:
    md = tmp_path / "note.md"
    md.write_text("# Заголовок\n\nЭто текст заметки.\n", encoding="utf-8")
    assert "Заголовок" in extract_text(md)
    assert "заметки" in extract_text(md)


def test_extract_docx(tmp_path: Path) -> None:
    path = tmp_path / "doc.docx"
    _make_docx(path, ["Первый абзац.", "Второй абзац."])
    text = extract_text(path)
    assert "Первый абзац." in text
    assert "Второй абзац." in text


def test_extract_pdf_without_text(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    _make_pdf(path)
    with pytest.raises(ExtractError):
        extract_text(path)


def test_extract_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "image.jpg"
    path.write_bytes(b"\xff\xd8\xff")
    with pytest.raises(ExtractError):
        extract_text(path)


def test_chunk_short_text_stays_in_one() -> None:
    chunks = chunk_text("Короткий текст.", chunk_size=600, chunk_overlap=80)
    assert chunks == ["Короткий текст."]


def test_chunk_paragraphs_combined() -> None:
    text = "Абзац один." + "\n\n" + "Абзац два." + "\n\n" + "Абзац три."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert "Абзац один." in chunks[0]
    assert "Абзац три." in chunks[0]


def test_chunk_splits_when_oversized() -> None:
    text = "A" * 1500
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk) <= 500
    for prev, nxt in pairwise(chunks):
        assert prev[-50:] == nxt[:50]


def test_chunk_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        chunk_text("a", chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        chunk_text("a", chunk_size=100, chunk_overlap=100)
