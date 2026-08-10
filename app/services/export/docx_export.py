"""Markdown-ish text → DOCX bytes (python-docx)."""

from __future__ import annotations

import io
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


def _strip_md(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text


def markdown_to_docx_bytes(markdown: str) -> bytes:
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.6)
        section.bottom_margin = Inches(0.6)
        section.left_margin = Inches(0.7)
        section.right_margin = Inches(0.7)

    for raw in (markdown or "").split("\n"):
        line = raw.strip()
        if not line or line == "---":
            continue

        if line.startswith("# "):
            para = doc.add_paragraph()
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = para.add_run(_strip_md(line[2:].strip()))
            run.bold = True
            run.font.size = Pt(14)
            para.space_after = Pt(8)
        elif line.startswith("## ") or line.startswith("### "):
            text = line[3:].strip() if line.startswith("## ") else line[4:].strip()
            para = doc.add_paragraph()
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = para.add_run(_strip_md(text))
            run.bold = True
            run.font.size = Pt(12)
            para.space_after = Pt(6)
        elif line.startswith("- "):
            para = doc.add_paragraph(_strip_md(line[2:].strip()), style="List Bullet")
            para.space_after = Pt(2)
        elif re.match(r"^\d+\.\s+", line):
            text = re.sub(r"^\d+\.\s+", "", line)
            para = doc.add_paragraph(_strip_md(text), style="List Number")
            para.space_after = Pt(2)
        else:
            para = doc.add_paragraph(_strip_md(line))
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            para.space_after = Pt(4)

    if len(doc.paragraphs) == 0:
        doc.add_paragraph("(Empty document)")

    buffer = io.BytesIO()
    doc.save(buffer)
    data = buffer.getvalue()
    buffer.close()
    return data
