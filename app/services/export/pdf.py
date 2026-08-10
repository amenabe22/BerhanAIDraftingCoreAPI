"""Markdown-ish text → PDF bytes (reportlab)."""

from __future__ import annotations

import io
import re
from functools import lru_cache
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from app.logging_config import get_logger

log = get_logger("export_pdf")

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


@lru_cache(maxsize=1)
def _unicode_font_name() -> str:
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                pdfmetrics.registerFont(TTFont("ExportUnicode", path))
                log.info("export_pdf_font", extra={"font_path": path})
                return "ExportUnicode"
            except Exception as exc:
                log.warning("export_pdf_font_failed", extra={"path": path, "error": str(exc)})
    log.warning("export_pdf_font_fallback", extra={"font": "Helvetica"})
    return "Helvetica"


def _md_inline_to_html(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def markdown_to_pdf_bytes(markdown: str) -> bytes:
    font = _unicode_font_name()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ExportTitle",
        parent=styles["Heading1"],
        fontName=font,
        fontSize=14,
        leading=18,
        spaceAfter=10,
        alignment=TA_LEFT,
        textColor=colors.black,
    )
    h2_style = ParagraphStyle(
        "ExportH2",
        parent=styles["Heading2"],
        fontName=font,
        fontSize=12,
        leading=15,
        spaceAfter=8,
        alignment=TA_LEFT,
        textColor=colors.black,
    )
    body_style = ParagraphStyle(
        "ExportBody",
        parent=styles["Normal"],
        fontName=font,
        fontSize=10,
        leading=13,
        spaceAfter=6,
        alignment=TA_LEFT,
        textColor=colors.black,
    )

    story: list = []
    for raw in (markdown or "").split("\n"):
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue
        if line == "---":
            story.append(Spacer(1, 12))
            continue
        if line.startswith("# "):
            story.append(Paragraph(_md_inline_to_html(line[2:].strip()), title_style))
        elif line.startswith("## "):
            story.append(Paragraph(_md_inline_to_html(line[3:].strip()), h2_style))
        elif line.startswith("### "):
            story.append(Paragraph(_md_inline_to_html(line[4:].strip()), h2_style))
        else:
            story.append(Paragraph(_md_inline_to_html(line), body_style))

    if not story:
        story.append(Paragraph("(Empty document)", body_style))

    doc.build(story)
    data = buffer.getvalue()
    buffer.close()
    return data
