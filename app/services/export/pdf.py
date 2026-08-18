"""Markdown-ish text → PDF bytes (reportlab)."""

from __future__ import annotations

import io
import re
from functools import lru_cache
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

from app.logging_config import get_logger
from app.services.export.md_lines import classify_md_line, heading_text

log = get_logger("export_pdf")

_EXPORT_FONT = "ExportUnicode"
_ETHIOPIC_FONT = "ExportEthiopic"
_FONTS_DIR = Path(__file__).resolve().parent / "fonts"

INK = colors.black

_ETHIOPIC_RE = re.compile(
    r"[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]+"
)

# Latin first (with real bold) so English headings have clear weight contrast.
# Abyssinica SIL is a last-resort Unicode face, not the English body font.
_LATIN_FONT_PAIRS = [
    (
        str(_FONTS_DIR / "NotoSans-Regular.ttf"),
        str(_FONTS_DIR / "NotoSans-Bold.ttf"),
    ),
    (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
    (
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ),
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
    (str(_FONTS_DIR / "AbyssinicaSIL-Regular.ttf"), None),
    ("/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-Regular.ttf", None),
    ("/usr/share/fonts/truetype/freefont/FreeSans.ttf", None),
]


def _register_ttfont(name: str, path: str) -> None:
    kwargs: dict = {}
    if path.lower().endswith(".ttc"):
        kwargs["subfontIndex"] = 0
    pdfmetrics.registerFont(TTFont(name, path, **kwargs))


def _register_latin_family(regular: str, bold: str | None) -> None:
    _register_ttfont(_EXPORT_FONT, regular)
    bold_name = f"{_EXPORT_FONT}-Bold"
    if bold and Path(bold).is_file():
        _register_ttfont(bold_name, bold)
        pdfmetrics.registerFontFamily(
            _EXPORT_FONT,
            normal=_EXPORT_FONT,
            bold=bold_name,
            italic=_EXPORT_FONT,
            boldItalic=bold_name,
        )
    else:
        pdfmetrics.registerFontFamily(
            _EXPORT_FONT,
            normal=_EXPORT_FONT,
            bold=_EXPORT_FONT,
            italic=_EXPORT_FONT,
            boldItalic=_EXPORT_FONT,
        )


@lru_cache(maxsize=1)
def _unicode_font_name() -> str:
    for regular, bold in _LATIN_FONT_PAIRS:
        if Path(regular).is_file():
            try:
                _register_latin_family(regular, bold)
                log.info("export_pdf_font", extra={"font_path": regular, "bold_path": bold})
                return _EXPORT_FONT
            except Exception as exc:
                log.warning("export_pdf_font_failed", extra={"path": regular, "error": str(exc)})
    log.warning("export_pdf_font_fallback", extra={"font": "Helvetica"})
    return "Helvetica"


@lru_cache(maxsize=1)
def _ethiopic_font_name() -> str | None:
    path = _FONTS_DIR / "EthiopicLessan-Regular.ttf"
    if not path.is_file():
        return None
    try:
        _register_ttfont(_ETHIOPIC_FONT, str(path))
        pdfmetrics.registerFontFamily(
            _ETHIOPIC_FONT,
            normal=_ETHIOPIC_FONT,
            bold=_ETHIOPIC_FONT,
            italic=_ETHIOPIC_FONT,
            boldItalic=_ETHIOPIC_FONT,
        )
        log.info("export_pdf_ethiopic_font", extra={"font_path": str(path)})
        return _ETHIOPIC_FONT
    except Exception as exc:
        log.warning("export_pdf_ethiopic_font_failed", extra={"error": str(exc)})
        return None


def _md_inline_to_html(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def _pdf_html(text: str) -> str:
    html = _md_inline_to_html(text)
    ethiopic = _ethiopic_font_name()
    if not ethiopic:
        return html
    return _ETHIOPIC_RE.sub(
        lambda m: f'<font name="{ethiopic}">{m.group(0)}</font>',
        html,
    )


def build_paragraph_styles(font: str) -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "ExportBody",
        parent=styles["Normal"],
        fontName=font,
        fontSize=10.5,
        leading=16.5,
        spaceAfter=8,
        alignment=TA_LEFT,
        textColor=INK,
    )
    return {
        "title": ParagraphStyle(
            "ExportTitle",
            parent=styles["Heading1"],
            fontName=font,
            fontSize=22,
            leading=28,
            spaceBefore=0,
            spaceAfter=8,
            alignment=TA_CENTER,
            textColor=INK,
        ),
        "h2": ParagraphStyle(
            "ExportH2",
            parent=styles["Heading2"],
            fontName=font,
            fontSize=16,
            leading=22,
            spaceBefore=18,
            spaceAfter=8,
            alignment=TA_LEFT,
            textColor=INK,
        ),
        "h3": ParagraphStyle(
            "ExportH3",
            parent=styles["Heading3"],
            fontName=font,
            fontSize=13,
            leading=18,
            spaceBefore=14,
            spaceAfter=6,
            alignment=TA_LEFT,
            textColor=INK,
        ),
        "body": body,
        "list": ParagraphStyle(
            "ExportList",
            parent=body,
            leftIndent=16,
            spaceAfter=4,
            leading=15,
        ),
    }


def markdown_to_pdf_bytes(markdown: str) -> bytes:
    font = _unicode_font_name()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
    )
    styles = build_paragraph_styles(font)

    story: list = []
    for raw in (markdown or "").split("\n"):
        kind = classify_md_line(raw)
        if kind == "blank":
            story.append(Spacer(1, 6))
            continue
        if kind == "rule":
            story.append(Spacer(1, 8))
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.4,
                    color=colors.Color(0.85, 0.85, 0.85),
                    spaceBefore=4,
                    spaceAfter=12,
                )
            )
            continue
        if kind == "title":
            story.append(
                Paragraph(
                    f"<b>{_pdf_html(heading_text(raw, kind))}</b>",
                    styles["title"],
                )
            )
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=1.5,
                    color=colors.black,
                    spaceBefore=2,
                    spaceAfter=14,
                )
            )
        elif kind == "h2":
            story.append(
                Paragraph(
                    f"<b>{_pdf_html(heading_text(raw, kind))}</b>",
                    styles["h2"],
                )
            )
        elif kind == "h3":
            story.append(
                Paragraph(
                    f"<b>{_pdf_html(heading_text(raw, kind))}</b>",
                    styles["h3"],
                )
            )
        elif kind == "bullet":
            story.append(
                Paragraph("• " + _pdf_html(heading_text(raw, kind)), styles["list"])
            )
        elif kind == "ordered":
            story.append(Paragraph(_pdf_html(raw.strip()), styles["list"]))
        else:
            story.append(Paragraph(_pdf_html(raw.strip()), styles["body"]))

    if not story:
        story.append(Paragraph("(Empty document)", styles["body"]))

    doc.build(story)
    data = buffer.getvalue()
    buffer.close()
    return data
