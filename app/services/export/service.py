"""Orchestrate TipTap → PDF/DOCX → Contabo upload."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from app.logging_config import get_logger
from app.services.export.contabo import ContaboNotConfiguredError, contabo_configured, upload_bytes
from app.services.export.docx_export import markdown_to_docx_bytes
from app.services.export.pdf import markdown_to_pdf_bytes
from app.services.export.tiptap_text import tiptap_to_markdown, validate_tiptap_document

log = get_logger("export_service")

ExportFormat = Literal["pdf", "docx"]

_CONTENT_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _safe_basename(filename: str | None) -> str:
    base = (filename or "document").strip() or "document"
    base = re.sub(r"[^\w.\-]+", "_", base, flags=re.UNICODE)
    base = base.rstrip("._") or "document"
    # strip extension if caller included one
    lower = base.lower()
    for ext in (".pdf", ".docx", ".doc"):
        if lower.endswith(ext):
            base = base[: -len(ext)]
            break
    return base[:120]


def export_document(
    document: dict[str, Any],
    *,
    formats: list[ExportFormat] | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    """Convert TipTap doc and upload requested formats. Returns URLs + keys."""
    validate_tiptap_document(document)
    if not contabo_configured():
        raise ContaboNotConfiguredError(
            "Contabo/S3 is not configured. Set S3_ENDPOINT_URL, "
            "S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, and S3_BUCKET_NAME."
        )

    requested = list(formats) if formats is not None else ["pdf", "docx"]
    unique: list[ExportFormat] = []
    for fmt in requested:
        if fmt in ("pdf", "docx") and fmt not in unique:
            unique.append(fmt)  # type: ignore[arg-type]
    if not unique:
        raise ValueError('formats must include at least one of "pdf" or "docx"')

    markdown = tiptap_to_markdown(document)
    if not markdown.strip():
        raise ValueError("document has no exportable text content")

    basename = _safe_basename(filename)
    now = datetime.now(timezone.utc)
    prefix = f"exports/{now:%Y}/{now:%m}/{uuid4().hex[:12]}-{basename}"

    pdf_url: str | None = None
    docx_url: str | None = None
    keys: dict[str, str] = {}

    if "pdf" in unique:
        key = f"{prefix}.pdf"
        data = markdown_to_pdf_bytes(markdown)
        pdf_url = upload_bytes(key, data, _CONTENT_TYPES["pdf"])
        keys["pdf"] = key
        log.info("exported_pdf", extra={"key": key, "bytes": len(data)})

    if "docx" in unique:
        key = f"{prefix}.docx"
        data = markdown_to_docx_bytes(markdown)
        docx_url = upload_bytes(key, data, _CONTENT_TYPES["docx"])
        keys["docx"] = key
        log.info("exported_docx", extra={"key": key, "bytes": len(data)})

    return {
        "filename": basename,
        "pdf_url": pdf_url,
        "docx_url": docx_url,
        "keys": keys,
    }
