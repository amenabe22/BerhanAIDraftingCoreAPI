"""TipTap export endpoint: POST /export → Contabo PDF/DOCX URLs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.logging_config import get_logger
from app.models.drafting.export import ExportRequest, ExportResponse
from app.services.export.contabo import ContaboNotConfiguredError
from app.services.export.service import export_document

log = get_logger("export_endpoint")

router = APIRouter(prefix="/export", tags=["export"])


@router.post("", response_model=ExportResponse)
@router.post("/", response_model=ExportResponse, include_in_schema=False)
async def export_tiptap(request: ExportRequest) -> ExportResponse:
    """Convert TipTap JSON to PDF and/or DOCX, upload to Contabo, return URLs."""
    try:
        result = export_document(
            request.document,
            formats=request.formats,
            filename=request.filename,
        )
        return ExportResponse(**result)
    except ContaboNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.error("export_failed", extra={"error": str(exc)}, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Export failed: {exc}",
        ) from exc
