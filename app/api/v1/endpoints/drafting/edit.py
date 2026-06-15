"""Semantic document edit endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.drafting.edit import EditRequest, EditResult
from app.services.drafting.editing.agent import SemanticEditAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/edit", tags=["semantic-edit"])

_agent: SemanticEditAgent | None = None


def _get_agent() -> SemanticEditAgent:
    global _agent
    if _agent is None:
        _agent = SemanticEditAgent()
    return _agent


@router.post("", response_model=EditResult)
async def semantic_edit(request: EditRequest) -> EditResult:
    """
    High-accuracy semantic document edit.

    Accepts full TipTap document + instruction; returns patched document,
    operations, diff, and metrics compatible with BerhanAdvisorCoreAPI Celery task.
    """
    try:
        agent = _get_agent()
        result = await agent.edit_document(
            doc_json=request.doc_json,
            instruction=request.instruction,
            doc_id=request.doc_id,
            document_language=request.document_language,
            compliance_context=request.compliance_context,
        )
        return EditResult(**result)
    except Exception as exc:
        logger.exception("Semantic edit endpoint error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
