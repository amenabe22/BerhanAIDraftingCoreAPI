"""Semantic document edit endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.drafting.edit import EditRequest, EditResult
from app.services.drafting.editing.agent import SemanticEditAgent
from app.services.drafting.editing.tiptap import envelope_from_blocks

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

    Accepts drafting blocks (preferred) or legacy TipTap doc_json + instruction;
    returns patched blocks document, operations, diff, and metrics.
    """
    try:
        if request.blocks is not None:
            doc = envelope_from_blocks(
                [b.model_dump() if hasattr(b, "model_dump") else dict(b) for b in request.blocks]
            )
        else:
            doc = request.doc_json or {}

        agent = _get_agent()
        result = await agent.edit_document(
            doc_json=doc,
            instruction=request.instruction,
            doc_id=request.doc_id,
            document_language=request.document_language,
            compliance_context=request.compliance_context,
        )
        if isinstance(result.get("document"), dict) and "blocks" in result["document"]:
            result.setdefault("blocks", result["document"].get("blocks"))
        return EditResult(**result)
    except Exception as exc:
        logger.exception("Semantic edit endpoint error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
