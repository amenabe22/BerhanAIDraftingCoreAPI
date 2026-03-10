"""Compliance analysis endpoint: POST /analyze. Loads document from Qdrant by doc_id (block_id, text, type per block)."""

from fastapi import APIRouter, HTTPException

from app.logging_config import get_logger
from app.models.drafting.compliance import ComplianceAnalysisRequest, ComplianceAnalysisResponse
from app.retrieval import get_document_blocks_by_doc_id
from app.services.drafting.compliance.analysis_agent import ComplianceAnalysisAgent

log = get_logger("compliance_endpoint")

router = APIRouter(prefix="/compliance", tags=["compliance"])


@router.post("/analyze", response_model=ComplianceAnalysisResponse)
def analyze_compliance(request: ComplianceAnalysisRequest) -> ComplianceAnalysisResponse:
    """Run sync compliance analysis. Accepts doc_id and optional check_level (quick/standard/deep); loads document blocks from the doc collection (Qdrant) so block_id and type are available for context and citations."""
    try:
        blocks = get_document_blocks_by_doc_id(request.doc_id)
        if not blocks:
            raise HTTPException(
                status_code=404,
                detail="Document not found or has no content",
            )
        agent = ComplianceAnalysisAgent()
        return agent.analyze_document(
            document_blocks=blocks,
            language=request.language or "en",
            document_type=None,
            check_level=request.check_level,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.error("compliance_analyze_error", extra={"error": str(e)})
        raise HTTPException(status_code=503, detail="Compliance analysis failed") from e
