"""Compliance analysis endpoint: POST /analyze, POST /analyze-stream. Loads document from Qdrant by doc_id."""

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.logging_config import get_logger
from app.models.drafting.compliance import (
    ComplianceAnalysisRequest,
    ComplianceAnalysisResponse,
)
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


@router.post("/analyze-stream")
async def analyze_compliance_stream(request: ComplianceAnalysisRequest) -> StreamingResponse:
    """Run compliance analysis with Server-Sent Events for coarse progress (percentage and phase).

    Each line is ``data: <json>``. Events:

    - ``{"type":"progress","phase":str,"percent":int,"message":str}`` — pipeline milestones.
    - ``{"type":"token","content":str}`` — LLM output chunks while analysis is running.
    - ``{"type":"result","data":{...}}`` — full analysis (same shape as POST /analyze JSON body).
    - ``{"type":"error","message":str}`` — client or server error (stream then ends).

    Percent is approximate; the longest gap is usually during the main LLM call (between ``analyze`` and ``parse``).
    """
    loop = asyncio.get_running_loop()
    progress_q: asyncio.Queue[dict[str, str | int] | None] = asyncio.Queue()

    def progress_callback(event: dict[str, str | int]) -> None:
        asyncio.run_coroutine_threadsafe(progress_q.put({"type": "progress", **event}), loop)

    def token_callback(content: str) -> None:
        asyncio.run_coroutine_threadsafe(
            progress_q.put({"type": "token", "content": content}), loop
        )

    def run() -> ComplianceAnalysisResponse | None:
        try:
            blocks = get_document_blocks_by_doc_id(request.doc_id)
            if not blocks:
                return None
            agent = ComplianceAnalysisAgent()
            return agent.analyze_document(
                document_blocks=blocks,
                language=request.language or "en",
                document_type=None,
                check_level=request.check_level,
                progress_callback=progress_callback,
                token_callback=token_callback,
            )
        finally:
            asyncio.run_coroutine_threadsafe(progress_q.put(None), loop)

    async def event_stream():
        task = asyncio.create_task(asyncio.to_thread(run))
        while True:
            ev = await progress_q.get()
            if ev is None:
                break
            yield f"data: {json.dumps(ev)}\n\n"
        try:
            result = await task
        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return
        except Exception as e:
            log.error("compliance_analyze_stream_error", extra={"error": str(e)})
            yield f"data: {json.dumps({'type': 'error', 'message': 'Compliance analysis failed'})}\n\n"
            return
        if result is None:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Document not found or has no content'})}\n\n"
            return
        payload = result.model_dump(mode="json")
        yield f"data: {json.dumps({'type': 'result', 'data': payload})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
