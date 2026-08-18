"""Structured one-shot Doc-Gen API: POST /generate (SSE) + legacy /generate/stream."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.logging_config import get_logger
from app.llm import resolve_model
from app.models.drafting.generate import GenerateRequest, StructuredGenerateRequest
from app.services.export.contabo import ContaboNotConfiguredError
from app.services.export.service import export_document
from app.services.generation.agent import GenerationAgent
from app.services.generation.requirements import (
    build_requirements,
    build_synthetic_prompt,
)
from app.services.generation.sse import SSEEmitter, format_sse
from app.services.generation.thread_store import thread_store
from app.services.generation.metadata_schema import load_drafting_metadata_schema

log = get_logger("generation_endpoint")

router = APIRouter(prefix="/generate", tags=["generation"])
_agent = GenerationAgent()


@router.get("/metadata-schema")
async def drafting_metadata_schema() -> dict[str, Any]:
    """Return the canonical JSON Schema (draft 2020-12) for ``metadata``."""
    return load_drafting_metadata_schema()


def _sse_response(queue: asyncio.Queue, thread_id: str, run_coro) -> StreamingResponse:
    async def event_stream():
        task = asyncio.create_task(run_coro)
        try:
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                yield format_sse(ev)
        finally:
            await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "X-Thread-ID": thread_id,
        },
    )


@router.post("")
@router.post("/", include_in_schema=False)
async def generate_structured(request: StructuredGenerateRequest) -> StreamingResponse:
    """One-shot agent drafting: structured JSON payload → TipTap SSE → Contabo export URLs.

    Required: ``doc_type``, ``type`` (pdf/docx), ``metadata`` (camelCase JSON Schema —
    see ``GET /drafting/generate/metadata-schema``; required keys ``title``, ``parties``,
    ``governingLaw``).
    ``language``: ``en`` | ``am`` (aliases ``amh`` / ``amharic``). Preference is pinned.

    Does **not** run clarification. Emits ``document_generated`` then ``export_ready``
    (or ``export_skipped`` if Contabo is unavailable).
    """
    state = thread_store.get_or_create(request.thread_id)
    thread_id = state.thread_id
    resolved_model = resolve_model(request.model)
    requirements = build_requirements(request)
    synthetic = build_synthetic_prompt(request, requirements)
    formats = list(request.type)

    log.info(
        "structured_generate_request",
        extra={
            "event": "structured_generate_request",
            "thread_id": thread_id,
            "doc_type": request.doc_type,
            "formats": formats,
            "model": resolved_model,
            "language": requirements.get("language"),
        },
    )

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    emitter = SSEEmitter(queue)

    async def run() -> None:
        try:
            await emitter.emit({"type": "thread_id", "thread_id": thread_id})
            await emitter.emit(
                {
                    "type": "model",
                    "model": resolved_model,
                    "enable_reasoning": request.enable_reasoning,
                }
            )
            await emitter.status("Generating document from structured requirements…")

            document = await _agent.generate_from_requirements(
                thread_id,
                emitter,
                requirements=requirements,
                synthetic_prompt=synthetic,
                model=request.model,
                enable_reasoning=request.enable_reasoning,
            )

            # Strip internal annotation keys before export
            export_doc = {
                k: v
                for k, v in (document or {}).items()
                if not str(k).startswith("_")
            }

            await emitter.status("Exporting to Contabo…")
            try:
                export_result = export_document(
                    export_doc,
                    formats=formats,  # type: ignore[arg-type]
                    filename=request.filename or request.doc_type,
                )
                await emitter.emit(
                    {
                        "type": "export_ready",
                        "thread_id": thread_id,
                        "filename": export_result.get("filename"),
                        "pdf_url": export_result.get("pdf_url"),
                        "docx_url": export_result.get("docx_url"),
                        "keys": export_result.get("keys") or {},
                    }
                )
            except ContaboNotConfiguredError as exc:
                await emitter.emit(
                    {
                        "type": "export_skipped",
                        "thread_id": thread_id,
                        "message": str(exc),
                    }
                )
            except Exception as exc:
                log.error(
                    "structured_export_error",
                    extra={"error": str(exc), "thread_id": thread_id},
                    exc_info=True,
                )
                await emitter.emit(
                    {
                        "type": "export_skipped",
                        "thread_id": thread_id,
                        "message": f"Export failed: {exc}",
                    }
                )
        except Exception as exc:
            log.error(
                "structured_generate_error",
                extra={"error": str(exc), "thread_id": thread_id},
                exc_info=True,
            )
            await emitter.error(str(exc) or "Document generation failed")
        finally:
            await emitter.close()

    return _sse_response(queue, thread_id, run())


@router.post("/stream")
async def generate_stream(request: GenerateRequest) -> StreamingResponse:
    """Legacy conversational generation (analyze / clarify / finalize)."""
    state = thread_store.get_or_create(request.thread_id)
    thread_id = state.thread_id
    resolved_model = resolve_model(request.model)
    language = request.language.value if request.language else None

    log.info(
        "generate_request",
        extra={
            "event": "generate_request",
            "thread_id": thread_id,
            "action": request.action,
            "model": resolved_model,
            "language": language,
            "has_file_url": bool(request.file_url),
        },
    )

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    emitter = SSEEmitter(queue)

    async def run() -> None:
        try:
            await emitter.emit({"type": "thread_id", "thread_id": thread_id})
            await emitter.emit(
                {
                    "type": "model",
                    "model": resolved_model,
                    "enable_reasoning": request.enable_reasoning,
                }
            )

            context: dict[str, Any] = {}
            if language:
                context["language"] = language
            if request.num_pages is not None:
                context["num_pages"] = request.num_pages
            if request.document_type:
                context["document_type"] = request.document_type

            if request.action == "start":
                analysis = await _agent.analyze_requirements(
                    thread_id,
                    request.message,
                    emitter,
                    context=context,
                    model=request.model,
                    enable_reasoning=request.enable_reasoning,
                    file_url=request.file_url,
                )
                # Belt-and-suspenders: re-pin preferred language like Advisor Core API.
                if language:
                    pinned = {"language": language}
                    if analysis.get("extracted_info") is None:
                        analysis["extracted_info"] = {}
                    analysis["extracted_info"]["language"] = language
                    analysis["language"] = language
                    thread_store.update_requirements(thread_id, pinned)
                if analysis.get("ready_to_generate"):
                    # Wait for action=finalize (green Start button). Do not auto-draft.
                    await emitter.ready_to_generate(
                        thread_id, analysis.get("response_message", "")
                    )
                else:
                    await emitter.clarification_needed(
                        thread_id,
                        questions=analysis.get("questions") or [],
                        response_message=analysis.get("response_message", ""),
                        ready_to_generate=False,
                    )

            elif request.action == "message":
                if language:
                    thread_store.update_requirements(thread_id, {"language": language})
                result = await _agent.process_clarification(
                    thread_id,
                    request.message,
                    emitter,
                    model=request.model,
                    enable_reasoning=request.enable_reasoning,
                    file_url=request.file_url,
                )
                if result.get("ready_to_generate"):
                    # Wait for action=finalize (green Start button). Do not auto-draft.
                    await emitter.ready_to_generate(
                        thread_id, result.get("response_message", "")
                    )
                else:
                    await emitter.clarification_needed(
                        thread_id,
                        questions=result.get("questions") or [],
                        response_message=result.get("response_message", ""),
                        ready_to_generate=False,
                    )

            elif request.action == "finalize":
                if request.message.strip():
                    thread_store.add_user_message(thread_id, request.message)
                state_now = thread_store.get(thread_id)
                if state_now and not state_now.extracted_requirements:
                    reqs: dict[str, Any] = {
                        "document_type": request.document_type or "contract",
                        "language": language or "en",
                        "summary": request.message,
                    }
                    if request.num_pages is not None:
                        reqs["num_pages"] = request.num_pages
                    thread_store.set_requirements(thread_id, reqs)
                elif state_now and language:
                    thread_store.update_requirements(thread_id, {"language": language})

                await _agent.generate_document(
                    thread_id,
                    emitter,
                    model=request.model,
                    enable_reasoning=request.enable_reasoning,
                )
            else:
                await emitter.error(f"Unknown action: {request.action}")
        except Exception as exc:
            log.error(
                "generate_stream_error",
                extra={"error": str(exc), "thread_id": thread_id},
                exc_info=True,
            )
            await emitter.error(str(exc) or "Document generation failed")
        finally:
            await emitter.close()

    return _sse_response(queue, thread_id, run())
