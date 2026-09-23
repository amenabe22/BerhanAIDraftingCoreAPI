import asyncio
import json
import uuid
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from app.api.v1.endpoints.drafting.compliance import router as compliance_router
from app.api.v1.endpoints.drafting.edit import router as edit_router
from app.api.v1.endpoints.drafting.export import router as export_router
from app.api.v1.endpoints.drafting.generate import router as generate_router
from app.graph import (
    DOC_CONSULTANT_SYSTEM,
    LEGAL_ADVISOR_SYSTEM,
    LEGAL_AGENT_SYSTEM,
    get_doc_graph,
    get_graph,
)
from app.llm import SupportedModel, resolve_model
from app.logging_config import get_logger, setup_logging

setup_logging()
log = get_logger("chat")


class Language(str, Enum):
    amharic = "am"
    english = "en"
    oromo = "om"


_LANG_INSTRUCTION: dict[Language, str] = {
    Language.amharic: "Always respond in Amharic (አማርኛ).",
    Language.english: "Always respond in English.",
    Language.oromo: "Always respond in Oromo (Afaan Oromoo).",
}

_LANG_DEFAULT_INSTRUCTION = (
    "Detect the language of the user's message and always respond in that same language."
)


def _apply_language(system_prompt: str, language: Language | None) -> str:
    instruction = (
        _LANG_INSTRUCTION.get(language, _LANG_DEFAULT_INSTRUCTION)
        if language
        else _LANG_DEFAULT_INSTRUCTION
    )
    # Language instruction goes at the TOP so the LLM sees it first,
    # without disrupting the tool-call directives that follow.
    return f"LANGUAGE: {instruction}\n\n{system_prompt}"


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
    language: Language | None = None
    # Optional URL of any file/image supported by Gemini multimodal (JPEG, PNG,
    # GIF, WEBP, PDF, MP4, MP3, WAV, …). Null / omitted → text-only message.
    file_url: str | None = None
    # Model selection — must be one of the supported OpenRouter model IDs.
    # Omit to use the server default (GEMINI_MODEL env var).
    model: SupportedModel | None = None
    # Enable extended reasoning (effort=medium). Increases latency and cost.
    enable_reasoning: bool = False


class DocChatRequest(BaseModel):
    message: str
    doc_id: str
    thread_id: str | None = None
    language: Language | None = None
    # TipTap JSON for apply_document_edit. When omitted, edit tool reports no document.
    doc_json: dict | None = None
    document_language: str | None = None


app = FastAPI(title="Berhan Advisor Knowledge Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://docs.berhan.ai",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type", "Authorization", "Accept", "Origin"],
    max_age=86400,
)

app.include_router(compliance_router, prefix="/drafting")
app.include_router(edit_router, prefix="/drafting")
app.include_router(generate_router, prefix="/drafting")
app.include_router(export_router, prefix="/drafting")

_TEST_CLIENT_PATH = Path(__file__).resolve().parent.parent / "test_client.html"


@app.get("/")
async def root():
    if _TEST_CLIENT_PATH.exists():
        return FileResponse(_TEST_CLIENT_PATH)
    return {
        "message": "Berhan Advisor API",
        "docs": "/docs",
        "health": "/health",
        "legal_search_stream": "/legal-search/stream",
        "legal_agent_stream": "/legal-agent/stream",
        "doc_agent_stream": "/doc-agent/stream",
        "compliance_analyze": "/drafting/compliance/analyze",
        "compliance_analyze_stream": "/drafting/compliance/analyze-stream",
        "semantic_edit": "/drafting/edit",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


def _parse_citations(tool_messages: list[ToolMessage]) -> list[dict]:
    """Extract unique citations from all ToolMessages produced during the run.

    Handles two header formats:
      [Source: <document_id> | Article <item_id> | <title>]  – legal knowledge
      [Doc: <doc_id> | Block: <block_id> | Type: <type>]     – user doc blocks
    """
    seen: set[str] = set()
    citations: list[dict] = []
    for tm in tool_messages:
        content = tm.content or ""
        if not isinstance(content, str):
            continue
        for block in content.split("\n\n"):
            block = block.strip()
            is_source = block.startswith("[Source:")
            is_doc = block.startswith("[Doc:")
            if not is_source and not is_doc:
                continue
            header_end = block.find("]")
            if header_end == -1:
                continue
            header = block[1:header_end]
            text_body = block[header_end + 1 :].strip()
            parts = [p.strip() for p in header.split("|")]

            if is_source:
                doc_id = parts[0].replace("Source:", "").strip() if parts else ""
                item_id = parts[1].replace("Article", "").strip() if len(parts) > 1 else ""
                title = parts[2].strip() if len(parts) > 2 else ""
                key = f"source:{doc_id}:{item_id}"
                if key in seen:
                    continue
                seen.add(key)
                citations.append(
                    {
                        "document_id": doc_id,
                        "item_id": item_id,
                        "title": title,
                        "content": text_body[:300],
                    }
                )
            else:  # [Doc: ...]
                doc_id = parts[0].replace("Doc:", "").strip() if parts else ""
                block_id = parts[1].replace("Block:", "").strip() if len(parts) > 1 else ""
                block_type = parts[2].replace("Type:", "").strip() if len(parts) > 2 else ""
                key = f"doc:{doc_id}:{block_id}"
                if key in seen:
                    continue
                seen.add(key)
                citations.append(
                    {
                        "doc_id": doc_id,
                        "block_id": block_id,
                        "type": block_type,
                        "content": text_body[:300],
                    }
                )
    return citations


def _is_new_thread(graph, thread_id: str) -> bool:
    """Returns True if this thread has no prior checkpoint (i.e. first turn)."""
    try:
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        return not state or not state.values.get("messages")
    except Exception:
        return True


def _human_message_content(message: str, file_url: str | None) -> str | list:
    """Build HumanMessage content: plain text, or text + image for Gemini multimodal."""
    url = (file_url or "").strip() or None
    if not url:
        return message
    return [
        {"type": "text", "text": message},
        {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
    ]


def _user_facing_stream_error(message: str) -> str:
    lower = (message or "").strip().lower()
    if "json error injected into sse stream" in lower:
        return "The AI service had a temporary connection issue. Please try again."
    return message or "Upstream error"


async def _stream_endpoint(
    request: ChatRequest,
    event_type: str,
    system_prompt: str,
    graph,
    status_message: str = "Searching legal knowledge base…",
    *,
    extra_configurable: dict | None = None,
) -> StreamingResponse:
    thread_id = request.thread_id or str(uuid.uuid4())
    file_url = (request.file_url or "").strip() or None
    resolved_model = resolve_model(getattr(request, "model", None))
    enable_reasoning = getattr(request, "enable_reasoning", False)
    log.info(
        event_type,
        extra={
            "event": event_type,
            "user_message": request.message,
            "thread_id": thread_id,
            "has_file_url": bool(file_url),
            "model": resolved_model,
            "enable_reasoning": enable_reasoning,
            "has_doc_json": bool((extra_configurable or {}).get("doc_json")),
        },
    )

    human_content = _human_message_content(request.message, request.file_url)

    # Only inject the SystemMessage on the very first turn of a thread.
    # On follow-up turns the checkpointer already has the system message stored.
    first_turn = _is_new_thread(graph, thread_id)
    if first_turn:
        initial_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]
    else:
        initial_messages = [HumanMessage(content=human_content)]

    async def event_stream():
        yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': thread_id})}\n\n"
        yield f"data: {json.dumps({'type': 'model', 'model': resolved_model, 'enable_reasoning': enable_reasoning})}\n\n"

        config = {
            "configurable": {
                "thread_id": thread_id,
                "model": resolved_model,
                "enable_reasoning": enable_reasoning,
                **(extra_configurable or {}),
            }
        }
        inputs = {"messages": initial_messages}
        tool_msg_buffers: dict[str, str] = {}
        status_sent = False
        saw_custom_tokens = False
        fallback_token_chunks: list[str] = []
        # Capture the last grounding event emitted by the ground node (if any)
        last_grounding_event: dict | None = None
        loop = asyncio.get_running_loop()
        stream_q: asyncio.Queue[tuple[str, object] | None] = asyncio.Queue()

        def run_graph_stream() -> None:
            try:
                for item in graph.stream(
                    inputs,
                    stream_mode=["custom", "messages"],
                    config=config,
                ):
                    asyncio.run_coroutine_threadsafe(stream_q.put(item), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    stream_q.put(("error", str(exc))),
                    loop,
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(stream_q.put(None), loop).result()

        stream_task = asyncio.create_task(asyncio.to_thread(run_graph_stream))

        try:
            while True:
                item = await stream_q.get()
                if item is None:
                    break

                mode, payload = item
                if mode == "custom":
                    if isinstance(payload, dict):
                        ptype = payload.get("type")
                        if ptype == "token":
                            content = payload.get("content", "")
                            if content:
                                saw_custom_tokens = True
                                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                        elif ptype == "thinking":
                            # Reasoning/thinking tokens from extended-reasoning models.
                            # Forwarded as-is so the client can populate the thinking panel.
                            content = payload.get("content", "")
                            if content:
                                yield f"data: {json.dumps({'type': 'thinking', 'content': content})}\n\n"
                        elif ptype == "status":
                            # Pass through status messages from grounding / repair
                            msg = payload.get("message", "")
                            if msg:
                                yield f"data: {json.dumps({'type': 'status', 'message': msg})}\n\n"
                        elif ptype == "grounding":
                            # Capture most-recent grounding event — emit at end of stream
                            last_grounding_event = {
                                "ok": payload.get("ok", True),
                                "repair_attempted": payload.get("repair_attempted", False),
                                "reason": payload.get("reason"),
                            }
                        elif ptype == "edit_started":
                            yield f"data: {json.dumps(payload)}\n\n"
                        elif ptype == "edit_result":
                            # Full TipTap document + ops for CoreAPI / clients
                            yield f"data: {json.dumps(payload, default=str)}\n\n"
                    continue

                if mode == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': _user_facing_stream_error(str(payload))})}\n\n"
                    return

                if mode != "messages":
                    continue

                chunk, meta = payload
                node = meta.get("langgraph_node", "") if isinstance(meta, dict) else ""
                content = getattr(chunk, "content", None)

                # Detect the moment the agent decides to call a tool and notify the client
                if not status_sent and getattr(chunk, "tool_calls", None) and node == "agent":
                    status_sent = True
                    tool_names = []
                    for tc in chunk.tool_calls or []:
                        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                        if name:
                            tool_names.append(name)
                    msg = status_message
                    if "apply_document_edit" in tool_names:
                        msg = "Applying document edit…"
                    yield f"data: {json.dumps({'type': 'status', 'message': msg})}\n\n"

                if isinstance(chunk, ToolMessage):
                    tid = getattr(chunk, "tool_call_id", None) or chunk.id or ""
                    tool_msg_buffers[tid] = tool_msg_buffers.get(tid, "") + (content or "")
                elif content and node == "agent" and not isinstance(chunk, ToolMessage):
                    # Buffer message-mode content and only emit it if custom token
                    # events never arrive. On Python 3.10 sync graph.stream()
                    # yields both messages and custom events for the same chunk.
                    fallback_token_chunks.append(content)

        except Exception as exc:
            log.error(
                "stream error",
                extra={"event": "stream_error", "error": str(exc), "thread_id": thread_id},
            )
            yield f"data: {json.dumps({'type': 'error', 'message': _user_facing_stream_error(str(exc))})}\n\n"
            return
        finally:
            await stream_task

        collected: list[ToolMessage] = [
            ToolMessage(content=text, tool_call_id=tid) for tid, text in tool_msg_buffers.items()
        ]
        if not saw_custom_tokens:
            for content in fallback_token_chunks:
                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

        # Existing citations event — unchanged
        citations = _parse_citations(collected)
        if citations:
            yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

        # Additive grounding event — old clients ignore unknown types
        if last_grounding_event is not None:
            yield f"data: {json.dumps({'type': 'grounding', **last_grounding_event})}\n\n"

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


@app.post("/legal-search/stream")
async def legal_search_stream(request: ChatRequest):
    """Streaming legal search: conversational answer synthesized from retrieved sources, with citations at the end."""
    try:
        graph = get_graph()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return await _stream_endpoint(
        request,
        event_type="legal_search_request",
        system_prompt=_apply_language(LEGAL_AGENT_SYSTEM, request.language),
        graph=graph,
    )


@app.post("/legal-agent/stream")
async def legal_agent_stream(request: ChatRequest):
    """Streaming legal consultant: advisory answers with citations at the end."""
    try:
        graph = get_graph()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return await _stream_endpoint(
        request,
        event_type="legal_agent_request",
        system_prompt=_apply_language(LEGAL_ADVISOR_SYSTEM, request.language),
        graph=graph,
    )


@app.post("/doc-agent/stream")
async def doc_agent_stream(request: DocChatRequest):
    """Hybrid document + law consultant agent (optional semantic edit).

    Requires ``doc_id`` — searches are scoped strictly to that document.
    Also searches the Ethiopian legal knowledge base for applicable law.
    When ``doc_json`` is provided, the agent may call ``apply_document_edit``.
    """
    enable_edit = isinstance(request.doc_json, dict) and request.doc_json.get("type") == "doc"
    try:
        graph = get_doc_graph(request.doc_id, enable_edit=enable_edit)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    doc_lang = request.document_language
    if not doc_lang and request.language:
        doc_lang = (
            request.language.value if hasattr(request.language, "value") else str(request.language)
        )

    return await _stream_endpoint(
        ChatRequest(
            message=request.message, thread_id=request.thread_id, language=request.language
        ),
        event_type="doc_agent_request",
        system_prompt=_apply_language(DOC_CONSULTANT_SYSTEM, request.language),
        graph=graph,
        status_message="Searching documents and legal knowledge base…",
        extra_configurable={
            "doc_id": request.doc_id,
            "doc_json": request.doc_json if enable_edit else None,
            "document_language": doc_lang,
        },
    )
