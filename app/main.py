import json
import uuid
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from app.api.v1.endpoints.drafting.compliance import router as compliance_router
from app.graph import (
    DOC_CONSULTANT_SYSTEM,
    LEGAL_ADVISOR_SYSTEM,
    LEGAL_AGENT_SYSTEM,
    get_doc_graph,
    get_graph,
)
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


class DocChatRequest(BaseModel):
    message: str
    doc_id: str
    thread_id: str | None = None
    language: Language | None = None


app = FastAPI(title="Berhan Advisor Knowledge Agent")

app.include_router(compliance_router, prefix="/drafting")

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


async def _stream_endpoint(
    request: ChatRequest,
    event_type: str,
    system_prompt: str,
    graph,
    status_message: str = "Searching legal knowledge base…",
) -> StreamingResponse:
    thread_id = request.thread_id or str(uuid.uuid4())
    log.info(
        event_type,
        extra={"event": event_type, "user_message": request.message, "thread_id": thread_id},
    )

    # Only inject the SystemMessage on the very first turn of a thread.
    # On follow-up turns the checkpointer already has the system message stored.
    first_turn = _is_new_thread(graph, thread_id)
    if first_turn:
        initial_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.message),
        ]
    else:
        initial_messages = [HumanMessage(content=request.message)]

    async def event_stream():
        yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': thread_id})}\n\n"

        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": initial_messages}
        tool_msg_buffers: dict[str, str] = {}
        status_sent = False

        try:
            async for chunk, meta in graph.astream(
                inputs,
                stream_mode="messages",
                config=config,
            ):
                node = meta.get("langgraph_node", "") if isinstance(meta, dict) else ""
                content = getattr(chunk, "content", None)

                # Detect the moment the agent decides to call a tool and notify the client
                if (
                    not status_sent
                    and isinstance(chunk, AIMessage)
                    and getattr(chunk, "tool_calls", None)
                    and node == "agent"
                ):
                    status_sent = True
                    yield f"data: {json.dumps({'type': 'status', 'message': status_message})}\n\n"

                if isinstance(chunk, ToolMessage):
                    tid = getattr(chunk, "tool_call_id", None) or chunk.id or ""
                    tool_msg_buffers[tid] = tool_msg_buffers.get(tid, "") + (content or "")
                elif content and node == "agent" and not isinstance(chunk, ToolMessage):
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

        except Exception as exc:
            log.error(
                "stream error",
                extra={"event": "stream_error", "error": str(exc), "thread_id": thread_id},
            )
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            return

        collected: list[ToolMessage] = [
            ToolMessage(content=text, tool_call_id=tid) for tid, text in tool_msg_buffers.items()
        ]
        citations = _parse_citations(collected)
        if citations:
            yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"

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
    """Hybrid document + law consultant agent.

    Requires ``doc_id`` — searches are scoped strictly to that document.
    Also searches the Ethiopian legal knowledge base for applicable law.
    """
    try:
        graph = get_doc_graph(request.doc_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    return await _stream_endpoint(
        ChatRequest(
            message=request.message, thread_id=request.thread_id, language=request.language
        ),
        event_type="doc_agent_request",
        system_prompt=_apply_language(DOC_CONSULTANT_SYSTEM, request.language),
        graph=graph,
        status_message="Searching documents and legal knowledge base…",
    )
