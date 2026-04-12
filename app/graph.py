import json
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    message_chunk_to_message,
    trim_messages,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config import settings
from app.logging_config import get_logger
from app.retrieval import get_doc_blocks_retriever_tool, get_retriever_tool

log = get_logger("graph")

# Maximum number of messages to keep in context (system + this many)
_MAX_MESSAGES = 20


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Cached singletons — built once at first request, reused for every subsequent call
@lru_cache(maxsize=1)
def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.GEMINI_MODEL,
        temperature=0.1,
        streaming=True,
        # Disable Gemini's internal "thinking" phase — thinking models buffer the
        # entire reasoning trace before emitting any tokens, which breaks SSE streaming.
        # extra_body must be a top-level param (not inside model_kwargs) so langchain-openai
        # passes it directly to the httpx request body.
        extra_body={"reasoning": {"effort": "none"}},
    )


@lru_cache(maxsize=1)
def _tool():
    return get_retriever_tool()


@lru_cache(maxsize=1)
def _llm_with_tools():
    return _llm().bind_tools([_tool()])


def _should_continue(state: dict) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"


# Legal search: retrieve then synthesize, no raw block dumping
LEGAL_AGENT_SYSTEM = """You are a legal information assistant with access to a legal knowledge base covering Ethiopian law.

MANDATORY: For EVERY user question, you MUST call search_legal_knowledge first — no exceptions. Never answer from memory alone.
After searching, synthesize what you retrieve into a clear, readable answer. Always reference the specific articles you found (e.g. "Under Article 1726 of the Civil Code..."). Do not dump raw source blocks. If the knowledge base returns no relevant content, say so clearly."""

# Legal consultant: advisory tone, same knowledge base, all legal topics
LEGAL_ADVISOR_SYSTEM = """You are a legal consultant and advisor specializing in Ethiopian law.

MANDATORY: For EVERY user question, you MUST call search_legal_knowledge first — no exceptions. Never answer from memory alone.
After searching, give clear, consultative answers: explain what the law says, what the implications or options are, and any practical considerations — in plain language. Always reference the specific articles or provisions you retrieved. Do not dump raw source blocks. If nothing relevant is found, say so and suggest the user seek qualified legal counsel."""

# Hybrid document + law consultant: understands user docs AND Ethiopian law
DOC_CONSULTANT_SYSTEM = """You are a hybrid legal consultant. A document has already been loaded and is ready for you to search — never ask the user to provide or upload anything.

You have two tools:
1. search_user_documents — searches the loaded document (contracts, agreements, letters, etc.)
2. search_legal_knowledge — searches the Ethiopian law knowledge base (statutes, proclamations, civil code, etc.)

DEFAULT BEHAVIOR — call search_user_documents FIRST for every user message, unless the question is purely about a legal concept with zero possible connection to any document. When in doubt, search the document first.

This means you must call search_user_documents immediately for ANY of the following — do NOT ask for clarification first:
- Vague or open-ended requests: "summarize", "what is this", "what is this about", "explain this", "tell me about this", "what does it say", "describe this"
- Questions about content: parties, dates, obligations, terms, conditions, penalties, clauses, scope, purpose, rights, warranties
- Comparison or analysis requests: "is this fair?", "what are the risks?", "what should I watch out for?"
- Follow-up turns that reference prior context: "what about that clause?", "and the other party?"

When searching, rephrase vague queries into specific search terms that will match document text (e.g. turn "summarize" into "title parties purpose scope obligations agreement").

After searching the document:
- If the question also involves applicable law, rights, or compliance — also call search_legal_knowledge.
- You may call both tools in sequence.

Give clear, practical answers in plain language. Explain what the document says, what the law requires, any risks or implications, and recommended actions. Reference specific clauses (by block_id) and legal articles where relevant. Do not dump raw blocks. If neither tool returns useful content, say so and advise the user to consult qualified legal counsel."""


def _trim(messages: list[BaseMessage], system: SystemMessage) -> list[BaseMessage]:
    """Keep the system message + the most recent _MAX_MESSAGES non-system messages."""
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    if len(non_system) > _MAX_MESSAGES:
        # Always keep the first human message so the LLM has the original question
        trimmed = trim_messages(
            non_system,
            max_tokens=_MAX_MESSAGES,
            token_counter=len,  # count by message count, not tokens
            strategy="last",
            start_on="human",
            include_system=False,
        )
    else:
        trimmed = non_system
    return [system, *trimmed]


def _chunk_to_text(content: object) -> str:
    """Convert LangChain chunk content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _stream_llm_response(llm_with_tools, messages_to_send: list[BaseMessage]) -> AIMessage:
    """Stream token chunks while still returning one final AIMessage for graph state."""
    writer = None
    try:
        writer = get_stream_writer()
    except RuntimeError:
        # Called outside a LangGraph runtime context (e.g. unit tests invoking
        # the node directly). In that case, skip custom stream events.
        writer = None
    full_chunk: AIMessageChunk | None = None
    saw_stream_chunk = False

    for chunk in llm_with_tools.stream(messages_to_send):
        if not isinstance(chunk, AIMessageChunk):
            continue
        saw_stream_chunk = True
        full_chunk = chunk if full_chunk is None else full_chunk + chunk
        text = _chunk_to_text(chunk.content)
        if text and writer is not None:
            writer({"type": "token", "content": text})

    if not saw_stream_chunk:
        return llm_with_tools.invoke(messages_to_send)

    if full_chunk is None:
        return llm_with_tools.invoke(messages_to_send)

    return message_chunk_to_message(full_chunk)


def _agent_node(state: dict) -> dict:
    messages = state["messages"]
    log.info(
        "agent step",
        extra={"event": "agent_step", "message_count": len(messages), "phase": "invoke"},
    )

    # Extract existing system message or fall back to the search default.
    # This preserves whatever persona was injected at thread creation time.
    if messages and isinstance(messages[0], SystemMessage):
        system = messages[0]
    else:
        system = SystemMessage(content=LEGAL_AGENT_SYSTEM)

    messages_to_send = _trim(messages, system)
    response = _stream_llm_response(_llm_with_tools(), messages_to_send)

    if getattr(response, "tool_calls", None):

        def _tc_repr(tc):
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            raw_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    raw_args = {"_raw": raw_args[:200]}
            return {"name": name, "args": raw_args}

        log.info(
            "tool_calls",
            extra={
                "event": "tool_calls",
                "phase": "retrieve",
                "tool_calls": [_tc_repr(tc) for tc in response.tool_calls],
            },
        )
    else:
        log.info("agent responding", extra={"event": "agent_responding", "phase": "respond"})

    return {"messages": [response]}


def build_graph():
    tool_node = ToolNode([_tool()])

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", _agent_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", _should_continue, {"tools": "tools", "__end__": END})
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=MemorySaver())


_graph: object | None = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Doc-consultant agent — uses both search_legal_knowledge + search_user_documents
# ---------------------------------------------------------------------------
# NOTE: The doc graph cannot be a module-level singleton because it embeds a
# Qdrant filter for a specific doc_id that varies per user/request.
# build_doc_graph(doc_id) is called once per new conversation thread; the
# compiled graph is then stored in the thread checkpointer so follow-up turns
# reuse the same scoped graph object without rebuilding.
# ---------------------------------------------------------------------------


def build_doc_graph(doc_id: str):
    """Build a doc-consultant graph scoped to a single document.

    Each call creates a fresh graph with a Qdrant filter locked to *doc_id*,
    so the search_user_documents tool never touches another user's data.
    The LLM and legal-knowledge tool are still cached singletons.
    """
    doc_tool = get_doc_blocks_retriever_tool(doc_id)
    llm_with_doc_tools = _llm().bind_tools([_tool(), doc_tool])

    def _node(state: dict) -> dict:
        messages = state["messages"]
        log.info(
            "doc agent step",
            extra={"event": "doc_agent_step", "message_count": len(messages), "phase": "invoke"},
        )
        system = (
            messages[0]
            if messages and isinstance(messages[0], SystemMessage)
            else SystemMessage(content=DOC_CONSULTANT_SYSTEM)
        )
        messages_to_send = _trim(messages, system)
        response = _stream_llm_response(llm_with_doc_tools, messages_to_send)

        if getattr(response, "tool_calls", None):

            def _tc_repr(tc):
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                raw_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        raw_args = {"_raw": raw_args[:200]}
                return {"name": name, "args": raw_args}

            log.info(
                "doc_tool_calls",
                extra={
                    "event": "doc_tool_calls",
                    "phase": "retrieve",
                    "doc_id": doc_id,
                    "tool_calls": [_tc_repr(tc) for tc in response.tool_calls],
                },
            )
        else:
            log.info(
                "doc agent responding",
                extra={"event": "doc_agent_responding", "phase": "respond", "doc_id": doc_id},
            )

        return {"messages": [response]}

    tool_node = ToolNode([_tool(), doc_tool])

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", _node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", _should_continue, {"tools": "tools", "__end__": END})
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=MemorySaver())


# Cache of compiled doc graphs keyed by doc_id so repeat requests for the
# same document reuse the same graph (and its MemorySaver checkpointer).
_doc_graphs: dict[str, object] = {}


def get_doc_graph(doc_id: str):
    """Return a cached compiled doc-graph for *doc_id*, building it if needed."""
    if doc_id not in _doc_graphs:
        _doc_graphs[doc_id] = build_doc_graph(doc_id)
    return _doc_graphs[doc_id]
