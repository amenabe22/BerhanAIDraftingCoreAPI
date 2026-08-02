import json
import time
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_chunk_to_message,
    trim_messages,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_config, get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.config import settings
from app.llm import build_chat_llm, resolve_model
from app.logging_config import get_logger
from app.retrieval import get_doc_blocks_retriever_tool, get_retriever_tool
from app.services.legal.grounding import (
    GroundingResult,
    repair_message,
    safe_fallback_answer,
    validate_answer,
)
from app.services.legal.instrument_router import route as _route_query

log = get_logger("graph")

# Maximum number of messages to keep in context (system + this many)
_MAX_MESSAGES = 20


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    repair_count: int  # number of grounding-repair attempts made (0 = none yet)
    grounding: dict  # last GroundingResult fields: {ok, reason, repair_attempted}


# Per-(model, enable_reasoning) LLM — delegates to shared factory in app.llm
def _get_llm_for_config() -> ChatOpenAI:
    """Return a ChatOpenAI instance appropriate for the current LangGraph config."""
    try:
        cfg = get_config().get("configurable", {})
    except Exception:
        cfg = {}
    model = resolve_model(cfg.get("model"))
    enable_reasoning = bool(cfg.get("enable_reasoning", False))
    return build_chat_llm(model=model, enable_reasoning=enable_reasoning, streaming=True)


@lru_cache(maxsize=1)
def _tool():
    return get_retriever_tool()


def _llm_with_tools():
    """Return an LLM bound to the legal retriever tool for the current config."""
    return _get_llm_for_config().bind_tools([_tool()])


def _should_continue(state: dict) -> Literal["tools", "ground"]:
    """Route to tools if there are pending tool calls, otherwise to the grounder."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "ground"


def _should_end_or_repair(state: dict) -> Literal["agent", "__end__"]:
    """After ground_node: retry with repair if first failure; end otherwise."""
    repair_count = state.get("repair_count", 0)
    grounding = state.get("grounding", {})
    # repair_count == 1 means ground_node just injected the repair HumanMessage.
    # Send back to agent for one more attempt.
    if repair_count == 1 and not grounding.get("ok", True):
        return "agent"
    return "__end__"


# ---------------------------------------------------------------------------
# System-prompt building blocks
# ---------------------------------------------------------------------------

_IDENTITY_GUARDRAIL = (
    "IDENTITY: You are a Legal AI Model built by BerhanAI. "
    "Only if the user explicitly asks who you are, what model you are, or who built or trained you, "
    'reply once with exactly: "I\'m a Legal AI Model built by BerhanAI." '
    "Do not state your identity, introduce yourself, or repeat that sentence in normal legal answers. "
    "Never mention Google, Gemini, OpenAI, or any other underlying provider or model name."
)

_TONE_GUIDANCE = (
    "TONE: Be professional, neutral, and helpful—like a knowledgeable colleague. "
    "Do not assume illegal intent. Avoid alarmist or moralizing language unless the user clearly asks about serious criminal conduct. "
    "Do not repeat the same point or warning twice in one reply. Prefer short paragraphs; use bullets only for three or more distinct points."
)

_ATTACHMENT_GUIDANCE = (
    "ATTACHMENTS: If the user message includes an image or file URL, first identify the specific product, label, or document content visible. "
    "Base your answer on that item—do not invent different products, substances, or scenarios (e.g. do not mention drugs unless clearly shown or asked). "
    'For import legality questions ("is this legal to import"), search using the identified product category and Ethiopian import, customs, and licensing rules—not generic "illegal import" queries.'
)

_RETRIEVAL_GUIDANCE = (
    "RETRIEVAL — MANDATORY RULES:\n"
    "1. SEARCH FIRST: For ANY question about Ethiopian law — statutes, contracts, "
    "obligations, rights, procedures, employment, tax, inheritance, property, commercial "
    "law, or any other legal topic — call search_legal_knowledge BEFORE stating any "
    "legal conclusion. Never assert law from general knowledge alone.\n"
    "2. CITATION CONTRACT: Every legal sentence must be backed by a pinpoint from the "
    "tool results. Format: 'Under Article N of [Instrument]…' or 'አንቀጽ N…'. Use the "
    "document_id and item_id from the [Source: … | Article …] headers returned by the "
    "tool. ONLY cite instruments that appear in the tool results just received.\n"
    "3. CORRECT INSTRUMENT: If the question concerns a company, director, board, "
    "shareholders, or commercial partnerships → cite from the Commercial Code "
    "(Proclamation No. 1243/2021) as the PRIMARY source. Do NOT answer company/director "
    "questions using only the Civil Code associations chapter even if Civil Code chunks "
    "appear in results.\n"
    "4. KB GAP: If search_legal_knowledge returns no relevant provision, state this "
    "explicitly: 'The knowledge base does not contain a specific provision on this point.' "
    "Do NOT invent article numbers, freeze durations, case holdings, or procedural "
    "timelines. Do NOT write 'ከላይ የተጠቀሱት አንቀጾች' unless those exact article numbers "
    "appeared in the tool text just received.\n"
    "5. FOLLOW-UPS: On every turn that adds or extends a legal claim — including "
    "multi-turn conversations — call search_legal_knowledge again. A procedural steps "
    "list or advisory playbook with no article citations is not acceptable."
)


# Legal search: retrieve then synthesize, no raw block dumping
LEGAL_AGENT_SYSTEM = f"""{_IDENTITY_GUARDRAIL}

{_TONE_GUIDANCE}

{_ATTACHMENT_GUIDANCE}

You are a legal information assistant with access to a legal knowledge base covering Ethiopian law.

{_RETRIEVAL_GUIDANCE}
After searching, synthesize what you retrieve into a clear, readable answer. Reference specific articles you found (e.g. "Under Article 1726 of the Civil Code…" or "አዋጅ ቁጥር 1156/2019 አንቀጽ 43 መሰረት…"). Do not dump raw source blocks. If the knowledge base returns no relevant content, say so clearly — do not invent provisions."""

# Legal consultant: advisory tone, same knowledge base, all legal topics
LEGAL_ADVISOR_SYSTEM = f"""{_IDENTITY_GUARDRAIL}

{_TONE_GUIDANCE}

{_ATTACHMENT_GUIDANCE}

You are a legal consultant and advisor specializing in Ethiopian law.

{_RETRIEVAL_GUIDANCE}
After searching, give clear, consultative answers: explain what the law says, implications or options, and practical considerations — in plain language. Reference specific articles or provisions you retrieved (e.g. "Under Article 1726 of the Civil Code…" or "አዋጅ ቁጥር 1156/2019 አንቀጽ 43 …"). Do not dump raw source blocks. If nothing relevant is found, say so and note that qualified legal counsel may be needed for the specific transaction — do not invent provisions."""

# Hybrid document + law consultant: understands user docs AND Ethiopian law
DOC_CONSULTANT_SYSTEM = f"""{_IDENTITY_GUARDRAIL}

{_TONE_GUIDANCE}

You are a hybrid legal consultant. A document has already been loaded and is ready for you to search — never ask the user to provide or upload anything.

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

{_RETRIEVAL_GUIDANCE}
Give clear, practical answers in plain language. Explain what the document says, what the law requires, any risks or implications, and recommended actions. Reference specific clauses (by block_id) and legal articles where relevant (with pinpoint article numbers from tool results). Do not dump raw blocks. If neither tool returns useful content, say so and advise the user to consult qualified legal counsel."""


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


_LLM_STREAM_MAX_ATTEMPTS = 3
# OpenRouter occasionally injects mid-stream error SSE when an upstream provider 500s.
_TRANSIENT_LLM_STREAM_MARKERS = (
    "json error injected into sse stream",
    "unexpected end of json",
    "jsondecodeerror",
    "connection reset by peer",
    "provider disconnected",
    "overloaded",
)


def _is_transient_llm_stream_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    if "timeout" in msg or "timeout" in name:
        return True
    return any(marker in msg for marker in _TRANSIENT_LLM_STREAM_MARKERS)


def _invoke_with_token_writer(
    llm_with_tools,
    messages_to_send: list[BaseMessage],
    writer,
) -> AIMessage:
    """Non-streaming fallback that still emits one token event for SSE clients."""
    response = llm_with_tools.invoke(messages_to_send)
    text = _chunk_to_text(getattr(response, "content", ""))
    if text and writer is not None:
        writer({"type": "token", "content": text})
    return response


def _stream_llm_response_once(
    llm_with_tools,
    messages_to_send: list[BaseMessage],
    writer,
) -> AIMessage:
    full_chunk: AIMessageChunk | None = None
    saw_stream_chunk = False
    emitted_thinking_status = False

    for chunk in llm_with_tools.stream(messages_to_send):
        if not isinstance(chunk, AIMessageChunk):
            continue
        saw_stream_chunk = True
        full_chunk = chunk if full_chunk is None else full_chunk + chunk

        # Forward reasoning/thinking tokens emitted before the answer starts.
        # OpenRouter sends these in additional_kwargs["reasoning"] while content
        # is still empty. We emit a one-time status so the client shows activity,
        # then stream each thinking chunk so the frontend can display it.
        reasoning_text = (chunk.additional_kwargs or {}).get("reasoning") or ""
        if reasoning_text and writer is not None:
            if not emitted_thinking_status:
                emitted_thinking_status = True
                writer({"type": "status", "message": "Reasoning…"})
            writer({"type": "thinking", "content": reasoning_text})

        text = _chunk_to_text(chunk.content)
        if text and writer is not None:
            writer({"type": "token", "content": text})

    if not saw_stream_chunk or full_chunk is None:
        return _invoke_with_token_writer(llm_with_tools, messages_to_send, writer)

    return message_chunk_to_message(full_chunk)


def _stream_llm_response(llm_with_tools, messages_to_send: list[BaseMessage]) -> AIMessage:
    """Stream token chunks while still returning one final AIMessage for graph state."""
    writer = None
    try:
        writer = get_stream_writer()
    except RuntimeError:
        # Called outside a LangGraph runtime context (e.g. unit tests invoking
        # the node directly). In that case, skip custom stream events.
        writer = None

    for attempt in range(1, _LLM_STREAM_MAX_ATTEMPTS + 1):
        try:
            return _stream_llm_response_once(llm_with_tools, messages_to_send, writer)
        except Exception as exc:
            if attempt < _LLM_STREAM_MAX_ATTEMPTS and _is_transient_llm_stream_error(exc):
                log.warning(
                    "LLM stream attempt failed (transient), retrying",
                    extra={
                        "event": "llm_stream_retry",
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )
                time.sleep(0.4 * attempt)
                continue
            log.warning(
                "LLM stream failed, falling back to invoke",
                extra={
                    "event": "llm_stream_fallback",
                    "attempt": attempt,
                    "error": str(exc),
                },
            )
            return _invoke_with_token_writer(llm_with_tools, messages_to_send, writer)


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


# ---------------------------------------------------------------------------
# Grounding node — validates the final answer and triggers one repair if needed
# ---------------------------------------------------------------------------


def _ground_node(state: dict) -> dict:
    """Verify the last AIMessage is adequately cited; trigger repair or fallback."""
    messages = state["messages"]
    repair_count = state.get("repair_count", 0)

    # Locate the last AI response
    last_ai: AIMessage | None = None
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            last_ai = m
            break

    if last_ai is None:
        # Nothing to verify; let it through
        return {"grounding": {"ok": True, "reason": None, "repair_attempted": False}}

    answer = _chunk_to_text(last_ai.content)

    # Collect all ToolMessages in this conversation for the grounding check
    tool_messages: list[ToolMessage] = [m for m in messages if isinstance(m, ToolMessage)]

    # Determine expect_kb_gap by routing the most recent user query
    last_human_text = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            content = m.content
            text = content if isinstance(content, str) else _chunk_to_text(content)
            # Skip the injected repair messages (they start with "Your previous answer…")
            if not text.startswith("Your previous answer") and not text.startswith(
                "የቀደመው መልስ"
            ):
                last_human_text = text
                break

    route_decision = _route_query(last_human_text)
    expect_kb_gap = route_decision.expect_kb_gap

    result: GroundingResult = validate_answer(
        answer, tool_messages, expect_kb_gap=expect_kb_gap
    )

    # ── emit grounding custom event for SSE ──
    writer = None
    try:
        writer = get_stream_writer()
    except RuntimeError:
        pass

    if result.ok:
        grounding_dict = {
            "ok": True,
            "reason": None,
            "repair_attempted": repair_count > 0,
            "route_decision": {
                "query_suffix": route_decision.query_suffix,
                "expect_kb_gap": expect_kb_gap,
                "forbidden_primary": route_decision.forbidden_primary,
            },
        }
        if writer is not None:
            writer(
                {
                    "type": "grounding",
                    "ok": True,
                    "repair_attempted": grounding_dict["repair_attempted"],
                    "reason": None,
                }
            )
        log.info(
            "grounding_ok",
            extra={
                "event": "grounding",
                "ok": True,
                "repair_count": repair_count,
                "route_suffix": route_decision.query_suffix,
            },
        )
        return {"grounding": grounding_dict}

    # ── Grounding failed ──
    log.warning(
        "grounding_failed",
        extra={
            "event": "grounding",
            "ok": False,
            "reason": result.reason,
            "repair_count": repair_count,
            "route_suffix": route_decision.query_suffix,
        },
    )

    if repair_count == 0:
        # First failure → inject repair HumanMessage, go back to agent
        if writer is not None:
            writer({"type": "status", "message": "Verifying citations, searching again…"})

        repair_msg = repair_message(answer)
        return {
            "messages": [HumanMessage(content=repair_msg)],
            "repair_count": 1,
            "grounding": {
                "ok": False,
                "reason": result.reason,
                "repair_attempted": False,
                "route_decision": {
                    "query_suffix": route_decision.query_suffix,
                    "expect_kb_gap": expect_kb_gap,
                    "forbidden_primary": route_decision.forbidden_primary,
                },
            },
        }
    else:
        # Second failure → replace last AI message with safe fallback
        fallback = safe_fallback_answer(answer)
        if writer is not None:
            writer(
                {
                    "type": "grounding",
                    "ok": False,
                    "repair_attempted": True,
                    "reason": "uncited_or_ungrounded",
                }
            )
        log.warning(
            "grounding_failed_after_repair",
            extra={
                "event": "grounding",
                "ok": False,
                "reason": "uncited_or_ungrounded_after_repair",
                "repair_count": repair_count,
            },
        )
        return {
            "messages": [AIMessage(content=fallback)],
            "repair_count": repair_count + 1,
            "grounding": {
                "ok": False,
                "reason": "uncited_or_ungrounded",
                "repair_attempted": True,
            },
        }


def build_graph():
    tool_node = ToolNode([_tool()])

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", _agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("ground", _ground_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", "ground": "ground"},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_conditional_edges(
        "ground",
        _should_end_or_repair,
        {"agent": "agent", "__end__": END},
    )

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
    Includes the grounding verifier so doc-agent answers are also citation-hardened.
    """
    doc_tool = get_doc_blocks_retriever_tool(doc_id)
    # Doc graph always uses the env-configured default model with reasoning off.
    _default_llm = build_chat_llm(
        model=resolve_model(None), enable_reasoning=False, streaming=True
    )
    llm_with_doc_tools = _default_llm.bind_tools([_tool(), doc_tool])

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
    workflow.add_node("ground", _ground_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", "ground": "ground"},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_conditional_edges(
        "ground",
        _should_end_or_repair,
        {"agent": "agent", "__end__": END},
    )

    return workflow.compile(checkpointer=MemorySaver())


# Cache of compiled doc graphs keyed by doc_id so repeat requests for the
# same document reuse the same graph (and its MemorySaver checkpointer).
_doc_graphs: dict[str, object] = {}


def get_doc_graph(doc_id: str):
    """Return a cached compiled doc-graph for *doc_id*, building it if needed."""
    if doc_id not in _doc_graphs:
        _doc_graphs[doc_id] = build_doc_graph(doc_id)
    return _doc_graphs[doc_id]
