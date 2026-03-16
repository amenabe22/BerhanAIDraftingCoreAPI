from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.config import settings
from app.logging_config import get_logger
from app.retrieval import get_doc_blocks_retriever_tool, get_retriever_tool

log = get_logger("graph")

# Maximum number of messages to keep in context (system + this many)
_MAX_MESSAGES = 20


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


def build_graph(system_prompt: str):
    """Build a react agent graph that streams tokens directly from the LLM.

    create_react_agent uses astream internally, so graph.astream(..., stream_mode="messages")
    yields AIMessageChunk objects for each LLM token rather than one complete AIMessage.
    """
    return create_react_agent(
        model=_llm(),
        tools=[_tool()],
        prompt=SystemMessage(content=system_prompt),
        checkpointer=MemorySaver(),
    )


_graphs: dict[str, object] = {}


def get_graph(system_prompt: str = LEGAL_AGENT_SYSTEM):
    if system_prompt not in _graphs:
        _graphs[system_prompt] = build_graph(system_prompt)
    return _graphs[system_prompt]


# ---------------------------------------------------------------------------
# Doc-consultant agent — uses both search_legal_knowledge + search_user_documents
# ---------------------------------------------------------------------------
# NOTE: The doc graph cannot be a module-level singleton because it embeds a
# Qdrant filter for a specific doc_id that varies per user/request.
# build_doc_graph(doc_id) is called once per new conversation thread; the
# compiled graph is then stored in the thread checkpointer so follow-up turns
# reuse the same scoped graph object without rebuilding.
# ---------------------------------------------------------------------------


def build_doc_graph(doc_id: str, system_prompt: str = DOC_CONSULTANT_SYSTEM):
    """Build a doc-consultant graph scoped to a single document.

    Each call creates a fresh graph with a Qdrant filter locked to *doc_id*,
    so the search_user_documents tool never touches another user's data.
    The LLM and legal-knowledge tool are still cached singletons.
    """
    doc_tool = get_doc_blocks_retriever_tool(doc_id)
    return create_react_agent(
        model=_llm(),
        tools=[_tool(), doc_tool],
        prompt=SystemMessage(content=system_prompt),
        checkpointer=MemorySaver(),
    )


# Cache of compiled doc graphs keyed by doc_id so repeat requests for the
# same document reuse the same graph (and its MemorySaver checkpointer).
_doc_graphs: dict[str, object] = {}


def get_doc_graph(doc_id: str):
    """Return a cached compiled doc-graph for *doc_id*, building it if needed."""
    if doc_id not in _doc_graphs:
        _doc_graphs[doc_id] = build_doc_graph(doc_id)
    return _doc_graphs[doc_id]
