"""
Shared fixtures for all test modules.

All external I/O (LLM, Qdrant, Cohere) is mocked here so tests run
offline and without any API keys.  The graph is patched to return
pre-canned streaming chunks, giving us full end-to-end SSE coverage
without hitting real infrastructure.
"""

# ---------------------------------------------------------------------------
# Minimal env so Settings() doesn't fail validation on import
# ---------------------------------------------------------------------------
import os
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("COHERE_API_KEY", "test-cohere-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-qdrant-key")


# ---------------------------------------------------------------------------
# Helpers to build fake SSE chunk streams
# ---------------------------------------------------------------------------


def _make_ai_token_chunk(content: str, node: str = "agent") -> tuple:
    msg = AIMessage(content=content)
    meta = {"langgraph_node": node}
    return (msg, meta)


def _make_ai_tool_call_chunk(tool_call_id: str = "tc1", node: str = "agent") -> tuple:
    msg = AIMessage(
        content="",
        tool_calls=[
            {"id": tool_call_id, "name": "search_legal_knowledge", "args": {"query": "contract"}}
        ],
    )
    meta = {"langgraph_node": node}
    return (msg, meta)


def _make_tool_message_chunk(content: str, tool_call_id: str = "tc1", node: str = "tools") -> tuple:
    msg = ToolMessage(content=content, tool_call_id=tool_call_id)
    meta = {"langgraph_node": node}
    return (msg, meta)


CITATION_BLOCK = (
    "[Source: english-civil-code-1960 | Article 1675 | Art. 1675. Contract defined.]\n"
    "A contract is an agreement whereby two or more persons create, vary or extinguish obligations."
)


async def _token_stream(chunks):
    """Async generator that yields (chunk, meta) tuples."""
    for chunk in chunks:
        yield chunk


# ---------------------------------------------------------------------------
# Graph mock factory — returns a mock whose astream yields the given chunks
# ---------------------------------------------------------------------------


def make_mock_graph(chunks: list, has_state: bool = False):
    """
    Build a mock graph object.
    - astream: yields the provided chunks
    - get_state: returns a mock with messages=[...] if has_state else empty
    """
    graph = MagicMock()

    async def _astream(*args, **kwargs):
        for c in chunks:
            yield c

    graph.astream = _astream

    state_mock = MagicMock()
    if has_state:
        state_mock.values = {"messages": [HumanMessage(content="previous")]}
    else:
        state_mock.values = {"messages": []}
    graph.get_state = MagicMock(return_value=state_mock)
    return graph


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def token_chunks():
    """A simple stream: two token chunks, no tool calls."""
    return [
        _make_ai_token_chunk("Hello "),
        _make_ai_token_chunk("world."),
    ]


@pytest.fixture
def retrieval_chunks():
    """A stream that includes a tool call, a tool message (with citation), then a final token."""
    return [
        _make_ai_tool_call_chunk("tc1"),
        _make_tool_message_chunk(CITATION_BLOCK, "tc1"),
        _make_ai_token_chunk("Based on Article 1675, a contract is..."),
    ]


@pytest.fixture
def empty_chunks():
    """A stream that produces no token content."""
    return []


@pytest.fixture
def error_chunks():
    """A stream that raises mid-way."""

    async def _bad_stream(*args, **kwargs):
        yield _make_ai_token_chunk("Starting...")
        raise RuntimeError("upstream LLM failure")

    graph = MagicMock()
    graph.astream = _bad_stream
    state_mock = MagicMock()
    state_mock.values = {"messages": []}
    graph.get_state = MagicMock(return_value=state_mock)
    return graph


@pytest_asyncio.fixture
async def client():
    """Async HTTP test client for the FastAPI app.
    The graph is NOT patched here — individual tests patch it themselves."""
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
