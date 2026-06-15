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
os.environ.setdefault("COHERE_API_KEY", "")  # force keyword fallback in unit tests
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


def _make_custom_token_event(content: str) -> tuple[str, dict]:
    """Build a custom stream-mode token event from graph nodes."""
    return ("custom", {"type": "token", "content": content})


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

    def _stream_impl(*args, **kwargs):
        stream_mode = kwargs.get("stream_mode")
        multi_mode = isinstance(stream_mode, list | tuple)
        for c in chunks:
            # Explicit mode payload supplied by test case.
            if (
                isinstance(c, tuple)
                and len(c) == 2
                and isinstance(c[0], str)
                and c[0] in {"custom", "messages"}
            ):
                if multi_mode:
                    yield c
                elif c[0] == "messages":
                    yield c[1]
                continue

            # Backward-compatible shorthand: raw (chunk, meta) pairs.
            if multi_mode:
                yield ("messages", c)
            else:
                yield c

    async def _astream(*args, **kwargs):
        for item in _stream_impl(*args, **kwargs):
            yield item

    def _stream(*args, **kwargs):
        yield from _stream_impl(*args, **kwargs)

    graph.astream = _astream
    graph.stream = _stream

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
        stream_mode = kwargs.get("stream_mode")
        if isinstance(stream_mode, list | tuple):
            yield ("messages", _make_ai_token_chunk("Starting..."))
        else:
            yield _make_ai_token_chunk("Starting...")
        raise RuntimeError("upstream LLM failure")

    def _bad_sync_stream(*args, **kwargs):
        stream_mode = kwargs.get("stream_mode")
        if isinstance(stream_mode, list | tuple):
            yield ("messages", _make_ai_token_chunk("Starting..."))
        else:
            yield _make_ai_token_chunk("Starting...")
        raise RuntimeError("upstream LLM failure")

    graph = MagicMock()
    graph.astream = _bad_stream
    graph.stream = _bad_sync_stream
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


# ---------------------------------------------------------------------------
# Semantic edit fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def en_doc():
    from tests.fixtures.tiptap_docs import make_en_doc

    return make_en_doc()


@pytest.fixture
def amharic_doc():
    from tests.fixtures.tiptap_docs import make_amharic_doc

    return make_amharic_doc()


@pytest.fixture
def multi_page_doc():
    from tests.fixtures.tiptap_docs import make_multi_page_doc

    return make_multi_page_doc()


def make_mock_locate_result(block_id: str, action: str = "replace", confidence: float = 0.9):
    return {
        "targets": [{"block_id": block_id, "action": action, "confidence": confidence, "reason": "test"}],
        "scope": "single",
        "confidence": confidence,
    }


def make_mock_edit_ops(block_id: str, new_text: str, op_type: str = "replace"):
    return {
        "operations": [
            {
                "op_id": "op123456",
                "type": op_type,
                "block_id": block_id,
                "payload": {"new_text": new_text},
            }
        ]
    }


def make_mock_verify_result(passed: bool = True, feedback: str = ""):
    return {"passed": passed, "issues": [] if passed else ["test issue"], "feedback": feedback}
