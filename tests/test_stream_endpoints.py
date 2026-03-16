"""
Integration-style tests for the streaming SSE endpoints:
  POST /legal-search/stream
  POST /legal-agent/stream

The real graph, LLM, Qdrant, and Cohere are ALL mocked.
We inject a fake graph whose astream() yields pre-canned chunks,
then parse the SSE events and assert on their structure/content.
"""

import contextlib
import json
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import HumanMessage

from tests.conftest import (
    CITATION_BLOCK,
    _make_ai_token_chunk,
    _make_ai_tool_call_chunk,
    _make_tool_message_chunk,
    make_mock_graph,
)

ENDPOINTS = ["/legal-search/stream", "/legal-agent/stream"]


# ---------------------------------------------------------------------------
# SSE parsing helper
# ---------------------------------------------------------------------------


def parse_sse(text: str) -> list[dict]:
    """Parse raw SSE text into a list of parsed JSON event dicts."""
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            with contextlib.suppress(json.JSONDecodeError):
                events.append(json.loads(line[6:]))
    return events


# ---------------------------------------------------------------------------
# Fixture: patched app client with a given mock graph
# ---------------------------------------------------------------------------


async def _client_with_graph(mock_graph):
    """Return an AsyncClient where get_graph() always returns mock_graph."""
    from app.main import app

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# 1. thread_id event is always the first event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_thread_id_event_is_first(endpoint):
    graph = make_mock_graph([_make_ai_token_chunk("Hello")])
    with patch("app.main.get_graph", return_value=graph):
        async with AsyncClient(
            transport=ASGITransport(app=__import__("app.main", fromlist=["app"]).app),
            base_url="http://test",
        ) as client:
            r = await client.post(endpoint, json={"message": "test"})
    assert r.status_code == 200
    events = parse_sse(r.text)
    assert events[0]["type"] == "thread_id"
    assert isinstance(events[0]["thread_id"], str)
    assert len(events[0]["thread_id"]) > 0


# ---------------------------------------------------------------------------
# 2. provided thread_id is echoed back unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_provided_thread_id_echoed(endpoint):
    graph = make_mock_graph([_make_ai_token_chunk("hi")])
    tid = "my-fixed-thread-123"
    with patch("app.main.get_graph", return_value=graph):
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(endpoint, json={"message": "hi", "thread_id": tid})
    events = parse_sse(r.text)
    assert events[0]["thread_id"] == tid


# ---------------------------------------------------------------------------
# 3. Token events stream correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_token_events_present(endpoint):
    chunks = [_make_ai_token_chunk("Hello "), _make_ai_token_chunk("world.")]
    graph = make_mock_graph(chunks)
    with patch("app.main.get_graph", return_value=graph):
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(endpoint, json={"message": "hello"})
    token_events = [e for e in parse_sse(r.text) if e["type"] == "token"]
    assert len(token_events) == 2
    assert token_events[0]["content"] == "Hello "
    assert token_events[1]["content"] == "world."


# ---------------------------------------------------------------------------
# 4. Status event emitted when tool call chunk arrives
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_status_event_on_tool_call(endpoint):
    chunks = [
        _make_ai_tool_call_chunk("tc1"),
        _make_tool_message_chunk(CITATION_BLOCK, "tc1"),
        _make_ai_token_chunk("Based on the law..."),
    ]
    graph = make_mock_graph(chunks)
    with patch("app.main.get_graph", return_value=graph):
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(endpoint, json={"message": "what is a contract"})
    events = parse_sse(r.text)
    status_events = [e for e in events if e["type"] == "status"]
    assert len(status_events) == 1
    assert "search" in status_events[0]["message"].lower()


# ---------------------------------------------------------------------------
# 5. Citations event emitted at the end when tool messages were received
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_citations_event_at_end(endpoint):
    chunks = [
        _make_ai_tool_call_chunk("tc1"),
        _make_tool_message_chunk(CITATION_BLOCK, "tc1"),
        _make_ai_token_chunk("Based on the law..."),
    ]
    graph = make_mock_graph(chunks)
    with patch("app.main.get_graph", return_value=graph):
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(endpoint, json={"message": "contract"})
    events = parse_sse(r.text)
    citation_events = [e for e in events if e["type"] == "citations"]
    assert len(citation_events) == 1
    cits = citation_events[0]["citations"]
    assert len(cits) >= 1
    assert cits[0]["document_id"] == "english-civil-code-1960"
    assert cits[0]["item_id"] == "1675"


# ---------------------------------------------------------------------------
# 6. No citations event when no tool messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_no_citations_event_without_retrieval(endpoint):
    graph = make_mock_graph([_make_ai_token_chunk("Direct answer.")])
    with patch("app.main.get_graph", return_value=graph):
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(endpoint, json={"message": "hello"})
    citation_events = [e for e in parse_sse(r.text) if e["type"] == "citations"]
    assert citation_events == []


# ---------------------------------------------------------------------------
# 7. Error event emitted when stream raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_error_event_on_stream_failure(endpoint, error_chunks):
    with patch("app.main.get_graph", return_value=error_chunks):
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(endpoint, json={"message": "hi"})
    assert r.status_code == 200  # SSE always 200; error is in the stream
    error_events = [e for e in parse_sse(r.text) if e["type"] == "error"]
    assert len(error_events) == 1
    assert "upstream LLM failure" in error_events[0]["message"]


# ---------------------------------------------------------------------------
# 8. 503 returned when graph initialisation fails (missing API key)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_503_when_graph_init_fails(endpoint):
    with patch("app.main.get_graph", side_effect=ValueError("COHERE_API_KEY is required")):
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(endpoint, json={"message": "hi"})
    assert r.status_code == 503
    assert "COHERE_API_KEY" in r.json()["detail"]


# ---------------------------------------------------------------------------
# 9. Missing required field returns 422
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_422_when_message_missing(endpoint):
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(endpoint, json={"thread_id": "abc"})  # no message
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# 10. Empty message is accepted (the model decides what to do)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_empty_message_accepted(endpoint):
    graph = make_mock_graph([_make_ai_token_chunk("How can I help?")])
    with patch("app.main.get_graph", return_value=graph):
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(endpoint, json={"message": ""})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# 11. Response headers are correct for SSE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_sse_headers(endpoint):
    graph = make_mock_graph([_make_ai_token_chunk("hi")])
    with patch("app.main.get_graph", return_value=graph):
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(endpoint, json={"message": "hi"})
    assert "text/event-stream" in r.headers["content-type"]
    assert r.headers.get("cache-control") == "no-cache"
    assert "x-thread-id" in r.headers


# ---------------------------------------------------------------------------
# 12. Input always contains exactly one HumanMessage (graph owns system prompt)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_input_contains_only_human_message(endpoint):
    """main.py passes only HumanMessage to the graph; system prompt is owned by the graph."""
    graph = make_mock_graph([_make_ai_token_chunk("Answer.")])
    captured = {}

    original_astream = graph.astream

    async def capturing_astream(inputs, **kwargs):
        captured["messages"] = inputs["messages"]
        async for chunk in original_astream(inputs, **kwargs):
            yield chunk

    graph.astream = capturing_astream

    with patch("app.main.get_graph", return_value=graph):
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(endpoint, json={"message": "test question"})

    assert len(captured["messages"]) == 1
    assert isinstance(captured["messages"][0], HumanMessage)
    assert captured["messages"][0].content == "test question"


# ---------------------------------------------------------------------------
# 13. get_graph is called with the language-prefixed system prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legal_search_calls_get_graph_with_system_prompt():
    from app.graph import LEGAL_AGENT_SYSTEM

    graph = make_mock_graph([_make_ai_token_chunk("ok")])
    captured_prompt = {}

    def fake_get_graph(system_prompt):
        captured_prompt["value"] = system_prompt
        return graph

    with patch("app.main.get_graph", side_effect=fake_get_graph):
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/legal-search/stream", json={"message": "q"})

    assert LEGAL_AGENT_SYSTEM in captured_prompt["value"]


@pytest.mark.asyncio
async def test_legal_agent_calls_get_graph_with_advisor_system_prompt():
    from app.graph import LEGAL_ADVISOR_SYSTEM

    graph = make_mock_graph([_make_ai_token_chunk("ok")])
    captured_prompt = {}

    def fake_get_graph(system_prompt):
        captured_prompt["value"] = system_prompt
        return graph

    with patch("app.main.get_graph", side_effect=fake_get_graph):
        from app.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/legal-agent/stream", json={"message": "q"})

    assert LEGAL_ADVISOR_SYSTEM in captured_prompt["value"]
