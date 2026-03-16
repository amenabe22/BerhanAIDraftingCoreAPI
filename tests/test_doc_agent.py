"""
Tests for the /doc-agent/stream endpoint, doc_blocks retriever tool (with doc_id filter),
DOC_CONSULTANT_SYSTEM prompt, and doc-graph helpers.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# ---------------------------------------------------------------------------
# _FlatPayloadQdrantVectorStore — new default fields (block_id, type, doc_id)
# ---------------------------------------------------------------------------


def _make_point(payload, point_id="xyz-999"):
    p = MagicMock()
    p.payload = payload
    p.id = point_id
    return p


def test_document_from_point_defaults_doc_blocks_fields():
    """block_id, type, doc_id must default to '' when absent."""
    from app.retrieval import _FlatPayloadQdrantVectorStore

    point = _make_point({"text": "some paragraph text"})
    doc = _FlatPayloadQdrantVectorStore._document_from_point(
        point, "doc_blocks", "text", "metadata"
    )
    assert doc.metadata["block_id"] == ""
    assert doc.metadata["type"] == ""
    assert doc.metadata["doc_id"] == ""


def test_document_from_point_preserves_doc_blocks_payload():
    from app.retrieval import _FlatPayloadQdrantVectorStore

    point = _make_point(
        {
            "text": "4.3 Tax and Pension Obligations…",
            "block_id": "b31",
            "type": "paragraph",
            "doc_id": "8749c6dc-4bb3-4f5c-b593-ae54d0da5437",
        },
        point_id="b31-id",
    )
    doc = _FlatPayloadQdrantVectorStore._document_from_point(
        point, "doc_blocks", "text", "metadata"
    )
    assert doc.page_content == "4.3 Tax and Pension Obligations…"
    assert doc.metadata["block_id"] == "b31"
    assert doc.metadata["type"] == "paragraph"
    assert doc.metadata["doc_id"] == "8749c6dc-4bb3-4f5c-b593-ae54d0da5437"
    assert "text" not in doc.metadata


# ---------------------------------------------------------------------------
# get_doc_blocks_retriever_tool — now requires doc_id
# ---------------------------------------------------------------------------


class _StubRetriever(BaseRetriever):
    docs: list = []

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return self.docs

    def invoke(self, query: str, **kwargs) -> list[Document]:  # type: ignore[override]
        return self.docs


def test_get_doc_blocks_retriever_tool_name():
    from app.retrieval import get_doc_blocks_retriever_tool

    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = _StubRetriever()

    with (
        patch("app.retrieval._get_qdrant_client", return_value=MagicMock()),
        patch("app.retrieval._get_embeddings", return_value=MagicMock()),
        patch("app.retrieval._FlatPayloadQdrantVectorStore", return_value=mock_vs),
    ):
        tool = get_doc_blocks_retriever_tool("8749c6dc-4bb3-4f5c-b593-ae54d0da5437")

    assert tool.name == "search_user_documents"


def test_get_doc_blocks_retriever_tool_uses_doc_blocks_collection():
    from app.retrieval import get_doc_blocks_retriever_tool

    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = _StubRetriever()

    with (
        patch("app.retrieval._get_qdrant_client", return_value=MagicMock()),
        patch("app.retrieval._get_embeddings", return_value=MagicMock()),
        patch("app.retrieval._FlatPayloadQdrantVectorStore", return_value=mock_vs) as MockVS,
        patch("app.retrieval.settings") as mock_settings,
    ):
        mock_settings.QDRANT_DEFAULT_COLLECTION = "doc_blocks"
        mock_settings.COHERE_API_KEY = "key"
        mock_settings.COHERE_EMBEDDING_MODEL = "embed-multilingual-v3.0"
        get_doc_blocks_retriever_tool("abc-123")
        call_kwargs = MockVS.call_args[1]

    assert call_kwargs["collection_name"] == "doc_blocks"
    assert call_kwargs["content_payload_key"] == "text"


def test_get_doc_blocks_retriever_tool_passes_doc_id_filter():
    """as_retriever must be called with a Qdrant filter on doc_id."""
    from qdrant_client.models import FieldCondition, Filter

    from app.config import settings
    from app.retrieval import get_doc_blocks_retriever_tool

    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = _StubRetriever()
    doc_id = "8749c6dc-4bb3-4f5c-b593-ae54d0da5437"

    with (
        patch("app.retrieval._get_qdrant_client", return_value=MagicMock()),
        patch("app.retrieval._get_embeddings", return_value=MagicMock()),
        patch("app.retrieval._FlatPayloadQdrantVectorStore", return_value=mock_vs),
    ):
        get_doc_blocks_retriever_tool(doc_id)

    call_kwargs = mock_vs.as_retriever.call_args[1]["search_kwargs"]
    assert call_kwargs["k"] == settings.RETRIEVAL_DOC_TOP_K
    qdrant_filter = call_kwargs["filter"]
    assert isinstance(qdrant_filter, Filter)
    condition = qdrant_filter.must[0]
    assert isinstance(condition, FieldCondition)
    assert condition.key == "doc_id"
    assert condition.match.value == doc_id


def test_get_doc_blocks_retriever_tool_different_doc_ids_give_different_filters():
    """Two calls with different doc_ids must produce different filters."""
    from app.retrieval import get_doc_blocks_retriever_tool

    filters = []
    for doc_id in ["doc-aaa", "doc-bbb"]:
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = _StubRetriever()
        with (
            patch("app.retrieval._get_qdrant_client", return_value=MagicMock()),
            patch("app.retrieval._get_embeddings", return_value=MagicMock()),
            patch("app.retrieval._FlatPayloadQdrantVectorStore", return_value=mock_vs),
        ):
            get_doc_blocks_retriever_tool(doc_id)
        filters.append(
            mock_vs.as_retriever.call_args[1]["search_kwargs"]["filter"].must[0].match.value
        )

    assert filters[0] == "doc-aaa"
    assert filters[1] == "doc-bbb"


# ---------------------------------------------------------------------------
# DOC_CONSULTANT_SYSTEM
# ---------------------------------------------------------------------------


def test_doc_consultant_system_mentions_both_tools():
    from app.graph import DOC_CONSULTANT_SYSTEM

    assert "search_user_documents" in DOC_CONSULTANT_SYSTEM
    assert "search_legal_knowledge" in DOC_CONSULTANT_SYSTEM


def test_doc_consultant_system_is_non_empty():
    from app.graph import DOC_CONSULTANT_SYSTEM

    assert len(DOC_CONSULTANT_SYSTEM) > 100


# ---------------------------------------------------------------------------
# build_doc_graph / get_doc_graph
# ---------------------------------------------------------------------------


def _mock_legal_tool():
    t = MagicMock()
    t.name = "search_legal_knowledge"
    return t


def test_build_doc_graph_returns_compiled_graph():
    from langgraph.graph.state import CompiledStateGraph

    from app.graph import build_doc_graph

    mock_graph = MagicMock(spec=CompiledStateGraph)
    mock_graph.nodes = {"agent": MagicMock(), "tools": MagicMock()}
    with (
        patch("app.graph._tool", return_value=_mock_legal_tool()),
        patch("app.graph._llm", return_value=MagicMock()),
        patch("app.graph.get_doc_blocks_retriever_tool", return_value=_mock_legal_tool()),
        patch("app.graph.create_react_agent", return_value=mock_graph),
    ):
        graph = build_doc_graph("test-doc-id")
    assert isinstance(graph, CompiledStateGraph)


def test_build_doc_graph_has_agent_and_tools_nodes():
    from app.graph import build_doc_graph

    mock_graph = MagicMock()
    mock_graph.nodes = {"agent": MagicMock(), "tools": MagicMock()}
    with (
        patch("app.graph._tool", return_value=_mock_legal_tool()),
        patch("app.graph._llm", return_value=MagicMock()),
        patch("app.graph.get_doc_blocks_retriever_tool", return_value=_mock_legal_tool()),
        patch("app.graph.create_react_agent", return_value=mock_graph),
    ):
        graph = build_doc_graph("test-doc-id")
    assert "agent" in set(graph.nodes)
    assert "tools" in set(graph.nodes)


def test_get_doc_graph_returns_same_instance_for_same_doc_id():
    """get_doc_graph called twice with the same doc_id must return the same graph."""
    import app.graph as graph_module
    from app.graph import get_doc_graph

    doc_id = "stable-doc-id-for-test"
    original = graph_module._doc_graphs.copy()
    try:
        graph_module._doc_graphs.pop(doc_id, None)
        mock_graph = MagicMock()
        with (
            patch("app.graph._tool", return_value=_mock_legal_tool()),
            patch("app.graph._llm", return_value=MagicMock()),
            patch("app.graph.get_doc_blocks_retriever_tool", return_value=_mock_legal_tool()),
            patch("app.graph.create_react_agent", return_value=mock_graph),
        ):
            g1 = get_doc_graph(doc_id)
            g2 = get_doc_graph(doc_id)
        assert g1 is g2
    finally:
        graph_module._doc_graphs.clear()
        graph_module._doc_graphs.update(original)


def test_get_doc_graph_builds_separate_graphs_for_different_doc_ids():
    import app.graph as graph_module
    from app.graph import get_doc_graph

    original = graph_module._doc_graphs.copy()
    try:
        graph_module._doc_graphs.clear()
        with (
            patch("app.graph._tool", return_value=_mock_legal_tool()),
            patch("app.graph._llm", return_value=MagicMock()),
            patch("app.graph.get_doc_blocks_retriever_tool", return_value=_mock_legal_tool()),
            patch("app.graph.create_react_agent", side_effect=lambda **_kw: MagicMock()),
        ):
            g1 = get_doc_graph("doc-aaa")
            g2 = get_doc_graph("doc-bbb")
        assert g1 is not g2
    finally:
        graph_module._doc_graphs.clear()
        graph_module._doc_graphs.update(original)


# ---------------------------------------------------------------------------
# POST /doc-agent/stream endpoint — now requires doc_id in body
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_doc_graph_fixture():
    from tests.conftest import _make_ai_token_chunk, make_mock_graph

    chunks = [
        _make_ai_token_chunk("The "),
        _make_ai_token_chunk("contract "),
        _make_ai_token_chunk("requires "),
        _make_ai_token_chunk("withholding."),
    ]
    return make_mock_graph(chunks=chunks)


@pytest.mark.asyncio
async def test_doc_agent_stream_returns_sse(mock_doc_graph_fixture):
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    with patch("app.main.get_doc_graph", return_value=mock_doc_graph_fixture):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/doc-agent/stream",
                json={"message": "What are my tax obligations?", "doc_id": "8749c6dc"},
            )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_doc_agent_stream_422_without_doc_id():
    """doc_id is required — omitting it must return 422."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/doc-agent/stream",
            json={"message": "What are my obligations?"},
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_doc_agent_stream_emits_thread_id(mock_doc_graph_fixture):
    import json as _json

    from httpx import ASGITransport, AsyncClient

    from app.main import app

    with patch("app.main.get_doc_graph", return_value=mock_doc_graph_fixture):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/doc-agent/stream",
                json={"message": "Explain clause 4", "doc_id": "8749c6dc"},
            )
    events = [_json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
    thread_events = [e for e in events if e.get("type") == "thread_id"]
    assert len(thread_events) == 1
    assert thread_events[0]["thread_id"]


@pytest.mark.asyncio
async def test_doc_agent_stream_passes_doc_id_to_get_doc_graph(mock_doc_graph_fixture):
    """The endpoint must call get_doc_graph with exactly the supplied doc_id."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    doc_id = "8749c6dc-4bb3-4f5c-b593-ae54d0da5437"
    with patch("app.main.get_doc_graph", return_value=mock_doc_graph_fixture) as mock_gdg:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/doc-agent/stream",
                json={"message": "any question", "doc_id": doc_id},
            )
    mock_gdg.assert_called_once_with(doc_id)


@pytest.mark.asyncio
async def test_doc_agent_stream_503_on_init_error():
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    with patch("app.main.get_doc_graph", side_effect=ValueError("COHERE_API_KEY missing")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/doc-agent/stream",
                json={"message": "anything", "doc_id": "some-doc"},
            )
    assert resp.status_code == 503
    assert "COHERE_API_KEY" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# _parse_citations — [Doc: ...] format
# ---------------------------------------------------------------------------


def test_parse_citations_doc_format():
    from langchain_core.messages import ToolMessage

    from app.main import _parse_citations

    content = (
        "[Doc: 8749c6dc | Block: b31 | Type: paragraph]\n"
        "The Employer shall withhold applicable taxes."
    )
    tm = ToolMessage(content=content, tool_call_id="tc1")
    citations = _parse_citations([tm])

    assert len(citations) == 1
    c = citations[0]
    assert c["doc_id"] == "8749c6dc"
    assert c["block_id"] == "b31"
    assert c["type"] == "paragraph"
    assert "withhold" in c["content"]


def test_parse_citations_mixed_source_and_doc():
    from langchain_core.messages import ToolMessage

    from app.main import _parse_citations

    content = (
        "[Source: ethiopian-law | Article 1675 | Contract defined.]\n"
        "A contract is an agreement.\n\n"
        "[Doc: 8749c6dc | Block: b31 | Type: paragraph]\n"
        "Employer shall withhold taxes."
    )
    tm = ToolMessage(content=content, tool_call_id="tc1")
    citations = _parse_citations([tm])

    assert len(citations) == 2
    types = {c.get("doc_id") for c in citations if "doc_id" in c}
    sources = {c.get("document_id") for c in citations if "document_id" in c}
    assert "8749c6dc" in types
    assert "ethiopian-law" in sources


def test_parse_citations_doc_deduplication():
    from langchain_core.messages import ToolMessage

    from app.main import _parse_citations

    block = "[Doc: abc | Block: b1 | Type: paragraph]\nSome text."
    tm = ToolMessage(content=f"{block}\n\n{block}", tool_call_id="tc1")
    citations = _parse_citations([tm])
    assert len(citations) == 1
