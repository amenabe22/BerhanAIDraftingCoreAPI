"""
Unit tests for _FlatPayloadQdrantVectorStore._document_from_point,
_get_embeddings, _get_qdrant_client, _LoggingRetriever, and
get_retriever_tool in app/retrieval.py.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# ---------------------------------------------------------------------------
# _FlatPayloadQdrantVectorStore._document_from_point
# ---------------------------------------------------------------------------


def _make_point(payload: dict, point_id: str = "abc-123"):
    point = MagicMock()
    point.payload = payload
    point.id = point_id
    return point


def test_document_from_point_maps_content_to_page_content():
    from app.retrieval import _FlatPayloadQdrantVectorStore

    point = _make_point(
        {
            "content": "This is the article text.",
            "document_id": "english-civil-code-1960",
            "item_id": "1675",
            "title": "Contract defined.",
        }
    )
    doc = _FlatPayloadQdrantVectorStore._document_from_point(
        point, "test_collection", "content", "metadata"
    )
    assert doc.page_content == "This is the article text."


def test_document_from_point_puts_other_fields_in_metadata():
    from app.retrieval import _FlatPayloadQdrantVectorStore

    point = _make_point(
        {
            "content": "Text",
            "document_id": "doc-x",
            "item_id": "42",
            "title": "Some Title",
            "language": "english",
        }
    )
    doc = _FlatPayloadQdrantVectorStore._document_from_point(point, "col", "content", "metadata")
    assert doc.metadata["document_id"] == "doc-x"
    assert doc.metadata["item_id"] == "42"
    assert doc.metadata["title"] == "Some Title"
    assert doc.metadata["language"] == "english"
    assert "content" not in doc.metadata


def test_document_from_point_injects_id_and_collection():
    from app.retrieval import _FlatPayloadQdrantVectorStore

    point = _make_point({"content": "text"}, point_id="my-id")
    doc = _FlatPayloadQdrantVectorStore._document_from_point(
        point, "my_collection", "content", "metadata"
    )
    assert doc.metadata["_id"] == "my-id"
    assert doc.metadata["_collection_name"] == "my_collection"


def test_document_from_point_defaults_empty_citation_fields():
    """document_id, item_id, title must always exist in metadata (for template rendering)."""
    from app.retrieval import _FlatPayloadQdrantVectorStore

    point = _make_point({"content": "text"})  # no document_id / item_id / title
    doc = _FlatPayloadQdrantVectorStore._document_from_point(point, "col", "content", "metadata")
    assert doc.metadata["document_id"] == ""
    assert doc.metadata["item_id"] == ""
    assert doc.metadata["title"] == ""


def test_document_from_point_empty_payload_gives_empty_content():
    from app.retrieval import _FlatPayloadQdrantVectorStore

    point = _make_point({})
    doc = _FlatPayloadQdrantVectorStore._document_from_point(point, "col", "content", "metadata")
    assert doc.page_content == ""


def test_document_from_point_none_payload_handled():
    from app.retrieval import _FlatPayloadQdrantVectorStore

    point = _make_point(None)
    point.payload = None
    doc = _FlatPayloadQdrantVectorStore._document_from_point(point, "col", "content", "metadata")
    assert doc.page_content == ""


# ---------------------------------------------------------------------------
# _get_embeddings — raises if COHERE_API_KEY is missing
# ---------------------------------------------------------------------------


def test_get_embeddings_raises_without_cohere_key():
    from app.retrieval import _get_embeddings

    with patch("app.retrieval.settings") as mock_settings:
        mock_settings.COHERE_API_KEY = None
        mock_settings.COHERE_EMBEDDING_MODEL = "embed-multilingual-v3.0"
        with pytest.raises(ValueError, match="COHERE_API_KEY"):
            _get_embeddings()


def test_get_embeddings_succeeds_with_key():
    from app.retrieval import _get_embeddings

    with patch("app.retrieval.settings") as mock_settings:
        mock_settings.COHERE_API_KEY = "test-key"
        mock_settings.COHERE_EMBEDDING_MODEL = "embed-multilingual-v3.0"
        # CohereEmbeddings constructor is patched so it doesn't make real calls
        with patch("app.retrieval.CohereEmbeddings") as mock_cohere:
            mock_cohere.return_value = MagicMock()
            _get_embeddings()
            mock_cohere.assert_called_once_with(
                model="embed-multilingual-v3.0",
                cohere_api_key="test-key",
            )


# ---------------------------------------------------------------------------
# _get_qdrant_client
# ---------------------------------------------------------------------------


def test_get_qdrant_client_passes_url_and_api_key():
    from app.retrieval import _get_qdrant_client

    with (
        patch("app.retrieval.settings") as mock_settings,
        patch("app.retrieval.QdrantClient") as MockClient,
    ):
        mock_settings.QDRANT_URL = "https://qdrant.example.com"
        mock_settings.QDRANT_API_KEY = "qkey-123"
        mock_settings.QDRANT_VERIFY_SSL = True
        MockClient.return_value = MagicMock()
        _get_qdrant_client()
        MockClient.assert_called_once_with(
            url="https://qdrant.example.com",
            api_key="qkey-123",
            check_compatibility=False,
            verify=True,
        )


# ---------------------------------------------------------------------------
# _LoggingRetriever
# ---------------------------------------------------------------------------


def _make_doc(content: str, document_id: str = "doc-1", item_id: str = "42") -> Document:
    return Document(
        page_content=content,
        metadata={"document_id": document_id, "item_id": item_id, "title": "Title"},
    )


# ---------------------------------------------------------------------------
# Minimal real BaseRetriever stub (MagicMock fails Pydantic validation)
# ---------------------------------------------------------------------------


class _StubRetriever(BaseRetriever):
    """A concrete BaseRetriever that returns a pre-set list of docs."""

    docs: list = []

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return self.docs

    def invoke(self, query: str, **kwargs) -> list[Document]:  # type: ignore[override]
        return self.docs


def test_logging_retriever_returns_docs_from_inner():
    from app.retrieval import _LoggingRetriever

    docs = [_make_doc("contract text"), _make_doc("family law text")]
    inner = _StubRetriever(docs=docs)

    retriever = _LoggingRetriever(retriever=inner)
    result = retriever._get_relevant_documents("what is a contract?")
    assert result == docs


def test_logging_retriever_passes_query_to_inner():
    from app.retrieval import _LoggingRetriever

    captured = []

    class _CapturingRetriever(_StubRetriever):
        def invoke(self, query: str, **kwargs) -> list[Document]:
            captured.append(query)
            return []

    inner = _CapturingRetriever()
    retriever = _LoggingRetriever(retriever=inner)
    retriever._get_relevant_documents("inheritance rights")
    assert captured == ["inheritance rights"]


def test_logging_retriever_handles_empty_results():
    from app.retrieval import _LoggingRetriever

    inner = _StubRetriever(docs=[])
    retriever = _LoggingRetriever(retriever=inner)
    result = retriever._get_relevant_documents("obscure query")
    assert result == []


def test_logging_retriever_handles_doc_without_metadata():
    from app.retrieval import _LoggingRetriever

    doc = Document(page_content="text", metadata={})
    inner = _StubRetriever(docs=[doc])
    retriever = _LoggingRetriever(retriever=inner)
    # Should not raise even when metadata fields are absent
    result = retriever._get_relevant_documents("query")
    assert len(result) == 1


def test_logging_retriever_truncates_content_preview():
    from app.retrieval import _LoggingRetriever

    long_content = "A" * 300
    inner = _StubRetriever(docs=[_make_doc(long_content)])

    retriever = _LoggingRetriever(retriever=inner)
    with patch("app.retrieval.log") as mock_log:
        retriever._get_relevant_documents("query")
        results_call = mock_log.info.call_args_list[1]
        citations = results_call[1]["extra"]["citations"]
        assert len(citations[0]["content_preview"]) <= 150


# ---------------------------------------------------------------------------
# get_retriever_tool — end-to-end construction (all external deps mocked)
# ---------------------------------------------------------------------------


def test_get_retriever_tool_returns_tool_with_correct_name():
    """get_retriever_tool() must return a tool named 'search_legal_knowledge'."""
    from app.retrieval import get_retriever_tool

    mock_client = MagicMock()
    mock_embeddings = MagicMock()
    mock_vector_store = MagicMock()
    inner_retriever = _StubRetriever()
    mock_vector_store.as_retriever.return_value = inner_retriever

    with (
        patch("app.retrieval._get_qdrant_client", return_value=mock_client),
        patch("app.retrieval._get_embeddings", return_value=mock_embeddings),
        patch(
            "app.retrieval._FlatPayloadQdrantVectorStore",
            return_value=mock_vector_store,
        ),
    ):
        tool = get_retriever_tool()

    assert tool.name == "search_legal_knowledge"


def test_get_retriever_tool_passes_correct_search_kwargs():
    from app.config import settings
    from app.retrieval import get_retriever_tool

    mock_vector_store = MagicMock()
    inner_retriever = _StubRetriever()
    mock_vector_store.as_retriever.return_value = inner_retriever

    with (
        patch("app.retrieval._get_qdrant_client", return_value=MagicMock()),
        patch("app.retrieval._get_embeddings", return_value=MagicMock()),
        patch(
            "app.retrieval._FlatPayloadQdrantVectorStore",
            return_value=mock_vector_store,
        ),
    ):
        get_retriever_tool()

    mock_vector_store.as_retriever.assert_called_once_with(
        search_kwargs={"k": settings.RETRIEVAL_LEGAL_TOP_K}
    )


# ---------------------------------------------------------------------------
# expand_chat_query
# ---------------------------------------------------------------------------


def test_expand_chat_query_returns_list_of_strings():
    """Mock LLM returns two sub-queries; function returns them as a list."""
    from app.services.drafting.knowledge_retrieval import expand_chat_query

    fake_response = MagicMock()
    fake_response.content = (
        "Ethiopian Civil Code contract formation elements offer acceptance\n"
        "consent validity essential requirements Article 1675"
    )

    with patch(
        "app.services.drafting.knowledge_retrieval._compliance_llm"
    ) as mock_llm_factory:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        mock_llm_factory.return_value = mock_llm

        result = expand_chat_query("steps to form a legal contract")

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(q, str) and q for q in result)


def test_expand_chat_query_fallback_on_llm_error():
    """If the LLM raises, expand_chat_query returns an empty list."""
    from app.services.drafting.knowledge_retrieval import expand_chat_query

    with patch(
        "app.services.drafting.knowledge_retrieval._compliance_llm"
    ) as mock_llm_factory:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        mock_llm_factory.return_value = mock_llm

        result = expand_chat_query("what is negligence?")

    assert result == []


def test_expand_chat_query_empty_input_returns_empty():
    """Empty / whitespace query skips the LLM call and returns []."""
    from app.services.drafting.knowledge_retrieval import expand_chat_query

    with patch(
        "app.services.drafting.knowledge_retrieval._compliance_llm"
    ) as mock_llm_factory:
        result = expand_chat_query("   ")

    mock_llm_factory.assert_not_called()
    assert result == []


# ---------------------------------------------------------------------------
# _RoutingRetriever — semantic pipeline
# ---------------------------------------------------------------------------


def _make_docs(count: int) -> list[Document]:
    return [
        Document(
            page_content=f"Article {1675 + i} text",
            metadata={"document_id": "civil-code", "item_id": str(1675 + i), "title": ""},
        )
        for i in range(count)
    ]


class _MockRouteDecision:
    def __init__(self, expect_kb_gap=False, forbidden_primary=None, query_suffix=""):
        self.expect_kb_gap = expect_kb_gap
        self.forbidden_primary = forbidden_primary
        self.query_suffix = query_suffix


def test_routing_retriever_uses_original_query_for_rerank():
    """Cohere rerank must always receive the original (non-enriched) query."""
    from app.retrieval import _RoutingRetriever

    candidates = _make_docs(15)
    mock_rerank_resp = MagicMock()
    mock_rerank_resp.results = [MagicMock(index=i) for i in range(12)]
    mock_cohere = MagicMock()
    mock_cohere.rerank.return_value = mock_rerank_resp

    fake_decision = _MockRouteDecision()

    # Patch at the source modules so the local imports inside _get_relevant_documents
    # pick up the mocks when the function re-executes the from-imports.
    with (
        patch("app.services.legal.instrument_router.route", return_value=fake_decision),
        patch(
            "app.services.drafting.knowledge_retrieval.expand_chat_query",
            return_value=["query angle 1"],
        ),
        patch(
            "app.services.drafting.knowledge_retrieval.search_legal_knowledge",
            return_value=candidates,
        ),
        patch("app.retrieval._get_cohere_client", return_value=mock_cohere),
        patch(
            "app.retrieval.settings",
            RETRIEVAL_LEGAL_FETCH_K=15,
            RETRIEVAL_LEGAL_RERANK_TOP_K=12,
            COHERE_RERANK_MODEL="rerank-multilingual-v3.0",
            COHERE_API_KEY="test-key",
        ),
    ):
        retriever = _RoutingRetriever(retriever=_StubRetriever())
        retriever._get_relevant_documents("steps to form a legal contract")

    # Verify the reranker was called with the ORIGINAL query
    mock_cohere.rerank.assert_called_once()
    call_kwargs = mock_cohere.rerank.call_args
    assert call_kwargs.kwargs.get("query") == "steps to form a legal contract"


def test_routing_retriever_fallback_no_cohere_key():
    """When COHERE_API_KEY is absent the rerank step is skipped; RRF top results returned."""
    from app.retrieval import _RoutingRetriever

    candidates = _make_docs(20)
    fake_decision = _MockRouteDecision()

    with (
        patch("app.services.legal.instrument_router.route", return_value=fake_decision),
        patch(
            "app.services.drafting.knowledge_retrieval.expand_chat_query",
            return_value=["q2"],
        ),
        patch(
            "app.services.drafting.knowledge_retrieval.search_legal_knowledge",
            return_value=candidates,
        ),
        patch(
            "app.retrieval._get_cohere_client",
            side_effect=ValueError("COHERE_API_KEY is required"),
        ),
        patch(
            "app.retrieval.settings",
            RETRIEVAL_LEGAL_FETCH_K=15,
            RETRIEVAL_LEGAL_RERANK_TOP_K=12,
            COHERE_RERANK_MODEL="rerank-multilingual-v3.0",
            COHERE_API_KEY="",
        ),
    ):
        retriever = _RoutingRetriever(retriever=_StubRetriever())
        result = retriever._get_relevant_documents("contract obligations")

    # Should return the top-12 from the RRF candidates, not error
    assert len(result) == 12
    assert result == candidates[:12]


def test_routing_retriever_fallback_rerank_error():
    """If Cohere rerank raises, the retriever falls back to top-k RRF results."""
    from app.retrieval import _RoutingRetriever

    candidates = _make_docs(20)
    fake_decision = _MockRouteDecision()
    mock_cohere = MagicMock()
    mock_cohere.rerank.side_effect = RuntimeError("Cohere service unavailable")

    with (
        patch("app.services.legal.instrument_router.route", return_value=fake_decision),
        patch(
            "app.services.drafting.knowledge_retrieval.expand_chat_query",
            return_value=["q2"],
        ),
        patch(
            "app.services.drafting.knowledge_retrieval.search_legal_knowledge",
            return_value=candidates,
        ),
        patch("app.retrieval._get_cohere_client", return_value=mock_cohere),
        patch(
            "app.retrieval.settings",
            RETRIEVAL_LEGAL_FETCH_K=15,
            RETRIEVAL_LEGAL_RERANK_TOP_K=12,
            COHERE_RERANK_MODEL="rerank-multilingual-v3.0",
            COHERE_API_KEY="key",
        ),
    ):
        retriever = _RoutingRetriever(retriever=_StubRetriever())
        result = retriever._get_relevant_documents("inheritance")

    assert len(result) == 12
    assert result == candidates[:12]


def test_routing_retriever_fallback_expansion_error():
    """If expand_chat_query returns [], the original query alone is used for search."""
    from app.retrieval import _RoutingRetriever

    candidates = _make_docs(10)
    mock_rerank_resp = MagicMock()
    mock_rerank_resp.results = [MagicMock(index=i) for i in range(10)]
    mock_cohere = MagicMock()
    mock_cohere.rerank.return_value = mock_rerank_resp
    fake_decision = _MockRouteDecision()

    with (
        patch("app.services.legal.instrument_router.route", return_value=fake_decision),
        patch(
            "app.services.drafting.knowledge_retrieval.expand_chat_query",
            return_value=[],  # expansion returned nothing
        ),
        patch(
            "app.services.drafting.knowledge_retrieval.search_legal_knowledge",
            return_value=candidates,
        ) as mock_search,
        patch("app.retrieval._get_cohere_client", return_value=mock_cohere),
        patch(
            "app.retrieval.settings",
            RETRIEVAL_LEGAL_FETCH_K=15,
            RETRIEVAL_LEGAL_RERANK_TOP_K=12,
            COHERE_RERANK_MODEL="rerank-multilingual-v3.0",
            COHERE_API_KEY="key",
        ),
    ):
        retriever = _RoutingRetriever(retriever=_StubRetriever())
        retriever._get_relevant_documents("employment termination notice")

    # search_legal_knowledge must have been called with only the original query
    call_args = mock_search.call_args
    queries_passed = call_args.args[0] if call_args.args else call_args.kwargs.get("queries", [])
    assert queries_passed == ["employment termination notice"]


def test_routing_retriever_fallback_search_error():
    """If search_legal_knowledge raises, the legacy dense retriever is used as final fallback."""
    from app.retrieval import _RoutingRetriever

    legacy_docs = _make_docs(5)
    stub = _StubRetriever(docs=legacy_docs)
    fake_decision = _MockRouteDecision()

    with (
        patch("app.services.legal.instrument_router.route", return_value=fake_decision),
        patch(
            "app.services.drafting.knowledge_retrieval.expand_chat_query",
            return_value=["q2"],
        ),
        patch(
            "app.services.drafting.knowledge_retrieval.search_legal_knowledge",
            side_effect=RuntimeError("Qdrant unreachable"),
        ),
        patch("app.retrieval._get_cohere_client", return_value=MagicMock()),
        patch(
            "app.retrieval.settings",
            RETRIEVAL_LEGAL_FETCH_K=15,
            RETRIEVAL_LEGAL_RERANK_TOP_K=12,
            COHERE_RERANK_MODEL="rerank-multilingual-v3.0",
            COHERE_API_KEY="key",
        ),
    ):
        retriever = _RoutingRetriever(retriever=stub)
        result = retriever._get_relevant_documents("property rights")

    # Falls back to the stub retriever's docs
    assert result == legacy_docs
