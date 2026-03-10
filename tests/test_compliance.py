"""Tests for compliance analysis: models, TipTap extraction, RRF, endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.models.drafting.compliance import (
    ComplianceAnalysisRequest,
    ComplianceAnalysisResponse,
    LegalCitation,
)


# ---------------------------------------------------------------------------
# ComplianceAnalysisRequest validation
# ---------------------------------------------------------------------------


def test_request_requires_doc_id():
    """Missing doc_id yields 422 (Pydantic validation)."""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    r = client.post("/drafting/compliance/analyze", json={"language": "en"})
    assert r.status_code == 422


def test_request_accepts_doc_id_and_language():
    r = ComplianceAnalysisRequest(doc_id="8749c6dc-4bb3-4f5c-b593-ae54d0da5437", language="en")
    assert r.doc_id == "8749c6dc-4bb3-4f5c-b593-ae54d0da5437"
    assert r.language == "en"


def test_request_language_default():
    r = ComplianceAnalysisRequest(doc_id="some-uuid")
    assert r.language == "en"


def test_request_check_level_default_and_accepted():
    r = ComplianceAnalysisRequest(doc_id="some-uuid")
    assert r.check_level == "quick"
    r_std = ComplianceAnalysisRequest(doc_id="x", check_level="standard")
    assert r_std.check_level == "standard"
    r_deep = ComplianceAnalysisRequest(doc_id="x", check_level="deep")
    assert r_deep.check_level == "deep"


# ---------------------------------------------------------------------------
# Level config
# ---------------------------------------------------------------------------


def test_get_compliance_limits_quick():
    from app.services.drafting.compliance.level_config import get_compliance_limits

    limits = get_compliance_limits("quick")
    assert limits["doc_char_limit"] == 12_000
    assert limits["blocks_limit"] == 50
    assert limits["rerank_top"] == 10
    assert limits["max_clauses_citations"] == 20


def test_get_compliance_limits_unknown_falls_back_to_quick():
    from app.services.drafting.compliance.level_config import get_compliance_limits

    limits = get_compliance_limits("unknown")
    assert limits["doc_char_limit"] == 12_000
    assert limits["blocks_limit"] == 50


def test_get_compliance_limits_standard_and_deep():
    from app.services.drafting.compliance.level_config import get_compliance_limits

    standard = get_compliance_limits("standard")
    assert standard["doc_char_limit"] == 28_000
    assert standard["blocks_limit"] == 100
    assert standard["max_clauses_citations"] == 35
    deep = get_compliance_limits("deep")
    assert deep["doc_char_limit"] == 50_000
    assert deep["max_clauses_citations"] == 50


# ---------------------------------------------------------------------------
# TipTap extraction
# ---------------------------------------------------------------------------


def test_extract_blocks_from_tiptap():
    from app.services.drafting.compliance.analysis_agent import extract_blocks_from_tiptap

    tiptap = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "First paragraph."}]},
            {"type": "heading", "attrs": {"level": 1}, "content": [{"type": "text", "text": "Section 1"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "Second."}]},
        ],
    }
    blocks = extract_blocks_from_tiptap(tiptap)
    assert len(blocks) >= 2
    texts = [b["text"] for b in blocks if b.get("text")]
    assert "First paragraph." in texts
    assert "Second." in texts


def test_extract_full_document_text_plain():
    from app.services.drafting.compliance.analysis_agent import _extract_full_document_text

    assert _extract_full_document_text(None, "Hello world") == "Hello world"
    assert _extract_full_document_text(None, "  x  ") == "x"


def test_extract_full_document_text_from_tiptap():
    from app.services.drafting.compliance.analysis_agent import _extract_full_document_text

    tiptap = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "A"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "B"}]},
        ],
    }
    assert "A" in _extract_full_document_text(tiptap, None)
    assert "B" in _extract_full_document_text(tiptap, None)


# ---------------------------------------------------------------------------
# RRF merge (knowledge_retrieval)
# ---------------------------------------------------------------------------


def test_rrf_merge_dedupes_by_document_item():
    from app.services.drafting.knowledge_retrieval import _doc_key, _rrf_merge

    d1 = Document(page_content="X", metadata={"document_id": "Code", "item_id": "1"})
    d2 = Document(page_content="Y", metadata={"document_id": "Code", "item_id": "2"})
    d1_dup = Document(page_content="X again", metadata={"document_id": "Code", "item_id": "1"})
    assert _doc_key(d1) == _doc_key(d1_dup)
    ranked = [[d1, d2], [d1_dup, d2]]
    merged = _rrf_merge(ranked, top_k=5)
    # Should dedupe (Code, 1) and (Code, 2)
    keys = [_doc_key(x) for x in merged]
    assert keys.count(("Code", "1")) == 1
    assert keys.count(("Code", "2")) == 1


# ---------------------------------------------------------------------------
# Citation format and response schema
# ---------------------------------------------------------------------------


def test_legal_citation_has_required_fields():
    c = LegalCitation(document_id="Civil Code", item_id="1802", title="Liability", excerpt="...")
    assert c.document_id == "Civil Code"
    assert c.item_id == "1802"


def test_compliance_response_schema():
    resp = ComplianceAnalysisResponse(
        document_type="Contract",
        overall_risk_level="LOW",
        risk_score=25.5,
        summary="Summary",
        clauses=[],
        issues_by_block_id={},
        recommendations=[],
        critical_issues=[],
        missing_clauses=[],
        citations=[],
    )
    assert resp.document_type == "Contract"
    assert resp.risk_score == 25.5
    assert resp.ethiopian_law_compliance is not None


# ---------------------------------------------------------------------------
# Endpoint integration (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compliance_analyze_endpoint_returns_422_without_doc_id():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    r = client.post("/drafting/compliance/analyze", json={"language": "en"})
    assert r.status_code == 422


def test_compliance_analyze_route_registered():
    """Compliance analyze route is mounted at POST /drafting/compliance/analyze."""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    r = client.post("/drafting/compliance/analyze", json={})
    assert r.status_code == 422


def test_compliance_analyze_endpoint_404_when_document_not_found():
    """When get_document_blocks_by_doc_id returns empty list, endpoint returns 404."""
    from fastapi.testclient import TestClient

    from app.main import app

    with patch("app.api.v1.endpoints.drafting.compliance.get_document_blocks_by_doc_id") as mock_get:
        mock_get.return_value = []
        client = TestClient(app)
        r = client.post(
            "/drafting/compliance/analyze",
            json={"doc_id": "missing-doc", "language": "en"},
        )
        assert r.status_code == 404
        assert "not found" in r.json().get("detail", "").lower() or "no content" in r.json().get("detail", "").lower()


def test_compliance_analyze_endpoint_returns_schema_with_mocked_agent():
    """With get_document_blocks_by_doc_id and agent mocked, response has required schema and blocks passed with block_id/type."""
    from fastapi.testclient import TestClient

    from app.main import app

    sample_blocks = [
        {"block_id": "b31", "text": "Tax and Pension clause.", "type": "paragraph", "doc_id": "some-uuid"},
    ]
    with (
        patch("app.api.v1.endpoints.drafting.compliance.get_document_blocks_by_doc_id") as mock_get_blocks,
        patch("app.api.v1.endpoints.drafting.compliance.ComplianceAnalysisAgent") as MockAgent,
    ):
        mock_get_blocks.return_value = sample_blocks
        mock_response = ComplianceAnalysisResponse(
            document_type="Contract",
            overall_risk_level="MEDIUM",
            risk_score=50.0,
            summary="Test summary",
            clauses=[],
            issues_by_block_id={},
            recommendations=[],
            critical_issues=[],
            missing_clauses=[],
            citations=[
                LegalCitation(document_id="Civil Code", item_id="1802", title="Art 1802", excerpt="..."),
            ],
        )
        MockAgent.return_value.analyze_document.return_value = mock_response

        client = TestClient(app)
        r = client.post(
            "/drafting/compliance/analyze",
            json={"doc_id": "some-uuid", "language": "en"},
        )
        assert r.status_code == 200
        mock_get_blocks.assert_called_once_with("some-uuid")
        MockAgent.return_value.analyze_document.assert_called_once()
        call_kw = MockAgent.return_value.analyze_document.call_args[1]
        assert call_kw["document_blocks"] == sample_blocks
        assert call_kw["language"] == "en"
        assert call_kw["document_type"] is None
        assert call_kw.get("check_level", "quick") == "quick"
        data = r.json()
        assert data["document_type"] == "Contract"
        assert data["overall_risk_level"] == "MEDIUM"
        assert data["risk_score"] == 50.0
        assert "summary" in data
        assert "citations" in data
        assert len(data["citations"]) == 1
        assert data["citations"][0]["document_id"] == "Civil Code"
        assert data["citations"][0]["item_id"] == "1802"


def test_compliance_analyze_endpoint_passes_check_level():
    """Endpoint passes check_level to agent.analyze_document."""
    from fastapi.testclient import TestClient

    from app.main import app

    sample_blocks = [{"block_id": "b1", "text": "Sample.", "type": "paragraph", "doc_id": "doc-1"}]
    with (
        patch("app.api.v1.endpoints.drafting.compliance.get_document_blocks_by_doc_id") as mock_get,
        patch("app.api.v1.endpoints.drafting.compliance.ComplianceAnalysisAgent") as MockAgent,
    ):
        mock_get.return_value = sample_blocks
        mock_resp = ComplianceAnalysisResponse(
            document_type="Contract",
            overall_risk_level="LOW",
            risk_score=10.0,
            summary="OK",
            clauses=[],
            issues_by_block_id={},
            recommendations=[],
            critical_issues=[],
            missing_clauses=[],
            citations=[],
        )
        MockAgent.return_value.analyze_document.return_value = mock_resp
        client = TestClient(app)
        r = client.post(
            "/drafting/compliance/analyze",
            json={"doc_id": "doc-1", "language": "en", "check_level": "standard"},
        )
        assert r.status_code == 200
        call_kw = MockAgent.return_value.analyze_document.call_args[1]
        assert call_kw["check_level"] == "standard"


# ---------------------------------------------------------------------------
# analysis_agent helpers (coverage)
# ---------------------------------------------------------------------------


def test_detect_document_type_nda():
    from app.services.drafting.compliance.analysis_agent import _detect_document_type

    assert _detect_document_type("Confidential. Do not disclose. NDA.") == "NDA"
    assert _detect_document_type("CONFIDENTIAL and disclose restrictions") == "NDA"


def test_detect_document_type_mou():
    from app.services.drafting.compliance.analysis_agent import _detect_document_type

    assert _detect_document_type("Memorandum of Understanding between parties") == "MOU"
    assert _detect_document_type("This MOU sets out...") == "MOU"


def test_detect_document_type_employment():
    from app.services.drafting.compliance.analysis_agent import _detect_document_type

    assert _detect_document_type("Employment contract between X and Y") == "Employment Agreement"


def test_detect_document_type_lease():
    from app.services.drafting.compliance.analysis_agent import _detect_document_type

    assert _detect_document_type("Lease agreement. Tenant shall pay.") == "Lease"


def test_detect_document_type_default_contract():
    from app.services.drafting.compliance.analysis_agent import _detect_document_type

    assert _detect_document_type("Some random agreement.") == "Contract"


def test_docs_to_citations():
    from app.services.drafting.compliance.analysis_agent import _docs_to_citations

    docs = [
        Document(page_content="Art 1802.", metadata={"document_id": "Civil Code", "item_id": "1802", "title": "Liability"}),
    ]
    cites = _docs_to_citations(docs)
    assert len(cites) == 1
    assert cites[0].document_id == "Civil Code"
    assert cites[0].item_id == "1802"
    assert "1802" in cites[0].excerpt


def test_docs_to_citations_uses_fallback_keys_and_skips_empty():
    from app.services.drafting.compliance.analysis_agent import _docs_to_citations

    docs = [
        Document(page_content="", metadata={}),
        Document(page_content="Article text.", metadata={"source_file": "family-code", "article_id": "85"}),
    ]
    cites = _docs_to_citations(docs)
    assert len(cites) == 1
    assert cites[0].document_id == "family-code"
    assert cites[0].item_id == "85"
    assert cites[0].excerpt == "Article text."


def test_format_legal_context():
    from app.services.drafting.compliance.analysis_agent import _format_legal_context

    docs = [
        Document(page_content="Text.", metadata={"document_id": "Code", "item_id": "1", "title": "T"}),
    ]
    s = _format_legal_context(docs)
    assert "[Source:" in s
    assert "Article 1" in s
    assert "Text." in s


def test_parse_analysis_response_raw_json():
    from app.services.drafting.compliance.analysis_agent import _parse_analysis_response

    out = _parse_analysis_response('{"document_type": "Contract", "risk_score": 50}')
    assert out["document_type"] == "Contract"
    assert out["risk_score"] == 50


def test_parse_analysis_response_markdown_code_block():
    from app.services.drafting.compliance.analysis_agent import _parse_analysis_response

    out = _parse_analysis_response('```json\n{"document_type": "NDA"}\n```')
    assert out["document_type"] == "NDA"


def test_parse_analysis_response_unclosed_code_fence():
    """Unclosed ```json fence is stripped so JSON parses (avoids 'Expecting value: line 1 column 1')."""
    from app.services.drafting.compliance.analysis_agent import _parse_analysis_response

    raw = '```json\n{"document_type": "Contract", "overall_risk_level": "MEDIUM", "risk_score": 65}'
    out = _parse_analysis_response(raw)
    assert out["document_type"] == "Contract"
    assert out["overall_risk_level"] == "MEDIUM"
    assert out["risk_score"] == 65


def test_parse_analysis_response_truncated_repaired():
    """Truncated JSON (unterminated string) is repaired so parsing succeeds; truncated value is preserved."""
    from app.services.drafting.compliance.analysis_agent import _parse_analysis_response

    # Simulates LLM output cut off mid-string (e.g. max_tokens). Repair at end of text preserves value.
    raw = '{"document_type": "Contract", "overall_risk_level": "MEDIUM", "risk_score": 65, "summary": "A long summary that got cut'
    out = _parse_analysis_response(raw)
    assert out["document_type"] == "Contract"
    assert out["overall_risk_level"] == "MEDIUM"
    assert out["risk_score"] == 65
    assert out.get("summary") == "A long summary that got cut"


def test_validate_and_dedupe_clauses():
    from app.services.drafting.compliance.analysis_agent import _validate_and_dedupe

    data = {"clauses": [{"clause_id": "c1", "text": "A"}, {"clause_id": "c1", "text": "B"}], "issues": []}
    out = _validate_and_dedupe(data)
    assert len(out["clauses"]) == 1
    assert out["clauses"][0]["clause_id"] == "c1"


def test_validate_and_dedupe_issues():
    from app.services.drafting.compliance.analysis_agent import _validate_and_dedupe

    data = {"clauses": [], "issues": [{"issue_id": "i1"}, {"issue_id": "i1"}]}
    out = _validate_and_dedupe(data)
    assert len(out["issues"]) == 1


def test_map_clauses_to_blocks():
    from app.services.drafting.compliance.analysis_agent import _map_clauses_to_blocks

    clauses = [{"clause_id": "1", "text": "First clause here.", "block_id": None}]
    blocks = [{"block_id": "b0", "text": "First clause here."}]
    _map_clauses_to_blocks(clauses, blocks)
    assert clauses[0]["block_id"] == "b0"


def test_clauses_fallback_from_blocks():
    from app.services.drafting.compliance.analysis_agent import _clauses_fallback_from_blocks

    blocks = [
        {"block_id": "b1", "text": "First paragraph.", "type": "paragraph"},
        {"block_id": "b2", "text": ""},
        {"block_id": "b3", "text": "Third paragraph."},
    ]
    out = _clauses_fallback_from_blocks(blocks)
    assert len(out) == 2
    assert out[0]["clause_id"] == "clause_1"
    assert out[0]["text"] == "First paragraph."
    assert out[0]["risk_level"] == "LOW"
    assert out[0]["block_id"] == "b1"
    assert out[1]["clause_id"] == "clause_3"
    assert out[1]["text"] == "Third paragraph."


# ---------------------------------------------------------------------------
# knowledge_retrieval (mocked)
# ---------------------------------------------------------------------------


def test_generate_targeted_queries_mocked_llm():
    from app.services.drafting.knowledge_retrieval import generate_targeted_queries

    with patch("app.services.drafting.knowledge_retrieval._compliance_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(
            content="Ethiopian Civil Code contract\nLabour Proclamation termination"
        )
        queries = generate_targeted_queries("Contract", "Summary of a contract.")
        assert len(queries) >= 1
        assert "Ethiopian" in queries[0] or "contract" in queries[0].lower()


def test_generate_targeted_queries_fallback_when_empty():
    from app.services.drafting.knowledge_retrieval import generate_targeted_queries

    with patch("app.services.drafting.knowledge_retrieval._compliance_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(content="")
        queries = generate_targeted_queries("NDA", "")
        assert len(queries) == 1
        assert "NDA" in queries[0] or "contract" in queries[0].lower()


def test_rerank_with_llm_empty_chunks():
    from app.services.drafting.knowledge_retrieval import rerank_with_llm

    assert rerank_with_llm("query", [], 5) == []


def test_rerank_with_llm_fewer_chunks_than_top_k():
    from app.services.drafting.knowledge_retrieval import rerank_with_llm

    docs = [Document(page_content="A", metadata={}), Document(page_content="B", metadata={})]
    result = rerank_with_llm("q", docs, 10)
    assert len(result) == 2


def test_rerank_with_llm_mocked():
    from app.services.drafting.knowledge_retrieval import rerank_with_llm

    d1 = Document(page_content="First", metadata={})
    d2 = Document(page_content="Second", metadata={})
    d3 = Document(page_content="Third", metadata={})
    d4 = Document(page_content="Fourth", metadata={})
    with patch("app.services.drafting.knowledge_retrieval._compliance_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(content="2, 1, 3")
        out = rerank_with_llm("q", [d1, d2, d3, d4], top_k=3)
        assert len(out) == 3
        assert out[0].page_content == "Second"
        assert out[1].page_content == "First"
        assert out[2].page_content == "Third"


def test_search_legal_knowledge_mocked_store():
    from app.services.drafting.knowledge_retrieval import search_legal_knowledge

    doc = Document(page_content="Law text.", metadata={"document_id": "Code", "item_id": "1"})
    with patch("app.services.drafting.knowledge_retrieval.get_legal_kb_vector_store") as mock_store:
        mock_store.return_value.similarity_search.return_value = [doc]
        results = search_legal_knowledge(["query1", "query2"], top_k_per_query=2, rrf_top_k=5)
        assert len(results) >= 1
        assert results[0].page_content == "Law text."


def test_build_clause_legal_query_mocked():
    """build_clause_legal_query returns LLM-generated query + dynamic synonyms; empty on failure."""
    from app.services.drafting.knowledge_retrieval import build_clause_legal_query

    # Two-line format: line 1 = query, line 2 = comma-separated synonyms
    with patch("app.services.drafting.knowledge_retrieval._compliance_llm") as mock_llm_cls:
        mock_llm_cls.return_value.invoke.return_value = MagicMock(
            content="Ethiopian law employee prohibition similar trade\nprivate trade, Article 29, Civil Code, restraint of trade"
        )
        out = build_clause_legal_query("Employee shall not solicit clients.", "Restraint of trade.")
        assert "Ethiopian law" in out or "prohibition" in out
        assert "private trade" in out
        assert "Article 29" in out
    # Single line only: still used as query
    with patch("app.services.drafting.knowledge_retrieval._compliance_llm") as mock_llm_cls:
        mock_llm_cls.return_value.invoke.return_value = MagicMock(
            content="Labour Proclamation termination notice period"
        )
        out = build_clause_legal_query("Termination clause.", "Notice.")
        assert "Labour Proclamation" in out or "termination" in out
    # Failure: returns empty
    with patch("app.services.drafting.knowledge_retrieval._compliance_llm") as mock_llm_cls:
        mock_llm_cls.return_value.invoke.side_effect = Exception("api error")
        out = build_clause_legal_query("Some clause.", "Implications.")
        assert out == ""

    # Long article-number list is capped: only a few article-like terms and max synonym count
    with patch("app.services.drafting.knowledge_retrieval._compliance_llm") as mock_llm_cls:
        # Simulate LLM dumping Article 3325, 3326, ... 3340
        synonyms = ", ".join([f"Article {n}" for n in range(3325, 3341)])
        mock_llm_cls.return_value.invoke.return_value = MagicMock(
            content=f"Ethiopian arbitration law\n{synonyms}"
        )
        out = build_clause_legal_query("Arbitration clause.", "Dispute resolution.")
        # Should contain query and at most MAX_ARTICLE_LIKE_TERMS article refs; total length capped
        assert "Ethiopian" in out or "arbitration" in out
        assert out.count("Article") <= 3
        assert len(out) <= 1200 + 50  # MAX_CLAUSE_QUERY_CHARS + small buffer


def test_get_document_blocks_by_doc_id_mocked():
    """get_document_blocks_by_doc_id returns list of blocks with block_id, text, type, doc_id."""
    from app.retrieval import get_document_blocks_by_doc_id

    point1 = MagicMock()
    point1.id = 1
    point1.payload = {"doc_id": "doc-1", "block_id": "b31", "text": "Tax and Pension clause.", "type": "paragraph", "index": 0}
    point2 = MagicMock()
    point2.id = 2
    point2.payload = {"doc_id": "doc-1", "text": "Second block.", "index": 1}
    with patch("app.retrieval._get_qdrant_client") as mock_client:
        mock_client.return_value.scroll.return_value = ([point1, point2], None)
        blocks = get_document_blocks_by_doc_id("doc-1")
    assert len(blocks) == 2
    assert blocks[0]["block_id"] == "b31"
    assert blocks[0]["text"] == "Tax and Pension clause."
    assert blocks[0]["type"] == "paragraph"
    assert blocks[0]["doc_id"] == "doc-1"
    assert blocks[1]["text"] == "Second block."
    assert blocks[1]["block_id"] == "2"
    assert blocks[1]["type"] == "paragraph"


def test_get_document_text_by_doc_id_mocked():
    """get_document_text_by_doc_id returns concatenated block text (via get_document_blocks_by_doc_id)."""
    from app.retrieval import get_document_text_by_doc_id

    point1 = MagicMock()
    point1.id = 1
    point1.payload = {"doc_id": "doc-1", "text": "First block.", "index": 0}
    point2 = MagicMock()
    point2.id = 2
    point2.payload = {"doc_id": "doc-1", "text": "Second block.", "index": 1}
    with patch("app.retrieval._get_qdrant_client") as mock_client:
        mock_client.return_value.scroll.return_value = ([point1, point2], None)
        out = get_document_text_by_doc_id("doc-1")
    assert "First block." in out
    assert "Second block." in out
    assert out == "First block.\n\nSecond block."


def test_get_document_blocks_by_doc_id_empty():
    """When no points found, returns empty list."""
    from app.retrieval import get_document_blocks_by_doc_id

    with patch("app.retrieval._get_qdrant_client") as mock_client:
        mock_client.return_value.scroll.return_value = ([], None)
        blocks = get_document_blocks_by_doc_id("nonexistent")
    assert blocks == []


def test_get_document_text_by_doc_id_empty():
    """When no points found, returns empty string."""
    from app.retrieval import get_document_text_by_doc_id

    with patch("app.retrieval._get_qdrant_client") as mock_client:
        mock_client.return_value.scroll.return_value = ([], None)
        out = get_document_text_by_doc_id("nonexistent")
    assert out == ""


def test_get_available_source_files_mocked():
    import app.services.drafting.knowledge_retrieval as kr

    kr._available_sources_cache = None
    try:
        with patch("app.retrieval._get_qdrant_client") as mock_client:
            mock_point = MagicMock()
            mock_point.payload = {"document_id": "Civil Code", "item_id": "1"}
            mock_client.return_value.scroll.return_value = ([mock_point], None)
            sources = kr.get_available_source_files()
            assert "Civil Code" in sources
    finally:
        kr._available_sources_cache = None


# ---------------------------------------------------------------------------
# Full agent analyze_document (all deps mocked)
# ---------------------------------------------------------------------------


def test_analyze_document_full_pipeline_mocked():
    from app.services.drafting.compliance.analysis_agent import ComplianceAnalysisAgent

    llm_json = """
    {
      "document_type": "Contract",
      "overall_risk_level": "LOW",
      "risk_score": 20,
      "summary": "Low risk.",
      "clauses": [{"clause_id": "c1", "text": "Clause.", "risk_level": "LOW", "implications": "", "block_id": null, "citations": []}],
      "issues": [],
      "ethiopian_law_compliance": {"summary": "OK", "applicable_laws": [], "concerns": []},
      "recommendations": [],
      "should_sign": true,
      "critical_issues": [],
      "missing_clauses": []
    }
    """
    with (
        patch("app.services.drafting.compliance.analysis_agent.generate_targeted_queries", return_value=["q1"]),
        patch("app.services.drafting.compliance.analysis_agent.search_legal_knowledge") as mock_search,
        patch("app.services.drafting.compliance.analysis_agent.rerank_with_llm") as mock_rerank,
        patch("app.services.drafting.compliance.analysis_agent.ChatOpenAI") as mock_llm_cls,
    ):
        mock_search.return_value = [
            Document(page_content="Law.", metadata={"document_id": "Code", "item_id": "1", "title": "T"}),
        ]
        mock_rerank.return_value = mock_search.return_value
        mock_llm_cls.return_value.invoke.return_value = MagicMock(content=llm_json)

        agent = ComplianceAnalysisAgent()
        resp = agent.analyze_document(document_text="Sample contract.", language="en")

        assert resp.document_type == "Contract"
        assert resp.overall_risk_level == "LOW"
        assert resp.risk_score == 20.0
        assert resp.summary == "Low risk."
        assert len(resp.clauses) == 1
        assert resp.should_sign is True
