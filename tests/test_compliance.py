"""Tests for compliance analysis: models, TipTap extraction, RRF, endpoint."""

import contextlib
import json
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
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
            {
                "type": "heading",
                "attrs": {"level": 1},
                "content": [{"type": "text", "text": "Section 1"}],
            },
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

    with patch(
        "app.api.v1.endpoints.drafting.compliance.get_document_blocks_by_doc_id"
    ) as mock_get:
        mock_get.return_value = []
        client = TestClient(app)
        r = client.post(
            "/drafting/compliance/analyze",
            json={"doc_id": "missing-doc", "language": "en"},
        )
        assert r.status_code == 404
        assert (
            "not found" in r.json().get("detail", "").lower()
            or "no content" in r.json().get("detail", "").lower()
        )


def test_compliance_analyze_endpoint_returns_schema_with_mocked_agent():
    """With get_document_blocks_by_doc_id and agent mocked, response has required schema and blocks passed with block_id/type."""
    from fastapi.testclient import TestClient

    from app.main import app

    sample_blocks = [
        {
            "block_id": "b31",
            "text": "Tax and Pension clause.",
            "type": "paragraph",
            "doc_id": "some-uuid",
        },
    ]
    with (
        patch(
            "app.api.v1.endpoints.drafting.compliance.get_document_blocks_by_doc_id"
        ) as mock_get_blocks,
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
                LegalCitation(
                    document_id="Civil Code", item_id="1802", title="Art 1802", excerpt="..."
                ),
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


def _parse_compliance_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            with contextlib.suppress(json.JSONDecodeError):
                events.append(json.loads(line[6:]))
    return events


@pytest.mark.asyncio
async def test_compliance_analyze_stream_emits_progress_and_result():
    """SSE stream includes progress events (percent) and a final result payload."""
    from app.main import app

    sample_blocks = [{"block_id": "b1", "text": "Sample.", "type": "paragraph", "doc_id": "doc-1"}]
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

    def fake_analyze(**kwargs):
        cb = kwargs.get("progress_callback")
        if cb:
            cb({"phase": "prepare", "percent": 5, "message": "started"})
        return mock_resp

    with (
        patch("app.api.v1.endpoints.drafting.compliance.get_document_blocks_by_doc_id") as mock_get,
        patch("app.api.v1.endpoints.drafting.compliance.ComplianceAnalysisAgent") as MockAgent,
    ):
        mock_get.return_value = sample_blocks
        MockAgent.return_value.analyze_document.side_effect = fake_analyze

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/drafting/compliance/analyze-stream",
                json={"doc_id": "doc-1", "language": "en"},
            )

    assert r.status_code == 200
    assert "text/event-stream" in (r.headers.get("content-type") or "")
    events = _parse_compliance_sse(r.text)
    progress_events = [e for e in events if e.get("type") == "progress"]
    assert len(progress_events) >= 1
    assert progress_events[0]["percent"] == 5
    assert progress_events[0]["phase"] == "prepare"
    result_events = [e for e in events if e.get("type") == "result"]
    assert len(result_events) == 1
    assert result_events[0]["data"]["document_type"] == "Contract"
    assert result_events[0]["data"]["risk_score"] == 10.0


@pytest.mark.asyncio
async def test_compliance_analyze_stream_emits_token_chunks():
    """SSE stream forwards LLM chunk events from token_callback during analysis."""
    from app.main import app

    sample_blocks = [{"block_id": "b1", "text": "Sample.", "type": "paragraph", "doc_id": "doc-1"}]
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

    def fake_analyze(**kwargs):
        token_cb = kwargs.get("token_callback")
        if token_cb:
            token_cb("Part 1 ")
            token_cb("Part 2")
        return mock_resp

    with (
        patch("app.api.v1.endpoints.drafting.compliance.get_document_blocks_by_doc_id") as mock_get,
        patch("app.api.v1.endpoints.drafting.compliance.ComplianceAnalysisAgent") as MockAgent,
    ):
        mock_get.return_value = sample_blocks
        MockAgent.return_value.analyze_document.side_effect = fake_analyze

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/drafting/compliance/analyze-stream",
                json={"doc_id": "doc-1", "language": "en"},
            )

    assert r.status_code == 200
    events = _parse_compliance_sse(r.text)
    token_events = [e for e in events if e.get("type") == "token"]
    assert [e["content"] for e in token_events] == ["Part 1 ", "Part 2"]


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
        Document(
            page_content="Art 1802.",
            metadata={"document_id": "Civil Code", "item_id": "1802", "title": "Liability"},
        ),
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
        Document(
            page_content="Article text.",
            metadata={"source_file": "family-code", "article_id": "85"},
        ),
    ]
    cites = _docs_to_citations(docs)
    assert len(cites) == 1
    assert cites[0].document_id == "family-code"
    assert cites[0].item_id == "85"
    assert cites[0].excerpt == "Article text."


def test_format_legal_context():
    from app.services.drafting.compliance.analysis_agent import _format_legal_context

    docs = [
        Document(
            page_content="Text.", metadata={"document_id": "Code", "item_id": "1", "title": "T"}
        ),
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

    data = {
        "clauses": [{"clause_id": "c1", "text": "A"}, {"clause_id": "c1", "text": "B"}],
        "issues": [],
    }
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


def test_map_clauses_to_blocks_clears_invalid_block_id():
    from app.services.drafting.compliance.analysis_agent import _map_clauses_to_blocks

    clauses = [{"clause_id": "1", "text": "First clause here.", "block_id": "b99"}]
    blocks = [{"block_id": "b0", "text": "First clause here."}]
    _map_clauses_to_blocks(clauses, blocks)
    assert clauses[0]["block_id"] == "b0"


def test_map_issues_to_blocks_clears_invalid_block_id():
    from app.services.drafting.compliance.analysis_agent import _map_issues_to_blocks

    issues = [
        {
            "issue_id": "i1",
            "description": "Problem in the liability section.",
            "block_id": "b31",
        }
    ]
    blocks = [
        {
            "block_id": "real-block-1",
            "text": "Problem in the liability section of this agreement.",
        }
    ]
    _map_issues_to_blocks(issues, blocks)
    assert issues[0]["block_id"] == "real-block-1"


def test_build_analysis_prompt_includes_block_ids():
    from app.services.drafting.compliance.analysis_agent import _build_analysis_prompt

    blocks = [
        {"block_id": "abc-123", "type": "paragraph", "text": "Parties agree to the terms."},
        {"block_id": "def-456", "type": "heading", "text": "Liability"},
    ]
    prompt = _build_analysis_prompt(
        full_text="Parties agree to the terms.\n\nLiability",
        blocks=blocks,
        legal_context="",
        document_type="Contract",
        language="en",
    )
    assert "[block_id: abc-123 | type: paragraph]" in prompt
    assert "[block_id: def-456 | type: heading]" in prompt
    assert "do not invent block_ids" in prompt


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
# ClauseAnalysis — ethiopian_law_implications and recommendations fields
# ---------------------------------------------------------------------------


def test_clause_analysis_new_fields_default_empty():
    from app.models.drafting.compliance import ClauseAnalysis

    c = ClauseAnalysis(clause_id="c1", text="Some clause.", risk_level="LOW")
    assert c.ethiopian_law_implications == []
    assert c.recommendations == []


def test_clause_analysis_new_fields_populated():
    from app.models.drafting.compliance import ClauseAnalysis

    c = ClauseAnalysis(
        clause_id="c2",
        text="Liability clause.",
        risk_level="HIGH",
        ethiopian_law_implications=["Art. 2035 Civil Code applies", "Commercial Code Art. 100"],
        recommendations=["Add liability cap", "Consult counsel"],
    )
    assert c.ethiopian_law_implications == [
        "Art. 2035 Civil Code applies",
        "Commercial Code Art. 100",
    ]
    assert c.recommendations == ["Add liability cap", "Consult counsel"]


# ---------------------------------------------------------------------------
# EditorFixSpec — schema and normalization
# ---------------------------------------------------------------------------


def test_editor_fix_spec_validates():
    from app.models.drafting.compliance import EditorFixLegalBasis, EditorFixSpec

    fix = EditorFixSpec(
        action="replace",
        block_id="b1",
        current_text="Vague compensation terms.",
        problem_summary="Compensation is not specified clearly.",
        legal_requirement="Wages must be stated in the contract.",
        rewrite_directive="State salary amount and payment frequency explicitly.",
        suggested_text="The Employee shall receive [AMOUNT] ETB monthly.",
        severity="high_risk",
        confidence=0.85,
        legal_basis=[
            EditorFixLegalBasis(
                source="ethiopian-labor-proclamation",
                article="Section on wages",
                rationale="Wages must be clear.",
            )
        ],
    )
    assert fix.action == "replace"
    assert fix.confidence == 0.85


def test_clause_analysis_editor_fix_optional():
    from app.models.drafting.compliance import ClauseAnalysis, EditorFixSpec

    fix = EditorFixSpec(
        action="replace",
        block_id="b1",
        current_text="Old text.",
        problem_summary="Problem.",
        legal_requirement="Requirement.",
        rewrite_directive="Rewrite explicitly.",
        suggested_text="New text.",
        severity="high_risk",
        confidence=0.9,
    )
    c = ClauseAnalysis(
        clause_id="c1",
        text="Old text.",
        risk_level="HIGH",
        editor_fix=fix,
    )
    assert c.editor_fix is not None
    assert c.editor_fix.suggested_text == "New text."

    c2 = ClauseAnalysis(clause_id="c2", text="Fine.", risk_level="LOW")
    assert c2.editor_fix is None


def test_build_analysis_prompt_includes_editor_fix():
    from app.services.drafting.compliance.analysis_agent import _build_analysis_prompt

    prompt = _build_analysis_prompt(
        full_text="Compensation clause.",
        blocks=[{"block_id": "b1", "type": "paragraph", "text": "Compensation."}],
        legal_context="",
        document_type="Employment Agreement",
        language="en",
    )
    assert "editor_fix" in prompt
    assert "MEDIUM, HIGH, or CRITICAL" in prompt
    assert '"editor_fix": null' in prompt or "editor_fix\": null" in prompt


def test_normalize_editor_fix_enriches_current_text_from_block():
    from app.services.drafting.compliance.analysis_agent import _normalize_editor_fix

    blocks = [
        {
            "block_id": "b1",
            "text": "The Employee's compensation may include benefits as defined internally by the Company.",
            "type": "paragraph",
        }
    ]
    clause = {
        "clause_id": "c1",
        "text": "Compensation excerpt.",
        "risk_level": "HIGH",
        "block_id": "b1",
        "citations": [],
    }
    raw = {
        "action": "replace",
        "block_id": "b1",
        "problem_summary": "Vague compensation terms.",
        "legal_requirement": "State wages clearly.",
        "rewrite_directive": "Specify salary and benefits explicitly.",
        "suggested_text": "The Employee shall receive [AMOUNT] ETB monthly.",
        "confidence": 0.85,
    }
    fix = _normalize_editor_fix(raw, clause, blocks, "en")
    assert fix is not None
    assert fix.current_text == blocks[0]["text"]
    assert fix.block_id == "b1"
    assert fix.severity == "high_risk"


def test_normalize_editor_fix_accepts_medium_clause():
    from app.services.drafting.compliance.analysis_agent import _normalize_editor_fix

    clause = {"clause_id": "c1", "text": "Clause.", "risk_level": "MEDIUM", "block_id": "b1"}
    raw = {
        "problem_summary": "Minor ambiguity.",
        "legal_requirement": "Terms should be explicit.",
        "rewrite_directive": "Clarify the liability cap amount.",
        "suggested_text": "Liability shall not exceed [AMOUNT] ETB.",
        "confidence": 0.75,
    }
    fix = _normalize_editor_fix(raw, clause, [], "en")
    assert fix is not None
    assert fix.severity == "medium_risk"


def test_normalize_editor_fix_strips_for_low_clause():
    from app.services.drafting.compliance.analysis_agent import _normalize_editor_fix

    clause = {"clause_id": "c1", "text": "Clause.", "risk_level": "LOW", "block_id": "b1"}
    raw = {
        "problem_summary": "Issue.",
        "rewrite_directive": "Fix it.",
        "suggested_text": "Fixed text.",
    }
    assert _normalize_editor_fix(raw, clause, [], "en") is None


def test_normalize_editor_fix_returns_none_when_incomplete():
    from app.services.drafting.compliance.analysis_agent import _normalize_editor_fix

    clause = {"clause_id": "c1", "text": "Clause.", "risk_level": "HIGH", "block_id": None}
    raw = {"problem_summary": "Issue.", "rewrite_directive": "Fix it."}
    assert _normalize_editor_fix(raw, clause, [], "en") is None


def test_risk_level_to_issue_severity_mapping():
    from app.models.drafting.compliance import IssueSeverity
    from app.services.drafting.compliance.analysis_agent import _risk_level_to_issue_severity

    assert _risk_level_to_issue_severity("CRITICAL") == IssueSeverity.ERROR
    assert _risk_level_to_issue_severity("HIGH") == IssueSeverity.ERROR
    assert _risk_level_to_issue_severity("MEDIUM") == IssueSeverity.WARNING
    assert _risk_level_to_issue_severity("LOW") == IssueSeverity.INFO
    assert _risk_level_to_issue_severity("BOGUS") == IssueSeverity.NORMAL


def test_determine_issue_type_from_description():
    from app.services.drafting.compliance.analysis_agent import _determine_issue_type

    assert _determine_issue_type("HIGH", "This clause is non-compliant with labor law") == "non_compliant_clause"
    assert _determine_issue_type("MEDIUM", "Missing termination notice period") == "missing_provision"
    assert _determine_issue_type("LOW", "Some issue") == "low_risk"


def test_analyze_document_maps_issues_for_advisor_schema():
    from unittest.mock import MagicMock, patch

    from langchain_core.documents import Document

    from app.models.drafting.compliance import IssueSeverity
    from app.services.drafting.compliance.analysis_agent import ComplianceAnalysisAgent

    llm_json = """
    {
      "document_type": "Contract",
      "summary": "Test.",
      "clauses": [{"clause_id": "c1", "text": "Clause.", "risk_level": "LOW", "implications": "", "block_id": "b1", "citations": [], "ethiopian_law_implications": [], "recommendations": [], "editor_fix": null}],
      "issues": [
        {"issue_id": "issue_1", "description": "Vague liability cap", "severity": "MEDIUM", "block_id": "b1", "citations": []},
        {"issue_id": "issue_2", "description": "Critical non-compliant termination", "severity": "CRITICAL", "block_id": "b1", "citations": []}
      ],
      "ethiopian_law_compliance": {"summary": "OK", "applicable_laws": [], "concerns": []},
      "recommendations": [],
      "should_sign": null,
      "critical_issues": [],
      "missing_clauses": []
    }
    """
    with (
        patch(
            "app.services.drafting.compliance.analysis_agent.generate_targeted_queries",
            return_value=["q"],
        ),
        patch(
            "app.services.drafting.compliance.analysis_agent.search_legal_knowledge",
            return_value=[
                Document(
                    page_content="Law.",
                    metadata={"document_id": "Code", "item_id": "1", "title": "T"},
                ),
            ],
        ),
        patch(
            "app.services.drafting.compliance.analysis_agent.rerank_with_llm",
            return_value=[
                Document(
                    page_content="Law.",
                    metadata={"document_id": "Code", "item_id": "1", "title": "T"},
                ),
            ],
        ),
        patch("app.services.drafting.compliance.analysis_agent.ChatOpenAI") as mock_llm_cls,
    ):
        mock_llm_cls.return_value.stream.return_value = [MagicMock(content=llm_json)]
        agent = ComplianceAnalysisAgent()
        resp = agent.analyze_document(
            document_blocks=[{"block_id": "b1", "text": "Clause.", "type": "paragraph"}],
            language="en",
        )

    assert "b1" in resp.issues_by_block_id
    issues = resp.issues_by_block_id["b1"]
    assert len(issues) == 2
    assert issues[0].severity == IssueSeverity.WARNING
    assert issues[0].issue_type
    assert issues[1].severity == IssueSeverity.ERROR
    assert issues[0].block_id == "b1"


def test_normalize_editor_fix_legal_basis_from_citations():
    from app.services.drafting.compliance.analysis_agent import _normalize_editor_fix

    clause = {
        "clause_id": "c1",
        "text": "Clause.",
        "risk_level": "CRITICAL",
        "block_id": None,
        "citations": [
            {
                "document_id": "Civil Code",
                "item_id": "2035",
                "title": "Liability",
                "excerpt": "Limits apply.",
            }
        ],
    }
    raw = {
        "problem_summary": "Unlimited liability.",
        "legal_requirement": "Cap liability.",
        "rewrite_directive": "Add a liability cap.",
        "suggested_text": "Liability shall not exceed [AMOUNT] ETB.",
        "confidence": 0.9,
    }
    fix = _normalize_editor_fix(raw, clause, [], "en")
    assert fix is not None
    assert fix.severity == "critical_risk"
    assert len(fix.legal_basis) == 1
    assert fix.legal_basis[0].source == "Civil Code"
    assert fix.legal_basis[0].article == "2035"


def test_to_clause_reads_editor_fix_from_llm_dict():
    """Integration: HIGH clause with editor_fix is parsed and returned in response."""
    from unittest.mock import MagicMock, patch

    from langchain_core.documents import Document

    from app.services.drafting.compliance.analysis_agent import ComplianceAnalysisAgent

    llm_json = """
    {
      "document_type": "Employment Agreement",
      "summary": "Test.",
      "clauses": [
        {
          "clause_id": "c1",
          "text": "Compensation excerpt.",
          "risk_level": "HIGH",
          "implications": "Vague compensation.",
          "block_id": "b1",
          "citations": [],
          "ethiopian_law_implications": ["Labor law requires clear wages"],
          "recommendations": ["Clarify compensation"],
          "editor_fix": {
            "action": "replace",
            "block_id": "b1",
            "clause_reference": "4. Compensation",
            "current_text": "Compensation excerpt.",
            "problem_summary": "Compensation terms are vague.",
            "offending_phrases": ["as defined internally by the Company"],
            "legal_requirement": "Employment terms must specify wages clearly.",
            "rewrite_directive": "Rewrite to state salary, payment frequency, and benefits explicitly.",
            "remove_phrases": ["as defined internally by the Company"],
            "add_elements": ["base salary", "payment frequency"],
            "suggested_text": "The Employee shall receive a monthly base salary of [AMOUNT] ETB.",
            "placeholder_policy": "use_bracketed_placeholders_when_values_unknown",
            "legal_basis": [{"source": "ethiopian-labor-proclamation", "article": "wages", "rationale": "Wages must be clear."}],
            "document_language": "en",
            "severity": "high_risk",
            "confidence": 0.85
          }
        }
      ],
      "issues": [],
      "ethiopian_law_compliance": {"summary": "OK", "applicable_laws": [], "concerns": []},
      "recommendations": [],
      "should_sign": null,
      "critical_issues": [],
      "missing_clauses": []
    }
    """
    blocks = [
        {
            "block_id": "b1",
            "text": "The Employee's compensation may include benefits as defined internally by the Company.",
            "type": "paragraph",
        }
    ]
    with (
        patch(
            "app.services.drafting.compliance.analysis_agent.generate_targeted_queries",
            return_value=["q"],
        ),
        patch(
            "app.services.drafting.compliance.analysis_agent.search_legal_knowledge",
            return_value=[
                Document(
                    page_content="Law.",
                    metadata={"document_id": "Code", "item_id": "1", "title": "T"},
                ),
            ],
        ),
        patch(
            "app.services.drafting.compliance.analysis_agent.rerank_with_llm",
            return_value=[
                Document(
                    page_content="Law.",
                    metadata={"document_id": "Code", "item_id": "1", "title": "T"},
                ),
            ],
        ),
        patch("app.services.drafting.compliance.analysis_agent.ChatOpenAI") as mock_llm_cls,
    ):
        mock_llm_cls.return_value.stream.return_value = [MagicMock(content=llm_json)]
        agent = ComplianceAnalysisAgent()
        resp = agent.analyze_document(document_blocks=blocks, language="en")

    assert len(resp.clauses) == 1
    clause = resp.clauses[0]
    assert clause.editor_fix is not None
    assert clause.editor_fix.rewrite_directive.startswith("Rewrite")
    assert clause.editor_fix.current_text == blocks[0]["text"]
    assert clause.editor_fix.block_id == "b1"
    assert "[AMOUNT]" in clause.editor_fix.suggested_text


def test_to_clause_reads_new_fields_from_llm_dict():
    """to_clause() builder correctly maps new fields from LLM response dict."""
    from unittest.mock import MagicMock, patch

    from langchain_core.documents import Document

    from app.services.drafting.compliance.analysis_agent import ComplianceAnalysisAgent

    llm_json = """
    {
      "document_type": "Contract",
      "summary": "Test.",
      "clauses": [
        {
          "clause_id": "c1",
          "text": "No-compete clause.",
          "risk_level": "HIGH",
          "implications": "Restricts competition.",
          "block_id": null,
          "citations": [],
          "ethiopian_law_implications": ["Commercial Code Art. 11 restricts anti-competitive terms"],
          "recommendations": ["Narrow the geographic scope", "Add sunset clause"]
        }
      ],
      "issues": [],
      "ethiopian_law_compliance": {"summary": "OK", "applicable_laws": [], "concerns": []},
      "recommendations": [],
      "should_sign": null,
      "critical_issues": [],
      "missing_clauses": []
    }
    """
    with (
        patch(
            "app.services.drafting.compliance.analysis_agent.generate_targeted_queries",
            return_value=["q"],
        ),
        patch(
            "app.services.drafting.compliance.analysis_agent.search_legal_knowledge",
            return_value=[
                Document(
                    page_content="Law.",
                    metadata={"document_id": "Code", "item_id": "1", "title": "T"},
                ),
            ],
        ),
        patch(
            "app.services.drafting.compliance.analysis_agent.rerank_with_llm",
            return_value=[
                Document(
                    page_content="Law.",
                    metadata={"document_id": "Code", "item_id": "1", "title": "T"},
                ),
            ],
        ),
        patch("app.services.drafting.compliance.analysis_agent.ChatOpenAI") as mock_llm_cls,
    ):
        mock_llm_cls.return_value.stream.return_value = [MagicMock(content=llm_json)]
        agent = ComplianceAnalysisAgent()
        resp = agent.analyze_document(document_text="No compete sample.", language="en")

    assert len(resp.clauses) == 1
    clause = resp.clauses[0]
    assert clause.risk_level == "HIGH"
    assert clause.ethiopian_law_implications == [
        "Commercial Code Art. 11 restricts anti-competitive terms"
    ]
    assert clause.recommendations == ["Narrow the geographic scope", "Add sunset clause"]


def test_to_clause_missing_new_fields_defaults_to_empty():
    """to_clause() defaults new fields to [] when LLM omits them (backward compat)."""
    from unittest.mock import MagicMock, patch

    from langchain_core.documents import Document

    from app.services.drafting.compliance.analysis_agent import ComplianceAnalysisAgent

    llm_json = """
    {
      "document_type": "Contract",
      "summary": "Old format.",
      "clauses": [{"clause_id": "c1", "text": "Clause.", "risk_level": "MEDIUM", "implications": "Some.", "block_id": null, "citations": []}],
      "issues": [],
      "ethiopian_law_compliance": {"summary": "", "applicable_laws": [], "concerns": []},
      "recommendations": [],
      "should_sign": null,
      "critical_issues": [],
      "missing_clauses": []
    }
    """
    with (
        patch(
            "app.services.drafting.compliance.analysis_agent.generate_targeted_queries",
            return_value=["q"],
        ),
        patch(
            "app.services.drafting.compliance.analysis_agent.search_legal_knowledge",
            return_value=[
                Document(
                    page_content="L.", metadata={"document_id": "C", "item_id": "1", "title": "T"}
                ),
            ],
        ),
        patch(
            "app.services.drafting.compliance.analysis_agent.rerank_with_llm",
            return_value=[
                Document(
                    page_content="L.", metadata={"document_id": "C", "item_id": "1", "title": "T"}
                ),
            ],
        ),
        patch("app.services.drafting.compliance.analysis_agent.ChatOpenAI") as mock_llm_cls,
    ):
        mock_llm_cls.return_value.stream.return_value = [MagicMock(content=llm_json)]
        agent = ComplianceAnalysisAgent()
        resp = agent.analyze_document(document_text="Some contract.", language="en")

    assert resp.clauses[0].ethiopian_law_implications == []
    assert resp.clauses[0].recommendations == []


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
    point1.payload = {
        "doc_id": "doc-1",
        "block_id": "b31",
        "text": "Tax and Pension clause.",
        "type": "paragraph",
        "index": 0,
    }
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
      "summary": "Low risk.",
      "clauses": [
        {"clause_id": "c1", "text": "Clause.", "risk_level": "LOW", "implications": "", "block_id": null, "citations": [], "ethiopian_law_implications": [], "recommendations": []},
        {"clause_id": "c2", "text": "Liability clause.", "risk_level": "MEDIUM", "implications": "Limits liability.", "block_id": null, "citations": [], "ethiopian_law_implications": ["Art. 2035 Civil Code applies"], "recommendations": ["Clarify cap amount"]}
      ],
      "issues": [],
      "ethiopian_law_compliance": {"summary": "OK", "applicable_laws": [], "concerns": []},
      "recommendations": [],
      "should_sign": true,
      "critical_issues": [],
      "missing_clauses": []
    }
    """
    with (
        patch(
            "app.services.drafting.compliance.analysis_agent.generate_targeted_queries",
            return_value=["q1"],
        ),
        patch(
            "app.services.drafting.compliance.analysis_agent.search_legal_knowledge"
        ) as mock_search,
        patch("app.services.drafting.compliance.analysis_agent.rerank_with_llm") as mock_rerank,
        patch("app.services.drafting.compliance.analysis_agent.ChatOpenAI") as mock_llm_cls,
    ):
        mock_search.return_value = [
            Document(
                page_content="Law.", metadata={"document_id": "Code", "item_id": "1", "title": "T"}
            ),
        ]
        mock_rerank.return_value = mock_search.return_value
        mock_llm_cls.return_value.stream.return_value = [MagicMock(content=llm_json)]

        agent = ComplianceAnalysisAgent()
        resp = agent.analyze_document(document_text="Sample contract.", language="en")

        assert resp.document_type == "Contract"
        assert resp.overall_risk_level == "LOW"  # raw_penalty=5, risk_score=3.33 → LOW
        assert resp.risk_score == round(5 / 150 * 100, 2)
        assert resp.compliance_score == round(100 - resp.risk_score, 2)
        assert resp.summary == "Low risk."
        assert len(resp.clauses) == 2
        assert resp.should_sign is True
        # LOW clause: new fields default to []
        low_clause = resp.clauses[0]
        assert low_clause.ethiopian_law_implications == []
        assert low_clause.recommendations == []
        # MEDIUM clause: new fields populated from LLM response
        medium_clause = resp.clauses[1]
        assert medium_clause.ethiopian_law_implications == ["Art. 2035 Civil Code applies"]
        assert medium_clause.recommendations == ["Clarify cap amount"]
        assert isinstance(resp.score_breakdown, dict)
        assert resp.score_breakdown["raw_penalty"] == 5  # 1 MEDIUM clause × 5
        assert resp.score_breakdown["clause_counts"]["MEDIUM"] == 1


# ---------------------------------------------------------------------------
# scoring.py — deterministic engine
# ---------------------------------------------------------------------------


def test_scoring_all_low_clauses_no_issues():
    from app.services.drafting.compliance.scoring import compute_risk_score

    result = compute_risk_score(
        clauses=[{"risk_level": "LOW"}, {"risk_level": "LOW"}],
        issues=[],
        missing_clauses=[],
        should_sign=True,
        concern_count=0,
    )
    assert result["raw_penalty"] == 0
    assert result["risk_score"] == 0.0
    assert result["compliance_score"] == 100.0
    assert result["overall_risk_level"] == "LOW"
    assert result["clause_counts"] == {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 2}


def test_scoring_clause_penalties():
    from app.services.drafting.compliance.scoring import compute_risk_score

    result = compute_risk_score(
        clauses=[
            {"risk_level": "CRITICAL"},
            {"risk_level": "HIGH"},
            {"risk_level": "MEDIUM"},
            {"risk_level": "LOW"},
        ],
        issues=[],
        missing_clauses=[],
        should_sign=True,
        concern_count=0,
    )
    # CRITICAL(20) + HIGH(10) + MEDIUM(5) + LOW(0) = 35
    assert result["clause_penalty_total"] == 35
    assert result["raw_penalty"] == 35
    assert result["issue_penalty_total"] == 0


def test_scoring_issue_penalties():
    from app.services.drafting.compliance.scoring import compute_risk_score

    result = compute_risk_score(
        clauses=[],
        issues=[
            {"severity": "CRITICAL"},
            {"severity": "HIGH"},
            {"severity": "MEDIUM"},
            {"severity": "LOW"},
        ],
        missing_clauses=[],
        should_sign=True,
        concern_count=0,
    )
    # CRITICAL(15) + HIGH(8) + MEDIUM(3) + LOW(1) = 27
    assert result["issue_penalty_total"] == 27
    assert result["raw_penalty"] == 27


def test_scoring_missing_clauses_penalty():
    from app.services.drafting.compliance.scoring import compute_risk_score

    result = compute_risk_score(
        clauses=[],
        issues=[],
        missing_clauses=["clause A", "clause B", "clause C"],
        should_sign=True,
        concern_count=0,
    )
    # 3 missing × 3 = 9
    assert result["missing_clauses_count"] == 3
    assert result["missing_clause_penalty_total"] == 9
    assert result["raw_penalty"] == 9


def test_scoring_should_not_sign_penalty():
    from app.services.drafting.compliance.scoring import compute_risk_score

    with_penalty = compute_risk_score(
        clauses=[], issues=[], missing_clauses=[], should_sign=False, concern_count=0
    )
    without_penalty = compute_risk_score(
        clauses=[], issues=[], missing_clauses=[], should_sign=True, concern_count=0
    )
    assert with_penalty["should_sign_penalty"] == 10
    assert without_penalty["should_sign_penalty"] == 0
    assert with_penalty["raw_penalty"] == 10


def test_scoring_concern_penalty():
    from app.services.drafting.compliance.scoring import compute_risk_score

    result = compute_risk_score(
        clauses=[], issues=[], missing_clauses=[], should_sign=True, concern_count=4
    )
    # 4 concerns × 2 = 8
    assert result["concern_penalty_total"] == 8
    assert result["raw_penalty"] == 8


def test_scoring_normalization_caps_at_100():
    from app.services.drafting.compliance.scoring import compute_risk_score

    # Flood with penalties to exceed the ceiling
    result = compute_risk_score(
        clauses=[{"risk_level": "CRITICAL"}] * 20,
        issues=[{"severity": "CRITICAL"}] * 20,
        missing_clauses=["x"] * 20,
        should_sign=False,
        concern_count=20,
        max_penalty=150,
    )
    assert result["risk_score"] == 100.0
    assert result["compliance_score"] == 0.0


def test_scoring_jv_document_scenario():
    """Reproduces the JV document from the original user query."""
    from app.services.drafting.compliance.scoring import compute_risk_score

    clauses = (
        [{"risk_level": "CRITICAL"}] * 1
        + [{"risk_level": "MEDIUM"}] * 2
        + [{"risk_level": "LOW"}] * 39
    )
    issues = [{"severity": "CRITICAL"}, {"severity": "HIGH"}]
    missing = ["x"] * 11

    result = compute_risk_score(
        clauses=clauses,
        issues=issues,
        missing_clauses=missing,
        should_sign=False,
        concern_count=4,
        max_penalty=150,
    )
    # CRITICAL(20) + 2×MEDIUM(10) + CRITICAL issue(15) + HIGH issue(8) + 11×missing(33) + sign(10) + 4×concern(8)
    assert result["raw_penalty"] == 104
    assert result["risk_score"] == round(104 / 150 * 100, 2)
    assert result["compliance_score"] == round(100 - result["risk_score"], 2)
    assert result["overall_risk_level"] == "HIGH"


def test_scoring_risk_level_thresholds():
    from app.services.drafting.compliance.scoring import compute_risk_score

    def score_at(raw, ceiling=100):
        return compute_risk_score(
            clauses=[],
            issues=[],
            missing_clauses=["x"] * (raw // 3),
            should_sign=not (raw % 3),
            concern_count=0,
            max_penalty=ceiling,
        )

    critical = compute_risk_score(
        clauses=[{"risk_level": "CRITICAL"}] * 6,
        issues=[],
        missing_clauses=[],
        should_sign=False,
        concern_count=0,
        max_penalty=150,
    )
    assert critical["overall_risk_level"] in ("CRITICAL", "HIGH")

    low_result = compute_risk_score(
        clauses=[{"risk_level": "LOW"}],
        issues=[],
        missing_clauses=[],
        should_sign=True,
        concern_count=0,
    )
    assert low_result["overall_risk_level"] == "LOW"


def test_scoring_unknown_risk_level_treated_as_low():
    from app.services.drafting.compliance.scoring import compute_risk_score

    result = compute_risk_score(
        clauses=[{"risk_level": "UNKNOWN"}, {"risk_level": ""}],
        issues=[{"severity": "BOGUS"}],
        missing_clauses=[],
        should_sign=None,
        concern_count=0,
    )
    # All unknowns fall back to LOW/LOW; LOW clause=0, LOW issue=1
    assert result["clause_penalty_total"] == 0
    assert result["issue_penalty_total"] == 1  # LOW issue = 1 pt


def test_scoring_breakdown_keys_present():
    from app.services.drafting.compliance.scoring import compute_risk_score

    result = compute_risk_score(
        clauses=[], issues=[], missing_clauses=[], should_sign=None, concern_count=0
    )
    expected_keys = {
        "clause_counts",
        "clause_penalty_total",
        "issue_counts",
        "issue_penalty_total",
        "missing_clauses_count",
        "missing_clause_penalty_total",
        "should_sign_penalty",
        "concern_count",
        "concern_penalty_total",
        "raw_penalty",
        "max_penalty",
        "risk_score",
        "compliance_score",
        "overall_risk_level",
    }
    assert expected_keys == result.keys()
