"""Tests for compliance determinism, Redis cache, and diff-aware anchoring."""

import json
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.models.drafting.compliance import ComplianceAnalysisResponse
from app.services.drafting.compliance.analysis_agent import (
    ComplianceAnalysisAgent,
    _apply_carryover_guard,
    _build_analysis_llm,
    _ensure_rubric_checks_complete,
)
from app.services.drafting.compliance.compliance_cache import compliance_cache_key
from app.services.drafting.compliance.content_hash import (
    compute_document_content_hash,
    compute_per_block_hashes,
)
from app.services.drafting.compliance.rubric import RUBRIC_VERSION, get_rubric_items_for_document_type


def _mock_rubric_checks_json(document_type: str = "Contract") -> str:
    items = get_rubric_items_for_document_type(document_type)
    checks = [
        {"id": item["id"], "status": "PRESENT", "block_id": None, "rationale": "OK"}
        for item in items
    ]
    return json.dumps(checks)


def _base_llm_json(*, rubric_status: str = "PRESENT", clause_risk: str = "LOW") -> str:
    rubric_checks = json.loads(_mock_rubric_checks_json())
    if rubric_status != "PRESENT":
        rubric_checks[0]["status"] = rubric_status
    return json.dumps(
        {
            "document_type": "Contract",
            "summary": "Test.",
            "rubric_checks": rubric_checks,
            "clauses": [
                {
                    "clause_id": "c1",
                    "text": "Clause.",
                    "risk_level": clause_risk,
                    "implications": "",
                    "block_id": "b1",
                    "citations": [],
                    "ethiopian_law_implications": [],
                    "recommendations": [],
                    "editor_fix": None,
                }
            ],
            "issues": [],
            "ethiopian_law_compliance": {"summary": "OK", "applicable_laws": [], "concerns": []},
            "recommendations": [],
            "should_sign": True,
            "critical_issues": [],
            "missing_clauses": [],
        }
    )


def test_content_hash_stable_for_same_blocks():
    blocks = [
        {"block_id": "b2", "text": "Second."},
        {"block_id": "b1", "text": "First."},
    ]
    h1 = compute_document_content_hash(blocks)
    h2 = compute_document_content_hash(blocks)
    assert h1 == h2


def test_content_hash_changes_when_text_changes():
    blocks_a = [{"block_id": "b1", "text": "Original."}]
    blocks_b = [{"block_id": "b1", "text": "Edited."}]
    assert compute_document_content_hash(blocks_a) != compute_document_content_hash(blocks_b)


def test_per_block_hashes_detect_single_block_change():
    blocks_a = [
        {"block_id": "b1", "text": "Same."},
        {"block_id": "b2", "text": "Unchanged."},
    ]
    blocks_b = [
        {"block_id": "b1", "text": "Changed."},
        {"block_id": "b2", "text": "Unchanged."},
    ]
    ha = compute_per_block_hashes(blocks_a)
    hb = compute_per_block_hashes(blocks_b)
    assert ha["b2"] == hb["b2"]
    assert ha["b1"] != hb["b1"]


def test_carryover_guard_reuses_unchanged_block_status():
    prior = [
        {
            "id": "jurisdiction_governing_law",
            "status": "PRESENT",
            "block_id": "b1",
            "rationale": "Prior OK",
        }
    ]
    new = [
        {
            "id": "jurisdiction_governing_law",
            "status": "MISSING",
            "block_id": "b1",
            "rationale": "LLM changed mind",
        },
        {
            "id": "contract_type_parties",
            "status": "MISSING",
            "block_id": "b2",
            "rationale": "New eval",
        },
    ]
    prior_hashes = {"b1": "hash1", "b2": "hash2"}
    current_hashes = {"b1": "hash1", "b2": "hash3"}
    out = _apply_carryover_guard(new, prior, prior_hashes, current_hashes)
    by_id = {c["id"]: c for c in out}
    assert by_id["jurisdiction_governing_law"]["status"] == "PRESENT"
    assert by_id["jurisdiction_governing_law"].get("carried_over") is True
    assert by_id["contract_type_parties"]["status"] == "MISSING"


def test_ensure_rubric_checks_complete_fills_missing():
    items = get_rubric_items_for_document_type("Contract")
    partial = [{"id": items[0]["id"], "status": "PRESENT", "block_id": None}]
    out = _ensure_rubric_checks_complete(partial, items)
    assert len(out) == len(items)
    assert all(c["id"] for c in out)


def test_build_analysis_llm_passes_seed_and_temperature():
    with patch("app.services.drafting.compliance.analysis_agent.ChatOpenAI") as mock_cls:
        _build_analysis_llm()
        kwargs = mock_cls.call_args[1]
        assert kwargs["temperature"] == 0.0
        assert kwargs["model_kwargs"]["seed"] == 7
        assert kwargs["model_kwargs"]["response_format"] == {"type": "json_object"}


def test_analyze_document_identical_rubric_score_on_repeat():
    blocks = [{"block_id": "b1", "text": "Clause.", "type": "paragraph"}]
    llm_json = _base_llm_json()

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
        patch("app.services.drafting.compliance.analysis_agent._build_analysis_llm") as mock_llm,
        patch("app.services.drafting.compliance.analysis_agent.store_last_rubric_result"),
    ):
        mock_llm.return_value.stream.return_value = [MagicMock(content=llm_json)]
        agent = ComplianceAnalysisAgent()
        r1 = agent.analyze_document(document_blocks=blocks, language="en")
        r2 = agent.analyze_document(document_blocks=blocks, language="en")

    assert r1.risk_score == r2.risk_score
    assert r1.score_breakdown["rubric"] == r2.score_breakdown["rubric"]
    assert r1.risk_score == 0.0


def test_cache_hit_returns_identical_payload():
    from fastapi.testclient import TestClient

    from app.main import app

    sample_blocks = [{"block_id": "b1", "text": "Sample.", "type": "paragraph", "doc_id": "doc-1"}]
    content_hash = compute_document_content_hash(sample_blocks)
    cache_key = compliance_cache_key(
        content_hash=content_hash,
        check_level="quick",
        language="en",
        rubric_version=RUBRIC_VERSION,
    )
    cached_payload = ComplianceAnalysisResponse(
        document_type="Contract",
        overall_risk_level="LOW",
        risk_score=0.0,
        compliance_score=100.0,
        summary="Cached",
        clauses=[],
        issues_by_block_id={},
        recommendations=[],
        critical_issues=[],
        missing_clauses=[],
        citations=[],
        score_breakdown={"rubric": {"risk_score": 0.0}},
    ).model_dump(mode="json")

    with (
        patch(
            "app.api.v1.endpoints.drafting.compliance.get_document_blocks_by_doc_id"
        ) as mock_get,
        patch(
            "app.api.v1.endpoints.drafting.compliance.get_cached_compliance",
            return_value=cached_payload,
        ) as mock_cache_get,
        patch("app.api.v1.endpoints.drafting.compliance.ComplianceAnalysisAgent") as MockAgent,
    ):
        mock_get.return_value = sample_blocks
        client = TestClient(app)
        r = client.post(
            "/drafting/compliance/analyze",
            json={"doc_id": "doc-1", "language": "en"},
        )

    assert r.status_code == 200
    mock_cache_get.assert_called_once_with(cache_key)
    MockAgent.assert_not_called()
    assert r.json()["summary"] == "Cached"
    assert r.json()["risk_score"] == 0.0


def test_analyze_document_stores_last_rubric_with_doc_id():
    blocks = [{"block_id": "b1", "text": "Clause.", "type": "paragraph"}]
    llm_json = _base_llm_json()

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
        patch("app.services.drafting.compliance.analysis_agent._build_analysis_llm") as mock_llm,
        patch(
            "app.services.drafting.compliance.analysis_agent.store_last_rubric_result"
        ) as mock_store,
    ):
        mock_llm.return_value.stream.return_value = [MagicMock(content=llm_json)]
        agent = ComplianceAnalysisAgent()
        agent.analyze_document(document_blocks=blocks, language="en", doc_id="doc-xyz")

    mock_store.assert_called_once()
    call_kw = mock_store.call_args[1]
    assert call_kw["content_hash"] == compute_document_content_hash(blocks)
    assert len(call_kw["checks"]) >= 1
