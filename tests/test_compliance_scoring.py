"""Tests for rubric-based compliance scoring."""

from app.services.drafting.compliance.rubric import RUBRIC_ITEMS, get_rubric_items_for_document_type
from app.services.drafting.compliance.scoring import compute_rubric_score


def _all_present_checks(document_type: str = "Contract") -> list[dict]:
    items = get_rubric_items_for_document_type(document_type)
    return [
        {"id": item["id"], "status": "PRESENT", "block_id": None, "rationale": "OK"}
        for item in items
    ]


def test_compute_rubric_score_all_present():
    checks = _all_present_checks()
    result = compute_rubric_score(checks)
    assert result["raw_penalty"] == 0
    assert result["risk_score"] == 0.0
    assert result["compliance_score"] == 100.0
    assert result["overall_risk_level"] == "LOW"
    assert len(result["item_breakdown"]) == len(checks)


def test_compute_rubric_score_non_compliant_item():
    checks = _all_present_checks()
    checks[0] = {
        "id": checks[0]["id"],
        "status": "NON_COMPLIANT",
        "block_id": "b1",
        "rationale": "Foreign governing law referenced.",
    }
    result = compute_rubric_score(checks)
    assert result["raw_penalty"] == 12
    assert result["status_counts"]["NON_COMPLIANT"] == 1
    assert result["risk_score"] == round(12 / 150 * 100, 2)


def test_compute_rubric_score_missing_and_partial():
    checks = [
        {"id": "jurisdiction_governing_law", "status": "MISSING", "block_id": None},
        {"id": "contract_type_parties", "status": "PARTIAL", "block_id": "b1"},
        {"id": "parties_obligations", "status": "PRESENT", "block_id": "b2"},
    ]
    result = compute_rubric_score(checks)
    # MISSING(8) + PARTIAL(4) + PRESENT(0) = 12
    assert result["raw_penalty"] == 12
    assert result["overall_risk_level"] == "LOW"


def test_compute_rubric_score_not_applicable_zero_penalty():
    checks = [{"id": "confidentiality", "status": "NOT_APPLICABLE", "block_id": None}]
    result = compute_rubric_score(checks)
    assert result["raw_penalty"] == 0


def test_compute_rubric_score_threshold_critical():
    checks = [
        {"id": f"item_{i}", "status": "NON_COMPLIANT", "block_id": None}
        for i in range(10)
    ]
    result = compute_rubric_score(checks, max_penalty=150)
    assert result["risk_score"] >= 70
    assert result["overall_risk_level"] == "CRITICAL"


def test_compute_rubric_score_unknown_status_treated_as_missing():
    checks = [{"id": "x", "status": "BOGUS", "block_id": None}]
    result = compute_rubric_score(checks)
    assert result["status_counts"]["MISSING"] == 1
    assert result["raw_penalty"] == 8


def test_rubric_items_cover_checklist_areas():
    ids = {item["id"] for item in RUBRIC_ITEMS}
    assert "jurisdiction_governing_law" in ids
    assert "dispute_resolution" in ids
    assert "mandatory_ethiopian_law" in ids
