"""Cassation (ሰበር) tools: retrieval shaping, citations, grounding and sanitizing."""

from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage

from app.main import _parse_citations, _select_citations
from app.retrieval_cassation import (
    _diversify_by_case,
    format_cassation_block,
    get_cassation_case,
)
from app.services.agent_response_sanitize import sanitize_user_facing_agent_text
from app.services.legal.grounding import extract_retrieved_item_ids, validate_answer


def _doc(case: str, section: str = "body", score: float = 0.9, idx: int = 0, text: str = "") -> Document:
    return Document(
        page_content=text or f"passage for {case} {section} {idx} " * 3,
        metadata={
            "document_id": f"cassation-published-25-{case}",
            "case_number": case,
            "title": f"ሰበር {case}",
            "section": section,
            "chunk_index": str(idx),
            "_rerank_score": score,
        },
    )


def test_diversify_caps_passages_per_case_and_case_count():
    ranked = [_doc("1", idx=i) for i in range(5)] + [_doc(str(n)) for n in range(2, 9)]
    kept = _diversify_by_case(ranked, max_cases=3, per_case=2, min_score=0.05)
    cases = [d.metadata["case_number"] for d in kept]
    assert cases == ["1", "1", "2", "3"]


def test_diversify_drops_low_rerank_scores():
    kept = _diversify_by_case(
        [_doc("1", score=0.5), _doc("2", score=0.01)], max_cases=5, per_case=2, min_score=0.05
    )
    assert [d.metadata["case_number"] for d in kept] == ["1"]


def test_get_cassation_case_orders_summary_then_focused_body():
    docs = [
        _doc("186626", "body", idx=40),
        _doc("186626", "holding"),
        _doc("186626", "body", idx=3),
        _doc("186626", "headnote"),
    ]
    with (
        patch("app.retrieval_cassation._scroll_case", return_value=docs) as scroll,
        patch("app.retrieval_cassation._rerank", side_effect=lambda q, d, n: d[:n]),
    ):
        out = get_cassation_case("ሰበር 186626", focus="guardian sale")
    scroll.assert_called_once_with("186626")
    assert [d.metadata["section"] for d in out] == ["headnote", "holding", "body", "body"]
    assert [d.metadata["chunk_index"] for d in out[2:]] == ["3", "40"]


def test_get_cassation_case_rejects_non_numeric_id():
    assert get_cassation_case("unknown") == []


def test_cassation_block_parses_into_citation():
    block = format_cassation_block(_doc("186626", "holding", text="ውሳኔ " * 20))
    cites = _parse_citations([ToolMessage(content=block, tool_call_id="t1")])
    assert cites[0]["doc_type"] == "cassation"
    assert cites[0]["case_number"] == "186626"
    assert cites[0]["document_id"] == "cassation-published-25-186626"


def test_select_citations_keeps_sources_the_answer_cites():
    cites = [
        {"doc_type": "statute", "document_id": "civil", "item_id": "1678"},
        {"doc_type": "cassation", "document_id": "c-1", "case_number": "186626"},
        {"doc_type": "cassation", "document_id": "c-2", "case_number": "43424"},
    ]
    picked = _select_citations(cites, "በሰበር መዝገብ ቁጥር 186626 ፍርድ ቤቱ …")
    assert [c["document_id"] for c in picked] == ["c-1"]


def test_select_citations_falls_back_to_retrieval_order():
    cites = [{"doc_type": "cassation", "document_id": "c-1", "case_number": "186626"}]
    assert _select_citations(cites, "no numbers here") == cites


def test_grounding_accepts_cassation_pinpoint():
    tm = ToolMessage(content=format_cassation_block(_doc("186626")), tool_call_id="t1")
    answer = "በኢትዮጵያ ሕግ ሞግዚት የልጁን ቤት ያለ ፍርድ ቤት ፈቃድ መሸጥ አይችልም (ሰበር መዝገብ ቁጥር 186626)።"
    assert validate_answer(answer, [tm]).ok
    assert "186626" in extract_retrieved_item_ids([tm])


def test_sanitizer_strips_echoed_headers_and_slugs():
    text = "See [Cassation: cassation-published-25-186626 | Case 186626 | t] and (cassation-published-25-186626)."
    cleaned = sanitize_user_facing_agent_text(text)
    assert "[Cassation:" not in cleaned
    assert "cassation-published" not in cleaned
