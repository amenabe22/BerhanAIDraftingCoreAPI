"""
Unit tests for graph/main utilities that still exist after the streaming refactor.

- _apply_language: system-prompt language prefixing (main.py)
- _parse_citations: citation extraction from ToolMessages (main.py)
"""

from langchain_core.messages import ToolMessage

from app.main import _apply_language

# ---------------------------------------------------------------------------
# _apply_language
# ---------------------------------------------------------------------------


def test_apply_language_none_uses_auto_detect():
    result = _apply_language("BASE PROMPT", None)
    assert result.startswith("LANGUAGE: Detect")
    assert "BASE PROMPT" in result


def test_apply_language_amharic():
    result = _apply_language("BASE", "am")
    assert "Amharic" in result
    assert "BASE" in result


def test_apply_language_english():
    result = _apply_language("BASE", "en")
    assert "English" in result


def test_apply_language_oromo():
    result = _apply_language("BASE", "om")
    assert "Oromoo" in result


def test_apply_language_instruction_is_before_prompt():
    result = _apply_language("BASE PROMPT", None)
    lang_pos = result.index("LANGUAGE:")
    prompt_pos = result.index("BASE PROMPT")
    assert lang_pos < prompt_pos


# ---------------------------------------------------------------------------
# _parse_citations (via main.py)
# ---------------------------------------------------------------------------


from app.main import _parse_citations  # noqa: E402

CITATION_BLOCK = (
    "[Source: english-civil-code-1960 | Article 1675 | Art. 1675. Contract defined.]\n"
    "A contract is an agreement whereby two or more persons create, vary or extinguish obligations."
)

DOC_BLOCK = (
    "[Doc: doc-abc | Block: b01 | Type: paragraph]\n"
    "The tenant agrees to pay rent on the first of each month."
)


def test_parse_citations_extracts_source_block():
    tm = ToolMessage(content=CITATION_BLOCK, tool_call_id="t1")
    citations = _parse_citations([tm])
    assert len(citations) == 1
    assert citations[0]["document_id"] == "english-civil-code-1960"
    assert citations[0]["item_id"] == "1675"
    assert "Contract defined" in citations[0]["title"]


def test_parse_citations_extracts_doc_block():
    tm = ToolMessage(content=DOC_BLOCK, tool_call_id="t2")
    citations = _parse_citations([tm])
    assert len(citations) == 1
    assert citations[0]["doc_id"] == "doc-abc"
    assert citations[0]["block_id"] == "b01"
    assert citations[0]["type"] == "paragraph"


def test_parse_citations_deduplicates():
    tm1 = ToolMessage(content=CITATION_BLOCK, tool_call_id="t1")
    tm2 = ToolMessage(content=CITATION_BLOCK, tool_call_id="t2")
    citations = _parse_citations([tm1, tm2])
    assert len(citations) == 1


def test_parse_citations_empty_input():
    assert _parse_citations([]) == []


def test_parse_citations_non_citation_content_ignored():
    tm = ToolMessage(content="This is a plain answer with no source blocks.", tool_call_id="t1")
    assert _parse_citations([tm]) == []
