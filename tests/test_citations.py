"""
Unit tests for _parse_citations() in app/main.py.

These tests cover the pure citation-extraction logic without touching
any HTTP layer or external services.
"""

from langchain_core.messages import ToolMessage

from app.main import _parse_citations


def _tm(content: str, tool_call_id: str = "tc1") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tool_call_id)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_single_citation_parsed_correctly():
    content = (
        "[Source: english-civil-code-1960 | Article 1675 | Art. 1675. Contract defined.]\n"
        "A contract is an agreement whereby two or more persons create obligations."
    )
    result = _parse_citations([_tm(content)])
    assert len(result) == 1
    c = result[0]
    assert c["document_id"] == "english-civil-code-1960"
    assert c["item_id"] == "1675"
    assert c["title"] == "Art. 1675. Contract defined."
    assert "contract" in c["content"].lower()


def test_multiple_citations_in_one_tool_message():
    content = (
        "[Source: doc-a | Article 1 | Title A]\nContent A.\n\n"
        "[Source: doc-b | Article 2 | Title B]\nContent B."
    )
    result = _parse_citations([_tm(content)])
    assert len(result) == 2
    assert result[0]["document_id"] == "doc-a"
    assert result[1]["document_id"] == "doc-b"


def test_citations_across_multiple_tool_messages():
    t1 = _tm("[Source: doc-a | Article 1 | Title A]\nContent A.", "tc1")
    t2 = _tm("[Source: doc-b | Article 2 | Title B]\nContent B.", "tc2")
    result = _parse_citations([t1, t2])
    assert len(result) == 2


def test_duplicate_citations_are_deduplicated():
    content = (
        "[Source: doc-a | Article 1 | Title A]\nContent A.\n\n"
        "[Source: doc-a | Article 1 | Title A]\nContent A again."
    )
    result = _parse_citations([_tm(content)])
    assert len(result) == 1


def test_content_truncated_to_300_chars():
    long_body = "x" * 500
    content = f"[Source: doc-a | Article 1 | Title A]\n{long_body}"
    result = _parse_citations([_tm(content)])
    assert len(result[0]["content"]) == 300


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_list_returns_empty():
    assert _parse_citations([]) == []


def test_tool_message_with_no_source_blocks():
    result = _parse_citations([_tm("No source blocks here.")])
    assert result == []


def test_non_string_content_skipped():
    tm = ToolMessage(content=None, tool_call_id="tc1")
    result = _parse_citations([tm])
    assert result == []


def test_malformed_header_missing_bracket_end():
    content = "[Source: doc-a | Article 1 | Title A\nContent without closing bracket."
    result = _parse_citations([_tm(content)])
    assert result == []


def test_missing_article_and_title_parts():
    """Header with only one pipe-separated part (just Source)."""
    content = "[Source: doc-only]\nSome content."
    result = _parse_citations([_tm(content)])
    assert len(result) == 1
    assert result[0]["document_id"] == "doc-only"
    assert result[0]["item_id"] == ""
    assert result[0]["title"] == ""


def test_amharic_content_parsed():
    """Non-ASCII content in citation body should parse without error."""
    content = "[Source: amharic-civil-code | Article 369 | ቊ 369]\nቊ 369 አንቀጽ ይዘት።"
    result = _parse_citations([_tm(content)])
    assert len(result) == 1
    assert result[0]["document_id"] == "amharic-civil-code"
