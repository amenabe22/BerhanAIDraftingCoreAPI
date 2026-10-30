"""Tests for deterministic monetary figure extraction from TipTap docs."""

from __future__ import annotations

from app.services.drafting.editing.financial_figures import (
    extract_financial_figures,
    format_financial_figures_for_tool,
)


def _para(block_id: str, text: str) -> dict:
    return {
        "type": "paragraph",
        "attrs": {"block_id": block_id},
        "content": [{"type": "text", "text": text}],
    }


def _heading(block_id: str, text: str, level: int = 2) -> dict:
    return {
        "type": "heading",
        "attrs": {"block_id": block_id, "level": level},
        "content": [{"type": "text", "text": text}],
    }


def test_extracts_etb_amount_and_ratio_with_section():
    doc = {
        "type": "doc",
        "content": [
            _heading("h1", "2. የአጋሮች አስተዋጽኦ"),
            _para("b1", "ገንዘብ: 500,000 ETB (አምስት መቶ ሺህ የኢትዮጵያ ብር)"),
            _heading("h2", "3. ትርፍ እና ኪሳራ መጋራት"),
            _para("b2", "ትርፍ እና ኪሳራ በአጋሮቹ መካከል 50/50 ይጋራል።"),
            _para("b3", "ፊርማ: _________________________"),
        ],
    }
    hits = extract_financial_figures(doc)
    amounts = {h["amount"] for h in hits}
    assert "500,000 ETB" in amounts
    assert "50/50" in amounts
    cash = next(h for h in hits if "500,000" in h["amount"])
    assert cash["block_id"] == "b1"
    assert "አስተዋጽኦ" in (cash.get("section") or "")
    ratio = next(h for h in hits if "50/50" in h["amount"])
    assert ratio["block_id"] == "b2"


def test_format_reports_none_when_empty():
    text = format_financial_figures_for_tool([])
    assert "No monetary amounts" in text


def test_format_includes_computed_half_target():
    hits = [
        {
            "block_id": "b1",
            "amount": "500,000 ETB",
            "section": "አስተዋጽኦ",
            "snippet": "ገንዘብ: 500,000 ETB",
        }
    ]
    text = format_financial_figures_for_tool(hits, factor=0.5)
    assert "250,000 ETB" in text
    assert "COMPUTED TARGET" in text
    assert "apply_document_edit NOW" in text


def test_skips_tiny_bare_integers():
    doc = {
        "type": "doc",
        "content": [
            _para("b1", "Clause 1 and Clause 2 of this agreement."),
        ],
    }
    hits = extract_financial_figures(doc)
    assert hits == []


def test_list_tool_reads_doc_json_from_config(monkeypatch):
    from app.services.drafting.editing import tools as edit_tools

    doc = {
        "type": "doc",
        "content": [
            _heading("h1", "Capital"),
            _para("b1", "Cash contribution: 500,000 ETB"),
        ],
    }

    monkeypatch.setattr(
        edit_tools,
        "get_config",
        lambda: {"configurable": {"doc_json": doc}},
    )
    out = edit_tools._list_document_financial_figures_impl()
    assert "500,000 ETB" in out
    assert "Found" in out
