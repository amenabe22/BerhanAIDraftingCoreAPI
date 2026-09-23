"""Tests for section expand and hybrid edit tool helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.drafting.editing.retrieval import expand_section_blocks
from app.services.drafting.editing.tools import (
    _apply_document_edit_impl,
    create_apply_document_edit_tool,
)


def test_expand_section_blocks_includes_body_until_next_heading():
    blocks = [
        {"block_id": "h1", "type": "heading", "attrs": {"level": 2}, "text": "1. DEFINITIONS"},
        {"block_id": "p1", "type": "paragraph", "attrs": {}, "text": "Confidential Information means…"},
        {"block_id": "p2", "type": "paragraph", "attrs": {}, "text": "Party means…"},
        {"block_id": "h2", "type": "heading", "attrs": {"level": 2}, "text": "2. OBLIGATIONS"},
        {"block_id": "p3", "type": "paragraph", "attrs": {}, "text": "Each party shall…"},
    ]
    expanded = expand_section_blocks(blocks, {"h1"})
    assert expanded == {"h1", "p1", "p2"}
    assert "p3" not in expanded
    assert "h2" not in expanded


def test_expand_section_blocks_keeps_non_heading_seeds():
    blocks = [
        {"block_id": "p1", "type": "paragraph", "attrs": {}, "text": "Only this"},
        {"block_id": "p2", "type": "paragraph", "attrs": {}, "text": "Other"},
    ]
    assert expand_section_blocks(blocks, {"p1"}) == {"p1"}


def test_create_apply_document_edit_tool_name():
    tool = create_apply_document_edit_tool()
    assert tool.name == "apply_document_edit"


def test_apply_document_edit_impl_without_doc_json():
    with patch(
        "app.services.drafting.editing.tools.get_config",
        return_value={"configurable": {}},
    ):
        raw = _apply_document_edit_impl("shorten definitions")
    import json

    data = json.loads(raw)
    assert data["ok"] is False
    assert data["no_effective_change"] is True


@pytest.mark.asyncio
async def test_section_expand_used_in_edit_agent(en_doc):
    """Locate on a heading with section scope expands targets beyond a single block."""
    from app.services.drafting.editing.agent import SemanticEditAgent
    from app.services.drafting.editing.tiptap import extract_blocks_from_tiptap

    # Build a mini definitions-style doc
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "heading",
                "attrs": {"level": 2, "block_id": "def-h"},
                "content": [{"type": "text", "text": "1. DEFINITIONS"}],
            },
            {
                "type": "paragraph",
                "attrs": {"block_id": "def-1"},
                "content": [{"type": "text", "text": "Agreement means this agreement."}],
            },
            {
                "type": "paragraph",
                "attrs": {"block_id": "def-2"},
                "content": [{"type": "text", "text": "Party means each signatory."}],
            },
            {
                "type": "heading",
                "attrs": {"level": 2, "block_id": "obl-h"},
                "content": [{"type": "text", "text": "2. OBLIGATIONS"}],
            },
        ],
    }

    locate = {
        "targets": [{"block_id": "def-h", "action": "rewrite", "confidence": 0.9}],
        "scope": "section",
        "confidence": 0.9,
    }
    ops = {
        "operations": [
            {
                "op_id": "a1",
                "type": "replace",
                "block_id": "def-1",
                "payload": {"new_text": "Agreement means this contract."},
            },
            {
                "op_id": "a2",
                "type": "replace",
                "block_id": "def-2",
                "payload": {"new_text": "Party means a signatory."},
            },
        ]
    }
    verify = {"passed": True, "issues": [], "feedback": ""}

    agent = SemanticEditAgent()
    with patch(
        "app.services.drafting.editing.agent.rank_blocks_for_instruction",
        return_value=extract_blocks_from_tiptap(doc),
    ):
        with patch(
            "app.services.drafting.editing.agent.edit_llm.complete_json",
            new=AsyncMock(side_effect=[locate, ops, verify]),
        ):
            with patch(
                "app.services.drafting.editing.agent.edit_llm.complete_text",
                new=AsyncMock(return_value="Shortened definitions."),
            ):
                result = await agent.edit_document(
                    doc, instruction="shorten the DEFINITIONS part", document_language="en"
                )

    assert result.get("no_effective_change") is False
    assert len(result.get("operations") or []) >= 1
    assert result["metrics"]["stages"].get("section_expand")
