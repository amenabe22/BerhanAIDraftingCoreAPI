"""Tests for SemanticEditAgent pipeline (mocked LLM + retrieval)."""

from unittest.mock import AsyncMock, patch

import pytest

from app.services.drafting.editing.agent import SemanticEditAgent
from app.services.drafting.editing.tiptap import extract_blocks_from_tiptap
from tests.conftest import (
    make_mock_edit_ops,
    make_mock_locate_result,
    make_mock_verify_result,
)


CONTRACT_KEYS = {
    "document",
    "diff",
    "operations",
    "metrics",
    "operation_id",
    "no_effective_change",
    "no_effective_change_reason",
}


@pytest.mark.asyncio
async def test_global_replace_fast_path_no_llm(en_doc):
    agent = SemanticEditAgent()
    with patch("app.services.drafting.editing.agent.edit_llm.complete_json") as mock_json:
        with patch(
            "app.services.drafting.editing.agent.edit_llm.complete_text",
            new=AsyncMock(return_value="Replaced forty with thirty-five."),
        ):
            result = await agent.edit_document(
                en_doc,
                instruction="Change forty to thirty-five everywhere",
                document_language="en",
            )
        mock_json.assert_not_called()

    assert set(result.keys()) >= CONTRACT_KEYS
    assert result["no_effective_change"] is False
    blocks = extract_blocks_from_tiptap(result["document"])
    b2 = next(b for b in blocks if b["block_id"] == "b2")
    assert "thirty-five" in b2["text"]


@pytest.mark.asyncio
async def test_locate_low_confidence_no_effective_change(en_doc):
    agent = SemanticEditAgent()
    with (
        patch(
            "app.services.drafting.editing.agent.rank_blocks_for_instruction",
            return_value=extract_blocks_from_tiptap(en_doc),
        ),
        patch(
            "app.services.drafting.editing.agent.edit_llm.complete_json",
            new=AsyncMock(
                return_value={"targets": [], "scope": "single", "confidence": 0.2}
            ),
        ),
    ):
        result = await agent.edit_document(en_doc, instruction="Make it better")

    assert result["no_effective_change"] is True
    assert result["document"] == en_doc or extract_blocks_from_tiptap(result["document"]) == extract_blocks_from_tiptap(en_doc)


@pytest.mark.asyncio
async def test_happy_path_edit_apply_verify(en_doc):
    agent = SemanticEditAgent()
    blocks = extract_blocks_from_tiptap(en_doc)

    async def fake_complete_json(system: str, user: str):
        if "block selector" in system.lower() or "selector" in system.lower():
            return make_mock_locate_result("b2")
        if "verify" in system.lower():
            return make_mock_verify_result(True)
        return make_mock_edit_ops("b2", "The Employee shall work thirty-five hours per week.")

    with (
        patch(
            "app.services.drafting.editing.agent.rank_blocks_for_instruction",
            return_value=blocks,
        ),
        patch(
            "app.services.drafting.editing.agent.edit_llm.complete_json",
            side_effect=fake_complete_json,
        ),
        patch(
            "app.services.drafting.editing.agent.edit_llm.complete_text",
            new=AsyncMock(return_value="Updated working hours clause."),
        ),
    ):
        result = await agent.edit_document(
            en_doc,
            instruction="Change working hours to thirty-five",
            document_language="en",
        )

    assert result["no_effective_change"] is False
    assert result["document"]["type"] == "doc"
    assert len(result["operations"]) >= 1
    b2 = next(b for b in extract_blocks_from_tiptap(result["document"]) if b["block_id"] == "b2")
    assert "thirty-five" in b2["text"]
    assert len(extract_blocks_from_tiptap(result["document"])) == 3


@pytest.mark.asyncio
async def test_verify_revise_loop(en_doc):
    agent = SemanticEditAgent()
    blocks = extract_blocks_from_tiptap(en_doc)
    edit_calls = {"n": 0}

    verify_calls = {"n": 0}

    async def fake_complete_json(system: str, user: str):
        if "selector" in system.lower():
            return make_mock_locate_result("b2")
        if "verify" in system.lower():
            verify_calls["n"] += 1
            return make_mock_verify_result(verify_calls["n"] > 1, "Fix the hours text")
        edit_calls["n"] += 1
        return make_mock_edit_ops("b2", f"Attempt {edit_calls['n']} thirty-five hours.")

    with (
        patch(
            "app.services.drafting.editing.agent.rank_blocks_for_instruction",
            return_value=blocks,
        ),
        patch(
            "app.services.drafting.editing.agent.edit_llm.complete_json",
            side_effect=fake_complete_json,
        ),
        patch(
            "app.services.drafting.editing.agent.edit_llm.complete_text",
            new=AsyncMock(return_value="Revised clause."),
        ),
        patch("app.config.settings.EDIT_MAX_REVISIONS", 2),
    ):
        result = await agent.edit_document(en_doc, instruction="Update hours", document_language="en")

    assert edit_calls["n"] >= 2
    assert result["no_effective_change"] is False


@pytest.mark.asyncio
async def test_invalid_ops_safe_no_change(en_doc):
    agent = SemanticEditAgent()
    blocks = extract_blocks_from_tiptap(en_doc)

    async def fake_complete_json(system: str, user: str):
        if "selector" in system.lower():
            return make_mock_locate_result("b2")
        return {"operations": [{"op_id": "x", "type": "replace", "payload": {}}]}

    with (
        patch(
            "app.services.drafting.editing.agent.rank_blocks_for_instruction",
            return_value=blocks,
        ),
        patch(
            "app.services.drafting.editing.agent.edit_llm.complete_json",
            side_effect=fake_complete_json,
        ),
    ):
        result = await agent.edit_document(en_doc, instruction="Update clause")

    assert result["no_effective_change"] is True
    assert len(extract_blocks_from_tiptap(result["document"])) == 3
