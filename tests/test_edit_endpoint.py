"""Tests for POST /drafting/edit endpoint."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import make_mock_edit_ops, make_mock_locate_result, make_mock_verify_result
from tests.fixtures.tiptap_docs import make_en_doc


def test_edit_request_validation_missing_instruction():
    client = TestClient(app)
    doc = make_en_doc()
    r = client.post("/drafting/edit", json={"doc_json": doc})
    assert r.status_code == 422


def test_edit_request_validation_missing_doc():
    client = TestClient(app)
    r = client.post("/drafting/edit", json={"instruction": "Change something"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_edit_endpoint_returns_contract(client, en_doc):
    mock_result = {
        "document": en_doc,
        "diff": "No changes",
        "operations": [],
        "metrics": {},
        "operation_id": "abc12345",
        "no_effective_change": True,
        "no_effective_change_reason": "test",
    }
    with patch(
        "app.api.v1.endpoints.drafting.edit._get_agent"
    ) as mock_get_agent:
        agent = AsyncMock()
        agent.edit_document = AsyncMock(return_value=mock_result)
        mock_get_agent.return_value = agent

        r = await client.post(
            "/drafting/edit",
            json={"doc_json": en_doc, "instruction": "test edit", "document_language": "en"},
        )

    assert r.status_code == 200
    data = r.json()
    assert data["operation_id"] == "abc12345"
    assert data["no_effective_change"] is True
    assert "document" in data
    assert "operations" in data


@pytest.mark.asyncio
async def test_edit_endpoint_success_with_mocked_agent(client, en_doc):
    blocks = [{"block_id": "b2", "type": "paragraph", "text": "updated"}]

    async def fake_edit(**kwargs):
        from app.services.drafting.editing.ops import apply_operations

        ops = make_mock_edit_ops("b2", "Updated text.")
        patched = apply_operations(kwargs["doc_json"], ops["operations"])
        return {
            "document": patched,
            "diff": "Updated b2",
            "operations": ops["operations"],
            "metrics": {},
            "operation_id": "test1234",
            "no_effective_change": False,
            "no_effective_change_reason": None,
        }

    with patch("app.api.v1.endpoints.drafting.edit._get_agent") as mock_get_agent:
        agent = AsyncMock()
        agent.edit_document = fake_edit
        mock_get_agent.return_value = agent

        r = await client.post(
            "/drafting/edit",
            json={"doc_json": en_doc, "instruction": "Change the weekly working hours to thirty-five"},
        )

    assert r.status_code == 200
    data = r.json()
    assert data["no_effective_change"] is False
    assert len(data["operations"]) == 1
