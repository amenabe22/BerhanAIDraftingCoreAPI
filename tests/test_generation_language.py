"""Tests for generation language preference pinning / Amharic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.drafting.generate import GenerateRequest, StructuredGenerateRequest
from app.services.generation.agent import GenerationAgent
from app.services.generation.language import normalize_language_code
from app.services.generation.prompt_builder import PromptBuilder
from app.services.generation.requirements import build_requirements, build_synthetic_prompt
from app.services.generation.sse import SSEEmitter
from app.services.generation.thread_store import ThreadStore


@pytest.fixture
def client():
    return TestClient(app)


async def _async_iter(items):
    for item in items:
        chunk = MagicMock()
        chunk.content = item
        yield chunk


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("am", "am"),
        ("amh", "am"),
        ("amharic", "am"),
        ("en", "en"),
        ("om", "om"),
        ("oro", "om"),
        ("oromo", "om"),
    ],
)
def test_normalize_language_aliases(raw, expected):
    assert normalize_language_code(raw) == expected


_META = {
    "title": "Mutual NDA",
    "parties": [
        {"name": "Acme PLC", "role": "Disclosing Party"},
        {"name": "Beta LLC", "role": "Receiving Party"},
    ],
    "governingLaw": "Ethiopian law",
}


def test_request_accepts_amh_alias():
    req = StructuredGenerateRequest(
        doc_type="NDA", type=["pdf"], language="amh", metadata=_META
    )
    assert req.language.value == "am"
    legacy = GenerateRequest(message="hi", language="amh")
    assert legacy.language.value == "am"


def test_prompt_builder_amharic_uses_example_and_final_reminder():
    pb = PromptBuilder()
    gen = pb.build_generation_prompt(
        {"language": "amh", "document_type": "NDA", "num_pages": 1},
        [],
        "",
    )
    assert "ENTIRE document in Amharic" in gen
    assert "FINAL LANGUAGE CHECK" in gen
    assert "አ" in gen  # Amharic structure example
    assert 'metadata.language MUST be "am"' in gen
    assert "OUTPUT LANGUAGE: Amharic" in gen
    assert "Write all clause text in Amharic" in gen


def test_structured_requirements_pin_amharic():
    req = StructuredGenerateRequest(
        doc_type="Employment Contract",
        type=["pdf"],
        language="am",
        instructions="Keep it short",
        metadata=_META,
    )
    reqs = build_requirements(req)
    assert reqs["language"] == "am"
    synthetic = build_synthetic_prompt(req, reqs)
    assert "Amharic" in synthetic
    assert 'MUST be "am"' in synthetic


@pytest.mark.asyncio
async def test_analyze_requirements_pins_preferred_language_over_llm():
    store = ThreadStore()
    state = store.create()
    agent = GenerationAgent(store=store)

    fake_llm = MagicMock()
    fake_llm.astream = MagicMock(
        return_value=_async_iter(
            [
                '{"ready_to_generate": true, "response_message": "ok", '
                '"questions": [], "document_type": "NDA", "language": "en", '
                '"extracted_info": {"language": "en", "parties": "A and B"}}'
            ]
        )
    )
    fake_llm.ainvoke = AsyncMock()

    queue: asyncio.Queue = asyncio.Queue()
    emitter = SSEEmitter(queue)

    with patch.object(agent, "_llm", return_value=(fake_llm, "model")):
        analysis = await agent.analyze_requirements(
            state.thread_id,
            "Draft an NDA between Acme and Beta",
            emitter,
            context={"language": "am"},
        )

    assert analysis["language"] == "am"
    assert analysis["extracted_info"]["language"] == "am"
    again = store.get(state.thread_id)
    assert again is not None
    assert again.extracted_requirements["language"] == "am"


def test_openapi_still_lists_generate(client: TestClient):
    paths = client.app.openapi()["paths"]
    assert "/drafting/generate" in paths
    assert "/drafting/generate/stream" in paths
