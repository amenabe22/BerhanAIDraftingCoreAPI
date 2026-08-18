"""Unit tests for document generation port (no live LLM)."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.generation.prompt_builder import PromptBuilder
from app.services.generation.sse import SSEEmitter, format_sse
from app.services.generation.thread_store import ThreadStore


@pytest.fixture
def client():
    return TestClient(app)


def test_generate_route_in_openapi(client: TestClient):
    paths = client.app.openapi()["paths"]
    assert "/drafting/generate/stream" in paths
    assert "/legal-agent/stream" in paths
    assert "/drafting/compliance/analyze-stream" in paths


def test_generate_requires_message(client: TestClient):
    r = client.post("/drafting/generate/stream", json={})
    assert r.status_code == 422


def test_generate_stream_returns_sse_without_credentials(client: TestClient):
    r = client.post(
        "/drafting/generate/stream",
        json={
            "message": "Draft an NDA. Keep it simple and just generate it.",
            "language": "en",
            "action": "finalize",
            "num_pages": 1,
        },
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    assert "thread_id" in r.text
    assert "model" in r.text
    # Without OPENROUTER_API_KEY we expect a sanitized auth error event
    assert "error" in r.text


def test_thread_store_multi_turn():
    store = ThreadStore()
    state = store.create()
    store.add_user_message(state.thread_id, "hello")
    store.add_system_response(state.thread_id, "hi")
    store.update_requirements(state.thread_id, {"language": "om"})
    again = store.get(state.thread_id)
    assert again is not None
    assert again.extracted_requirements["language"] == "om"
    hist = store.history(again)
    assert len(hist) == 2
    assert hist[0]["role"] == "user"


def test_prompt_builder_oromo():
    pb = PromptBuilder()
    analysis = pb.build_analysis_prompt("NDA please", [], {"language": "om"})
    assert "Afaan Oromo" in analysis
    gen = pb.build_generation_prompt(
        {"language": "om", "document_type": "NDA", "num_pages": 2},
        [],
        "",
    )
    assert "Afaan Oromo" in gen
    assert 'MUST be "om"' in gen


def test_openapi_supports_oromo_and_models(client: TestClient):
    spec = client.app.openapi()
    lang = spec["components"]["schemas"]["Language"]["enum"]
    assert set(lang) == {"am", "en", "om"}
    models = spec["components"]["schemas"]["GenerateRequest"]["properties"]["model"]
    blob = json.dumps(models)
    for m in (
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
        "anthropic/claude-sonnet-4",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
    ):
        assert m in blob
    edit_lang = spec["components"]["schemas"]["EditRequest"]["properties"]["document_language"]
    assert "om" in json.dumps(edit_lang)


@pytest.mark.asyncio
async def test_sse_emitter_queue():
    q: asyncio.Queue = asyncio.Queue()
    emitter = SSEEmitter(q)
    await emitter.status("working")
    await emitter.close()
    ev = await q.get()
    assert ev == {"type": "status", "message": "working"}
    assert await q.get() is None
    assert format_sse({"type": "token", "content": "x"}).startswith("data: ")
