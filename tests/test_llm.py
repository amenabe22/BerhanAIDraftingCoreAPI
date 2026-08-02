"""
Unit tests for app/llm.py:
  - resolve_model defaults, env fallback, explicit selection
  - build_chat_llm caching and kwarg forwarding
  - SupportedModel Literal enforcement via ChatRequest
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.llm import SUPPORTED_MODELS, build_chat_llm, resolve_model
from tests.conftest import _make_ai_token_chunk, make_mock_graph


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------


def test_resolve_model_returns_explicit_supported_model():
    assert resolve_model("openai/gpt-4.1") == "openai/gpt-4.1"


def test_resolve_model_none_falls_back_to_env_when_in_allowlist():
    with patch("app.llm.settings") as mock_settings:
        mock_settings.GEMINI_MODEL = "google/gemini-2.5-pro"
        result = resolve_model(None)
    assert result == "google/gemini-2.5-pro"


def test_resolve_model_falls_back_to_hard_default_when_env_not_in_allowlist():
    with patch("app.llm.settings") as mock_settings:
        mock_settings.GEMINI_MODEL = "some/unknown-model"
        result = resolve_model(None)
    assert result == "google/gemini-2.5-flash"


def test_resolve_model_unsupported_explicit_falls_back_to_env():
    """When an unsupported model string is passed it falls back to env default."""
    with patch("app.llm.settings") as mock_settings:
        mock_settings.GEMINI_MODEL = "openai/gpt-4.1-mini"
        result = resolve_model("some/unknown-model")
    assert result == "openai/gpt-4.1-mini"


def test_all_supported_models_resolve_to_themselves():
    for m in SUPPORTED_MODELS:
        assert resolve_model(m) == m


# ---------------------------------------------------------------------------
# build_chat_llm — caching
# ---------------------------------------------------------------------------


def test_build_chat_llm_same_key_returns_same_instance():
    build_chat_llm.cache_clear()
    with patch("app.llm.ChatOpenAI") as MockLLM:
        MockLLM.return_value = MagicMock()
        a = build_chat_llm(model="google/gemini-2.5-flash", enable_reasoning=False)
        b = build_chat_llm(model="google/gemini-2.5-flash", enable_reasoning=False)
        assert a is b
    build_chat_llm.cache_clear()


def test_build_chat_llm_different_model_different_instance():
    build_chat_llm.cache_clear()
    with patch("app.llm.ChatOpenAI") as MockLLM:
        MockLLM.side_effect = lambda **kw: MagicMock()
        a = build_chat_llm(model="google/gemini-2.5-flash", enable_reasoning=False)
        b = build_chat_llm(model="openai/gpt-4.1", enable_reasoning=False)
        assert a is not b
    build_chat_llm.cache_clear()


def test_build_chat_llm_different_reasoning_different_instance():
    build_chat_llm.cache_clear()
    with patch("app.llm.ChatOpenAI") as MockLLM:
        MockLLM.side_effect = lambda **kw: MagicMock()
        a = build_chat_llm(model="google/gemini-2.5-flash", enable_reasoning=False)
        b = build_chat_llm(model="google/gemini-2.5-flash", enable_reasoning=True)
        assert a is not b
    build_chat_llm.cache_clear()


def test_build_chat_llm_reasoning_off_passes_effort_none():
    build_chat_llm.cache_clear()
    with patch("app.llm.ChatOpenAI") as MockLLM:
        MockLLM.return_value = MagicMock()
        build_chat_llm(model="google/gemini-2.5-flash", enable_reasoning=False)
        _, kwargs = MockLLM.call_args
        assert kwargs["extra_body"] == {"reasoning": {"effort": "none"}}
    build_chat_llm.cache_clear()


def test_build_chat_llm_reasoning_on_passes_effort_medium():
    build_chat_llm.cache_clear()
    with patch("app.llm.ChatOpenAI") as MockLLM:
        MockLLM.return_value = MagicMock()
        build_chat_llm(model="google/gemini-2.5-flash", enable_reasoning=True)
        _, kwargs = MockLLM.call_args
        assert kwargs["extra_body"] == {"reasoning": {"effort": "medium"}}
    build_chat_llm.cache_clear()


# ---------------------------------------------------------------------------
# ChatRequest Literal validation → 422
# ---------------------------------------------------------------------------


ENDPOINTS = ["/legal-search/stream", "/legal-agent/stream"]


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_unsupported_model_returns_422(endpoint):
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            endpoint, json={"message": "hi", "model": "some/unsupported-model"}
        )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# model SSE event emitted on valid requests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_model_event_emitted_with_default(endpoint):
    import contextlib
    import json as _json

    from app.llm import resolve_model
    from app.main import app

    graph = make_mock_graph([_make_ai_token_chunk("Hi")])
    with patch("app.main.get_graph", return_value=graph):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(endpoint, json={"message": "hi"})

    events = []
    for line in r.text.splitlines():
        if line.startswith("data: "):
            with contextlib.suppress(_json.JSONDecodeError):
                events.append(_json.loads(line[6:]))

    model_events = [e for e in events if e.get("type") == "model"]
    assert len(model_events) == 1
    assert model_events[0]["model"] == resolve_model(None)
    assert model_events[0]["enable_reasoning"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_model_event_reflects_requested_model(endpoint):
    import contextlib
    import json as _json

    from app.main import app

    graph = make_mock_graph([_make_ai_token_chunk("Hi")])
    with patch("app.main.get_graph", return_value=graph):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                endpoint,
                json={"message": "hi", "model": "openai/gpt-4.1-mini", "enable_reasoning": True},
            )

    events = []
    for line in r.text.splitlines():
        if line.startswith("data: "):
            with contextlib.suppress(_json.JSONDecodeError):
                events.append(_json.loads(line[6:]))

    model_events = [e for e in events if e.get("type") == "model"]
    assert len(model_events) == 1
    assert model_events[0]["model"] == "openai/gpt-4.1-mini"
    assert model_events[0]["enable_reasoning"] is True
