"""LLM client for semantic edit agent (OpenRouter)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


def _llm(*, temperature: float = 0.0, json_mode: bool = False) -> ChatOpenAI:
    model = settings.EDIT_MODEL or settings.GEMINI_MODEL
    kwargs: dict[str, Any] = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": settings.OPENROUTER_API_KEY,
        "model": model,
        "temperature": temperature,
        "streaming": False,
    }
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(**kwargs)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


async def complete_json(system: str, user: str) -> dict[str, Any] | None:
    llm = _llm(json_mode=True)
    response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = response.content if isinstance(response.content, str) else str(response.content)
    return _extract_json(content)


async def complete_text(system: str, user: str) -> str:
    llm = _llm(json_mode=False)
    response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content if isinstance(response.content, str) else str(response.content)
