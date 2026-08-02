"""Shared LLM factory for chat stream endpoints.

Provides a validated model allowlist and a cached ChatOpenAI builder keyed by
(model, enable_reasoning) so each unique combination is only instantiated once.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_openai import ChatOpenAI

from app.config import settings

SUPPORTED_MODELS = frozenset(
    [
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
        "anthropic/claude-sonnet-4",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
    ]
)

# Models where reasoning is mandatory — OpenRouter rejects effort=none for these.
# Omitting the reasoning field entirely lets them use their built-in default.
REASONING_REQUIRED_MODELS = frozenset(
    [
        "google/gemini-2.5-pro",
        "anthropic/claude-sonnet-4",
    ]
)

# Type alias used in request bodies for Pydantic Literal validation
SupportedModel = Literal[
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
]

_DEFAULT_MODEL = "google/gemini-2.5-flash"


def resolve_model(requested: str | None) -> str:
    """Return a supported model string.

    Uses ``requested`` when provided and in the allowlist, otherwise falls back
    to ``settings.GEMINI_MODEL`` and finally to the hard-coded default.
    """
    if requested and requested in SUPPORTED_MODELS:
        return requested
    env_default = settings.GEMINI_MODEL
    if env_default in SUPPORTED_MODELS:
        return env_default
    return _DEFAULT_MODEL


def reasoning_is_required(model: str) -> bool:
    """True if OpenRouter requires reasoning to always be on for this model."""
    return model in REASONING_REQUIRED_MODELS


@lru_cache(maxsize=10)
def build_chat_llm(
    *,
    model: str,
    enable_reasoning: bool,
    streaming: bool = True,
) -> ChatOpenAI:
    """Build (or return cached) a ChatOpenAI client for OpenRouter.

    Cached on (model, enable_reasoning, streaming) so callers never create
    duplicate clients for the same configuration.

    For models where reasoning is mandatory (gemini-2.5-pro, claude-sonnet-4),
    we omit the reasoning field entirely when enable_reasoning=False so OpenRouter
    doesn't reject the request with a 400. When enable_reasoning=True we send
    effort=medium for all models.
    """
    kwargs: dict = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": settings.OPENROUTER_API_KEY,
        "model": model,
        "temperature": 0.1,
        "streaming": streaming,
    }
    if enable_reasoning:
        kwargs["extra_body"] = {"reasoning": {"effort": "medium"}}
    elif not reasoning_is_required(model):
        # Explicitly disable reasoning only for models that support toggling it.
        kwargs["extra_body"] = {"reasoning": {"effort": "none"}}
    # For reasoning-required models with enable_reasoning=False, omit the field
    # entirely so OpenRouter uses the model's built-in default reasoning behaviour.
    return ChatOpenAI(**kwargs)
