"""Sanitize generation errors for client-facing SSE messages."""

from __future__ import annotations


def sanitize_error_message(error: Exception) -> str:
    """Return a user-safe error string without leaking secrets."""
    error_str = str(error) or type(error).__name__
    lower = error_str.lower()

    # Rate limits often mention openrouter.ai — check before auth heuristics.
    if (
        "rate-limited" in lower
        or "rate limited" in lower
        or "rate limit" in lower
        or "429" in lower
        or "temporarily rate-limited" in lower
    ):
        return "Model is temporarily rate-limited. Please retry shortly or try another model."
    if "timeout" in lower or "timed out" in lower:
        return "Request timeout: the operation took too long. Please try again."
    if any(
        k in lower
        for k in (
            "api key",
            "api_key",
            "missing credentials",
            "unauthorized",
            "authentication",
            "401",
            "403",
        )
    ):
        return "Authentication error: please check your API configuration."
    if "ssl" in lower or "certificate" in lower:
        return "Network security error: please try again later."

    # Cap length so huge LLM dumps don't flood the SSE client
    if len(error_str) > 800:
        return error_str[:800] + "…"
    return error_str
