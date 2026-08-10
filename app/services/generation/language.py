"""Normalize and pin document-generation language preferences."""

from __future__ import annotations

from typing import Any

# Canonical codes used in prompts / TipTap metadata: am | en | om
_LANG_ALIASES: dict[str, str] = {
    "am": "am",
    "amh": "am",
    "amharic": "am",
    "አማርኛ": "am",
    "en": "en",
    "eng": "en",
    "english": "en",
    "om": "om",
    "oro": "om",
    "oromo": "om",
    "afaan": "om",
    "afaan_oromo": "om",
    "afaanoromo": "om",
}

_LANG_LABELS: dict[str, str] = {
    "am": "Amharic (አማርኛ)",
    "en": "English",
    "om": "Afaan Oromo",
}


def normalize_language_code(value: Any, default: str = "en") -> str:
    """Map aliases (amh, oro, …) to canonical am/en/om."""
    if value is None or value == "":
        return default
    if hasattr(value, "value"):
        value = value.value
    raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _LANG_ALIASES.get(raw, default if raw not in _LANG_ALIASES else raw)


def language_label(code: Any) -> str:
    canon = normalize_language_code(code)
    return _LANG_LABELS.get(canon, canon)


def coerce_language_input(value: Any) -> Any:
    """Pydantic before-validator: accept aliases, leave None alone."""
    if value is None or value == "":
        return value
    if hasattr(value, "value"):
        return value
    raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    mapped = _LANG_ALIASES.get(raw)
    return mapped if mapped else value


def pin_preferred_language(
    requirements: dict[str, Any],
    preferred: Any | None,
) -> dict[str, Any]:
    """Force user/agent language preference over any LLM-guessed language."""
    out = dict(requirements or {})
    if preferred is None or preferred == "":
        if "language" in out:
            out["language"] = normalize_language_code(out.get("language"))
        return out
    out["language"] = normalize_language_code(preferred)
    return out
