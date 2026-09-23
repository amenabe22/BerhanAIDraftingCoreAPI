"""Classify markdown-ish export lines for PDF/DOCX layout."""

from __future__ import annotations

import re

_ORDERED = re.compile(r"^(\d+)\.\s+(.+)$")
_LIST_CUES = re.compile(
    r"\b(shall|must|will|may not|agrees|including|the receiving|the disclosing)\b",
    re.I,
)


def looks_like_section_title(text: str) -> bool:
    """True for short clause titles like 'Purpose', not full sentences."""
    title = text.strip()
    if not title or len(title) > 90:
        return False
    if _LIST_CUES.search(title):
        return False
    words = title.split()
    if re.search(r"[.?!።]$", title) and len(words) > 6:
        return False
    return True


def classify_md_line(line: str) -> str:
    raw = (line or "").strip()
    if not raw:
        return "blank"
    if raw == "---":
        return "rule"
    if raw.startswith("# "):
        return "title"
    if raw.startswith("## "):
        return "h2"
    if raw.startswith("### "):
        return "h3"
    if raw.startswith("- "):
        return "bullet"
    match = _ORDERED.match(raw)
    if match:
        if looks_like_section_title(match.group(2)):
            return "h2"
        return "ordered"
    return "body"


def heading_text(line: str, kind: str) -> str:
    raw = line.strip()
    if kind == "title":
        return raw[2:].strip()
    if kind == "h2" and raw.startswith("## "):
        return raw[3:].strip()
    if kind == "h3":
        return raw[4:].strip()
    if kind == "h2":
        return raw
    if kind == "bullet":
        return raw[2:].strip()
    if kind == "ordered":
        return re.sub(r"^\d+\.\s+", "", raw)
    return raw
