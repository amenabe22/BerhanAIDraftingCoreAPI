"""Deterministic extraction of monetary / ratio figures from TipTap docs.

Semantic Qdrant search often misses short amount lines (e.g. ``ገንዘብ: 500,000 ETB``)
in favour of headings. This scanner reads the live ``doc_json`` so clarify/edit
flows can ground options in real figures.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.drafting.editing.tiptap import extract_blocks_from_tiptap

# Currency amounts, percentages, and simple ratios (50/50)
_AMOUNT_RE = re.compile(
    r"("
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:\s*(?:ETB|ብር|Birr|birr))?"
    r"|"
    r"\d+(?:\.\d+)?\s*(?:ETB|ብር|Birr|birr)"
    r"|"
    r"\d+(?:\.\d+)?\s*%"
    r"|"
    r"\d+\s*/\s*\d+"
    r")",
    re.UNICODE,
)

_NOISE_ONLY = re.compile(r"^[\s_\.]+$")


def _heading_level(block: dict[str, Any]) -> int:
    attrs = block.get("attrs") or {}
    try:
        return int(attrs.get("level") or 2)
    except (TypeError, ValueError):
        return 2


def extract_financial_figures(
    doc_json: dict[str, Any] | None,
    *,
    max_hits: int = 24,
) -> list[dict[str, Any]]:
    """Return unique amount hits with surrounding block/section context."""
    if not isinstance(doc_json, dict) or doc_json.get("type") != "doc":
        return []

    blocks = extract_blocks_from_tiptap(doc_json)
    current_heading = ""
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()

    for block in blocks:
        text = (block.get("text") or "").strip()
        if not text or _NOISE_ONLY.match(text):
            continue
        if block.get("type") == "heading":
            current_heading = text[:120]
            continue

        for match in _AMOUNT_RE.finditer(text):
            raw = match.group(1).strip()
            # Skip lone tiny integers that aren't currency-like unless % or ratio
            if (
                re.fullmatch(r"\d{1,3}", raw)
                and "%" not in raw
                and "/" not in raw
            ):
                continue
            key = f"{block.get('block_id')}|{raw}|{text[:80]}"
            if key in seen:
                continue
            seen.add(key)
            # Prefer a short window around the match for the label
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)
            snippet = text[start:end].strip()
            if start > 0:
                snippet = "…" + snippet
            if end < len(text):
                snippet = snippet + "…"
            hits.append(
                {
                    "block_id": block.get("block_id"),
                    "amount": raw,
                    "section": current_heading or None,
                    "snippet": snippet[:200],
                    "block_text": text[:280],
                }
            )
            if len(hits) >= max_hits:
                return hits

    return hits


def format_financial_figures_for_tool(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return (
            "No monetary amounts, currency figures, percentages, or ratios "
            "(e.g. 50/50) were found in the live document JSON."
        )
    lines = [
        f"Found {len(hits)} financial figure(s) in the live document. "
        "Use these for clarify options or edit instructions:"
    ]
    for i, h in enumerate(hits, 1):
        section = h.get("section") or "(no heading)"
        lines.append(
            f"{i}. amount={h.get('amount')!r} section={section!r} "
            f"block_id={h.get('block_id')} snippet={h.get('snippet')!r}"
        )
    return "\n".join(lines)
