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


_CURRENCY_AMOUNT_RE = re.compile(
    r"^(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<cur>ETB|ብር|Birr|birr)?$",
    re.UNICODE,
)

_PERCENT_FACTOR_RE = re.compile(
    r"(?:(?:lower|reduce|cut|decrease|መቀነስ|ቀንስ|አሳንስ).{0,40}?(?:to\s+)?(?:like\s+)?"
    r"(?P<pct>\d+(?:\.\d+)?)\s*%"
    r"|(?P<pct2>\d+(?:\.\d+)?)\s*%.{0,40}?(?:lower|reduce|cut|decrease|መቀነስ)"
    r"|(?:by\s+)?(?:a\s+)?half|ግማሽ|halve)",
    re.IGNORECASE,
)


def parse_percent_factor(text: str) -> float | None:
    raw = text or ""
    if re.search(r"\bhalf\b|halve|ግማሽ", raw, re.IGNORECASE):
        return 0.5
    m = _PERCENT_FACTOR_RE.search(raw)
    if not m:
        return None
    pct_s = m.groupdict().get("pct") or m.groupdict().get("pct2")
    if not pct_s:
        return 0.5
    try:
        pct = float(pct_s)
    except ValueError:
        return None
    if pct <= 0 or pct > 100:
        return None
    return pct / 100.0


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def format_financial_figures_for_tool(
    hits: list[dict[str, Any]],
    *,
    factor: float | None = None,
) -> str:
    if not hits:
        return (
            "No monetary amounts, currency figures, percentages, or ratios "
            "(e.g. 50/50) were found in the live document JSON."
        )
    lines = [
        f"Found {len(hits)} financial figure(s) in the live document. "
        "These ARE in the document — never claim none were found. "
        "Use these for clarify options or edit instructions:"
    ]
    for i, h in enumerate(hits, 1):
        section = h.get("section") or "(no heading)"
        lines.append(
            f"{i}. amount={h.get('amount')!r} section={section!r} "
            f"block_id={h.get('block_id')} snippet={h.get('snippet')!r}"
        )
    if factor is not None:
        pct = int(round(factor * 100))
        lines.append("")
        lines.append(
            f"COMPUTED TARGET (factor={factor} ≈ {pct}% of current currency amounts). "
            "Do NOT ask the user to type a new birr/ETB amount."
        )
        proposals = 0
        for h in hits:
            m = _CURRENCY_AMOUNT_RE.match(str(h.get("amount") or "").strip())
            if not m:
                continue
            try:
                value = float(m.group("num").replace(",", ""))
            except ValueError:
                continue
            cur = (m.group("cur") or "ETB").upper()
            if cur in ("BIRR", "ብር"):
                cur = "ETB"
            proposed = f"{_format_number(value * factor)} {cur}"
            lines.append(
                f"- Replace {h.get('amount')!r} → {proposed!r} "
                f"(section={h.get('section')!r})"
            )
            proposals += 1
        if proposals == 1:
            lines.append(
                "ACTION: apply_document_edit NOW with that replacement. "
                "Do not ask_clarifying_questions for the new amount."
            )
        elif proposals > 1:
            lines.append(
                "ACTION: If the target figure is known, apply_document_edit immediately. "
                "Only ask which currency amount(s) — never ask for the new number."
            )
    return "\n".join(lines)


_FINANCIAL_INTENT_RE = re.compile(
    r"(?:financial|amount|payment|penalty|fee|capital|contribution|"
    r"ገንዘብ|ብር|ETB|አስተዋጾ|አስተዋጽኦ|ክፍያ|ቅጣት|ድርሻ|አሃዝ|"
    r"(?:lower|reduce|halve|raise|increase|change|መቀነስ|ቀንስ).{0,48}"
    r"(?:\d+\s*%|half|50|ግማሽ))"
    r"|(?:\d+\s*%).{0,48}(?:lower|reduce|halve|raise|financial|amount|መቀነስ)",
    re.IGNORECASE,
)


def looks_like_financial_edit(text: str) -> bool:
    return bool(_FINANCIAL_INTENT_RE.search(text or ""))


def augment_message_with_financial_figures(
    message: str,
    doc_json: dict[str, Any] | None,
) -> str:
    """Append live amount scan so the LLM cannot invent 'no figures found'."""
    text = (message or "").strip()
    if not text or not looks_like_financial_edit(text):
        return message
    # Already augmented by CoreAPI hybrid path
    if "AUTHORITATIVE live document financial figures" in text or "COMPUTED TARGET" in text:
        return message
    hits = extract_financial_figures(doc_json)
    factor = parse_percent_factor(text)
    body = format_financial_figures_for_tool(hits, factor=factor)
    return (
        f"{text}\n\n---\nAUTHORITATIVE live document financial figures "
        f"(do not claim none exist):\n{body}\n---"
    )
