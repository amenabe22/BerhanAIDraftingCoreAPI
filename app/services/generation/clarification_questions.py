"""Normalize clarification questions to a shared structured shape."""

from __future__ import annotations

import re
import uuid
from typing import Any


QUESTION_OBJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["id", "prompt", "kind"],
    "properties": {
        "id": {"type": "string"},
        "prompt": {"type": "string"},
        "kind": {"type": "string", "enum": ["single_choice", "text"]},
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                },
            },
        },
        "allow_other": {"type": "boolean"},
    },
}


def _slug(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return (cleaned[:48] or fallback)


def normalize_option(raw: Any, index: int) -> dict[str, str] | None:
    if isinstance(raw, str):
        label = raw.strip()
        if not label:
            return None
        return {"id": _slug(label, f"opt_{index + 1}"), "label": label}
    if isinstance(raw, dict):
        label = str(raw.get("label") or raw.get("text") or raw.get("value") or "").strip()
        if not label:
            return None
        oid = str(raw.get("id") or "").strip() or _slug(label, f"opt_{index + 1}")
        return {"id": oid, "label": label}
    return None


def normalize_question(raw: Any, index: int) -> dict[str, Any] | None:
    """Coerce string or object into ClarificationQuestion; return None if unusable."""
    if isinstance(raw, str):
        prompt = raw.strip()
        if not prompt:
            return None
        return {
            "id": f"q_{index + 1}",
            "prompt": prompt,
            "kind": "text",
            "allow_other": False,
        }

    if not isinstance(raw, dict):
        return None

    prompt = str(
        raw.get("prompt") or raw.get("question") or raw.get("text") or ""
    ).strip()
    if not prompt:
        return None

    kind_raw = str(raw.get("kind") or raw.get("type") or "").strip().lower()
    options_in = raw.get("options") if isinstance(raw.get("options"), list) else []
    options: list[dict[str, str]] = []
    for i, opt in enumerate(options_in):
        normalized = normalize_option(opt, i)
        if normalized:
            options.append(normalized)

    if kind_raw in ("single_choice", "choice", "select", "radio") or (
        not kind_raw and len(options) >= 2
    ):
        kind = "single_choice"
    elif kind_raw in ("text", "free_text", "open"):
        kind = "text"
    else:
        kind = "single_choice" if len(options) >= 2 else "text"

    if kind == "single_choice" and len(options) < 2:
        # Not enough choices — fall back to free text.
        kind = "text"
        options = []

    qid = str(raw.get("id") or "").strip() or f"q_{index + 1}"
    allow_other = raw.get("allow_other")
    if allow_other is None:
        allow_other = kind == "single_choice"
    else:
        allow_other = bool(allow_other)

    out: dict[str, Any] = {
        "id": qid,
        "prompt": prompt,
        "kind": kind,
        "allow_other": allow_other if kind == "single_choice" else False,
    }
    if kind == "single_choice":
        out["options"] = options
    return out


def normalize_questions(raw: Any, *, max_questions: int = 5) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if len(out) >= max_questions:
            break
        q = normalize_question(item, i)
        if q:
            # Ensure unique ids
            existing = {x["id"] for x in out}
            if q["id"] in existing:
                q["id"] = f"{q['id']}_{uuid.uuid4().hex[:6]}"
            out.append(q)
    return out


def questions_prompt_block() -> str:
    """Shared JSON instructions for analysis / clarification prompts."""
    return """
QUESTION OBJECT FORMAT (required):
Each item in "questions" MUST be an object:
{
  "id": "q1",
  "prompt": "Short question text (1-2 lines max)",
  "kind": "single_choice" | "text",
  "options": [{"id": "a", "label": "Option A"}, {"id": "b", "label": "Option B"}],
  "allow_other": true
}

RULES:
- Prefer kind "single_choice" with 2-5 short options whenever answers are enumerable
  (document type, length band, governing law, yes/no, party roles, etc.).
- Use kind "text" ONLY when the answer must be free-form (names, addresses, unique amounts).
- For single_choice, set allow_other true unless unsafe; the UI adds an Other field.
- Keep prompt short; put choices in options, not inside the prompt text.
- Do NOT return questions as plain strings.
""".strip()
