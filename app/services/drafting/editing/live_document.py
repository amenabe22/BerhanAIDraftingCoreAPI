"""Ground doc-agent consult turns on live TipTap JSON (not only Qdrant)."""

from __future__ import annotations

from typing import Any

from app.services.drafting.editing.tiptap import extract_blocks_from_tiptap


def plain_text_from_tiptap(doc_json: dict[str, Any] | None, *, max_chars: int = 14000) -> str:
    """Flatten TipTap JSON into readable plain text for LLM grounding."""
    if not isinstance(doc_json, dict) or doc_json.get("type") != "doc":
        return ""
    blocks = extract_blocks_from_tiptap(doc_json)
    parts: list[str] = []
    for block in blocks:
        text = " ".join(str(block.get("text") or "").split()).strip()
        if not text:
            continue
        node_type = str(block.get("type") or "")
        if node_type == "heading":
            parts.append(text)
        else:
            parts.append(text)
    body = "\n".join(parts).strip()
    if not body:
        return ""
    if len(body) > max_chars:
        return body[:max_chars].rstrip() + "\n…[truncated for length]"
    return body


def augment_message_with_live_document(
    message: str, doc_json: dict[str, Any] | None, *, max_chars: int = 14000
) -> str:
    """Append authoritative live document text so consult turns do not ask for uploads."""
    body = plain_text_from_tiptap(doc_json, max_chars=max_chars)
    if not body:
        return message
    return (
        f"{message}\n\n"
        "--- CURRENT OPEN DOCUMENT (authoritative live editor text; "
        "answer from this — never ask the user to describe, paste, or upload the document) ---\n"
        f"{body}\n"
        "--- END CURRENT OPEN DOCUMENT ---"
    )
