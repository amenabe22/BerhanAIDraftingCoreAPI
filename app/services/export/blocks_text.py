"""Drafting blocks → markdown-ish text for PDF/DOCX export."""

from __future__ import annotations

import html as html_lib
import re
from typing import Any

from app.models.drafting.blocks import DraftingBlock


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(value: str) -> str:
    return _TAG_RE.sub("", value or "").strip()


def _block_plain(block: dict[str, Any] | DraftingBlock) -> str:
    if isinstance(block, DraftingBlock):
        text = block.plain_text or _strip_html(block.html)
        return text.strip()
    text = (block.get("plain_text") or "").strip()
    if text:
        return text
    return _strip_html(str(block.get("html") or ""))


def _block_type(block: dict[str, Any] | DraftingBlock) -> str:
    if isinstance(block, DraftingBlock):
        return block.type
    return str(block.get("type") or "paragraph")


def _block_level(block: dict[str, Any] | DraftingBlock) -> int:
    if isinstance(block, DraftingBlock):
        level = block.level or 1
    else:
        level = int(block.get("level") or 1)
    return max(1, min(level, 6))


def normalize_blocks(blocks: list[Any]) -> list[dict[str, Any]]:
    """Coerce mixed block payloads into plain drafting-block dicts."""
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(blocks or []):
        if isinstance(raw, DraftingBlock):
            data = raw.model_dump()
        elif isinstance(raw, dict):
            data = dict(raw)
        else:
            continue
        block_id = data.get("id") or data.get("block_id") or f"b{i + 1}"
        plain = (data.get("plain_text") or data.get("text") or "").strip()
        html = data.get("html") or ""
        if not plain and html:
            plain = _strip_html(html)
        if not html and plain:
            btype = str(data.get("type") or "paragraph")
            escaped = html_lib.escape(plain)
            if btype == "heading":
                level = max(1, min(int(data.get("level") or 1), 6))
                html = f"<h{level}>{escaped}</h{level}>"
            else:
                html = f"<p>{escaped}</p>"
        level = data.get("level")
        out.append(
            {
                "id": str(block_id),
                "type": str(data.get("type") or "paragraph"),
                "index": int(data.get("index", i)),
                "level": int(level) if level is not None else None,
                "html": str(html or ""),
                "plain_text": plain,
            }
        )
    for i, block in enumerate(out):
        block["index"] = i
    return out


def validate_blocks_document(document: Any) -> list[dict[str, Any]]:
    """Validate a blocks document or list; return normalized blocks.

    Accepts:
    - list of blocks
    - ``{"blocks": [...], "metadata"?: ...}``
    - legacy TipTap ``{"type": "doc", "content": [...]}`` (converted)
    """
    if isinstance(document, list):
        blocks = normalize_blocks(document)
    elif isinstance(document, dict):
        if isinstance(document.get("blocks"), list):
            blocks = normalize_blocks(document["blocks"])
        elif document.get("type") == "doc":
            from app.services.drafting.editing.tiptap import tiptap_to_drafting_blocks

            blocks = tiptap_to_drafting_blocks(document)
        else:
            raise ValueError('document must include a non-empty "blocks" array')
    else:
        raise ValueError("document must be a blocks list or object")

    if not blocks:
        raise ValueError("blocks must be a non-empty array")
    for block in blocks:
        if not block.get("id"):
            raise ValueError("each block requires an id")
        if not block.get("type"):
            raise ValueError("each block requires a type")
    return blocks


def blocks_to_markdown(document: Any) -> str:
    """Convert drafting blocks (or legacy TipTap) to markdown-formatted text."""
    blocks = validate_blocks_document(document)
    lines: list[str] = []

    for block in blocks:
        btype = _block_type(block)
        text = _block_plain(block)
        if btype == "pageBreak":
            lines.extend(["", "---", ""])
            continue
        if not text:
            continue
        if btype == "heading":
            level = _block_level(block)
            lines.append(f"{'#' * level} {text}")
            lines.append("")
        elif btype in ("bulletList", "listItem"):
            for part in text.split("\n"):
                part = part.strip()
                if part:
                    lines.append(f"- {part}")
            lines.append("")
        elif btype == "orderedList":
            for idx, part in enumerate(
                (p.strip() for p in text.split("\n") if p.strip()), start=1
            ):
                lines.append(f"{idx}. {part}")
            lines.append("")
        elif btype == "blockquote":
            for part in text.split("\n"):
                lines.append(f"> {part}")
            lines.append("")
        else:
            lines.append(text)
            lines.append("")

    out: list[str] = []
    blank = 0
    for line in lines:
        if not line.strip():
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(line)
    return "\n".join(out).strip()
