"""Deterministic drafting-block operation application."""

from __future__ import annotations

import logging
from typing import Any

from app.services.drafting.editing.schemas import normalize_operation_type
from app.services.drafting.editing.tiptap import (
    _html_for_text,
    as_edit_blocks,
    block_id_of,
    block_text_of,
    envelope_from_blocks,
    generate_block_id,
    resolve_document_blocks,
)

logger = logging.getLogger(__name__)


def _to_blocks_envelope(document: Any) -> dict[str, Any]:
    if isinstance(document, dict) and isinstance(document.get("blocks"), list):
        meta = document.get("metadata") if isinstance(document.get("metadata"), dict) else None
        return envelope_from_blocks(document["blocks"], meta)
    blocks = resolve_document_blocks(document)
    meta = None
    if isinstance(document, dict) and isinstance(document.get("metadata"), dict):
        meta = document["metadata"]
    return envelope_from_blocks(blocks, meta)


def apply_operations(document: Any, operations: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply edit operations to a blocks document (or legacy TipTap inbound)."""
    result = _to_blocks_envelope(document)
    blocks = as_edit_blocks(result["blocks"])
    existing = {block_id_of(b) for b in blocks}

    for op in operations:
        op_type_raw = op.get("type", "")
        op_type = normalize_operation_type(op_type_raw)
        block_id = op.get("block_id")
        payload = op.get("payload") or {}

        if op_type in ("merge_pages", "delete_page", "move_page") or op_type_raw in (
            "merge_pages",
            "delete_page",
            "move_page",
        ):
            logger.debug("Skipping page operation %s on blocks document", op.get("op_id"))
            continue

        if not block_id:
            logger.warning("Operation %s missing block_id", op.get("op_id"))
            continue

        if block_id not in existing and not str(block_id).startswith("auto_"):
            logger.warning("Block %s not in document, skipping op %s", block_id, op.get("op_id"))
            continue

        try:
            if op_type == "replace":
                blocks = _apply_replace(blocks, block_id, payload)
            elif op_type == "insert":
                blocks = _apply_insert(blocks, block_id, payload)
                existing = {block_id_of(b) for b in blocks}
            elif op_type == "remove":
                blocks = _apply_remove(blocks, block_id)
                existing = {block_id_of(b) for b in blocks}
            elif op_type == "change_heading":
                blocks = _apply_change_heading(blocks, block_id, payload)
            else:
                logger.warning("Unknown operation type: %s", op_type)
        except Exception as exc:
            logger.error("Operation %s failed: %s", op.get("op_id"), exc)

    meta = result.get("metadata") if isinstance(result.get("metadata"), dict) else None
    return envelope_from_blocks(blocks, meta)


def _find_index(blocks: list[dict[str, Any]], block_id: str) -> int:
    for i, block in enumerate(blocks):
        if block_id_of(block) == block_id:
            return i
    return -1


def _apply_replace(
    blocks: list[dict[str, Any]], block_id: str, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    new_text = payload.get("new_text", "")
    idx = _find_index(blocks, block_id)
    if idx < 0:
        return blocks
    block = dict(blocks[idx])
    btype = block.get("type") or "paragraph"
    level = block.get("level")
    block["plain_text"] = new_text
    block["text"] = new_text
    block["html"] = _html_for_text(btype, new_text, level)
    out = list(blocks)
    out[idx] = block
    return out


def _apply_insert(
    blocks: list[dict[str, Any]], block_id: str, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    new_text = payload.get("new_text", "")
    position = payload.get("position", "after")
    idx = _find_index(blocks, block_id)
    if idx < 0:
        return blocks
    bid = generate_block_id()
    new_block = {
        "id": bid,
        "block_id": bid,
        "type": "paragraph",
        "index": 0,
        "level": None,
        "html": _html_for_text("paragraph", new_text),
        "plain_text": new_text,
        "text": new_text,
    }
    insert_at = idx if position == "before" else idx + 1
    out = list(blocks)
    out.insert(insert_at, new_block)
    return out


def _apply_remove(blocks: list[dict[str, Any]], block_id: str) -> list[dict[str, Any]]:
    return [b for b in blocks if block_id_of(b) != block_id]


def _apply_change_heading(
    blocks: list[dict[str, Any]], block_id: str, payload: dict[str, Any]
) -> list[dict[str, Any]]:
    new_level = payload.get("level", 1)
    new_text = payload.get("new_text", "")
    idx = _find_index(blocks, block_id)
    if idx < 0:
        return blocks
    block = dict(blocks[idx])
    if (block.get("type") or "") != "heading":
        block["type"] = "heading"
    block["level"] = new_level
    text = new_text if new_text else block_text_of(block)
    block["plain_text"] = text
    block["text"] = text
    block["html"] = _html_for_text("heading", text, new_level)
    out = list(blocks)
    out[idx] = block
    return out


def compute_simple_diff(before: Any, after: Any) -> str:
    before_blocks = {block_id_of(b): block_text_of(b) for b in resolve_document_blocks(before)}
    after_blocks = {block_id_of(b): block_text_of(b) for b in resolve_document_blocks(after)}
    parts: list[str] = []
    for bid, text in before_blocks.items():
        if bid not in after_blocks:
            parts.append(f"Removed block {bid}")
        elif after_blocks[bid] != text:
            parts.append(f"Modified block {bid}")
    for bid in after_blocks:
        if bid not in before_blocks:
            parts.append(f"Added block {bid}")
    return "; ".join(parts) if parts else "No changes detected"
