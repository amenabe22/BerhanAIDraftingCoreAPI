"""Document helpers: drafting-block extraction, legacy TipTap inbound, IDs."""

from __future__ import annotations

import html as html_lib
import re
import uuid
from typing import Any


def detect_language(text: str) -> str:
    """Return ``am`` if Amharic script is present, else ``en``."""
    if not text:
        return "en"
    if re.search(r"[\u1200-\u137F]", text):
        return "am"
    return "en"


def extract_text_from_node(node: dict[str, Any]) -> str:
    parts: list[str] = []

    def collect(n: dict[str, Any]) -> None:
        if n.get("type") == "text":
            parts.append(n.get("text", ""))
        for child in n.get("content") or []:
            collect(child)

    collect(node)
    return " ".join(parts)


def block_id_of(block: dict[str, Any]) -> str:
    return str(block.get("id") or block.get("block_id") or "")


def block_text_of(block: dict[str, Any]) -> str:
    return str(block.get("plain_text") or block.get("text") or "").strip()


def as_edit_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize drafting blocks for edit retrieval/prompts (block_id + text)."""
    out: list[dict[str, Any]] = []
    for i, block in enumerate(blocks or []):
        bid = block_id_of(block) or f"auto_{i}"
        text = block_text_of(block)
        out.append(
            {
                **block,
                "block_id": bid,
                "id": bid,
                "type": block.get("type") or "paragraph",
                "text": text,
                "plain_text": text,
                "index": int(block.get("index", i)),
                "html": block.get("html") or "",
                "level": block.get("level"),
            }
        )
    return out


def _html_for_text(block_type: str, text: str, level: int | None = None) -> str:
    escaped = html_lib.escape(text or "")
    if block_type == "heading":
        lvl = max(1, min(int(level or 1), 6))
        return f"<h{lvl}>{escaped}</h{lvl}>"
    if block_type == "blockquote":
        return f"<blockquote>{escaped}</blockquote>"
    return f"<p>{escaped}</p>"


def extract_blocks_from_tiptap(doc_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Legacy TipTap inbound → edit-shaped blocks (block_id, text, attrs, node)."""
    blocks: list[dict[str, Any]] = []
    seen_block_ids: set[str] = set()

    def traverse(node: dict[str, Any], is_top_level: bool = False) -> None:
        if not isinstance(node, dict):
            return

        attrs = node.get("attrs") or {}
        block_id = attrs.get("block_id")
        node_type = node.get("type")
        text_content = extract_text_from_node(node)

        should_extract = False
        final_block_id: str | None = None

        if block_id:
            if block_id not in seen_block_ids:
                should_extract = True
                final_block_id = block_id
                seen_block_ids.add(block_id)
        elif is_top_level and node_type in ("heading", "paragraph"):
            should_extract = True
            final_block_id = f"auto_{len(blocks)}"

        if should_extract and final_block_id:
            block_data: dict[str, Any] = {
                "block_id": final_block_id,
                "type": node_type,
                "text": text_content,
                "attrs": attrs,
                "node": node,
            }
            original_uuid = attrs.get("original_uuid")
            if original_uuid:
                block_data["original_uuid"] = original_uuid
            blocks.append(block_data)

        content = node.get("content") or []
        if isinstance(content, list):
            for child in content:
                child_is_top_level = (
                    (
                        is_top_level
                        and not block_id
                        and node_type
                        not in ("listItem", "bulletList", "orderedList", "pageBreak")
                    )
                    or node_type == "page"
                )
                traverse(child, child_is_top_level)

    for child in (doc_json or {}).get("content") or []:
        traverse(child, is_top_level=True)

    return blocks


def tiptap_to_drafting_blocks(doc_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert legacy TipTap doc JSON into canonical drafting blocks."""
    if not isinstance(doc_json, dict):
        return []

    drafting: list[dict[str, Any]] = []
    seen: set[str] = set()

    def walk(node: dict[str, Any], *, is_top_level: bool = False) -> None:
        if not isinstance(node, dict):
            return
        node_type = node.get("type") or ""
        if node_type in ("doc", "page"):
            for child in node.get("content") or []:
                walk(child, is_top_level=True)
            return
        if node_type == "pageBreak":
            drafting.append(
                {
                    "id": f"pb{len(drafting) + 1}",
                    "type": "pageBreak",
                    "index": len(drafting),
                    "level": None,
                    "html": "",
                    "plain_text": "",
                }
            )
            return

        attrs = node.get("attrs") or {}
        text = extract_text_from_node(node).strip()
        block_id = (
            attrs.get("block_id")
            or attrs.get("id")
            or attrs.get("blockId")
            or (
                f"b{len(drafting) + 1}"
                if text or node_type in ("heading", "paragraph")
                else None
            )
        )
        extractable = bool(block_id) or (
            is_top_level
            and node_type
            in ("heading", "paragraph", "blockquote", "bulletList", "orderedList")
        )
        if extractable and node_type not in ("text",):
            bid = str(block_id or f"auto_{len(drafting)}")
            if bid in seen:
                bid = f"{bid}_{len(drafting)}"
            seen.add(bid)
            level = attrs.get("level") if node_type == "heading" else None
            drafting.append(
                {
                    "id": bid,
                    "type": node_type or "paragraph",
                    "index": len(drafting),
                    "level": int(level) if level is not None else None,
                    "html": _html_for_text(node_type or "paragraph", text, level),
                    "plain_text": text,
                }
            )
            return

        for child in node.get("content") or []:
            walk(child, is_top_level=False)

    walk(doc_json, is_top_level=True)
    for i, block in enumerate(drafting):
        block["index"] = i
    return drafting


def resolve_document_blocks(document: Any) -> list[dict[str, Any]]:
    """Resolve blocks from a drafting envelope, raw list, or legacy TipTap doc."""
    if isinstance(document, list):
        return as_edit_blocks([b if isinstance(b, dict) else {} for b in document])
    if not isinstance(document, dict):
        return []
    if isinstance(document.get("blocks"), list):
        return as_edit_blocks(document["blocks"])
    if document.get("type") == "doc":
        return as_edit_blocks(tiptap_to_drafting_blocks(document))
    return []


def validate_blocks_structure(document: Any) -> bool:
    """True if document is a non-empty drafting blocks envelope or list."""
    if isinstance(document, list):
        return len(document) > 0 and all(isinstance(b, dict) for b in document)
    if not isinstance(document, dict):
        return False
    blocks = document.get("blocks")
    return isinstance(blocks, list) and len(blocks) > 0


def find_block_by_id(document: Any, block_id: str) -> dict[str, Any] | None:
    """Find a block by id in a blocks document or legacy TipTap doc."""
    if isinstance(document, dict) and isinstance(document.get("blocks"), list):
        for block in document["blocks"]:
            if isinstance(block, dict) and block_id_of(block) == block_id:
                return block
        return None

    def search(node: dict[str, Any]) -> dict[str, Any] | None:
        attrs = node.get("attrs") or {}
        if attrs.get("block_id") == block_id:
            return node
        for child in node.get("content") or []:
            found = search(child)
            if found:
                return found
        return None

    if not isinstance(document, dict):
        return None
    for child in document.get("content") or []:
        found = search(child)
        if found:
            return found
    return None


def generate_block_id(prefix: str = "b") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def validate_tiptap_structure(doc_json: dict[str, Any]) -> bool:
    """Legacy TipTap structure check (kept for inbound compatibility)."""
    return (
        isinstance(doc_json, dict)
        and doc_json.get("type") == "doc"
        and isinstance(doc_json.get("content"), list)
    )


def validate_document_structure(document: Any) -> bool:
    """Accept drafting blocks envelope/list or legacy TipTap doc."""
    return validate_blocks_structure(document) or (
        isinstance(document, dict) and validate_tiptap_structure(document)
    )


def find_page_by_index(doc_json: dict[str, Any], page_index: int) -> dict[str, Any] | None:
    pages = [n for n in doc_json.get("content") or [] if n.get("type") == "page"]
    if 0 <= page_index < len(pages):
        return pages[page_index]
    return None


def get_all_pages(doc_json: dict[str, Any]) -> list[dict[str, Any]]:
    return [n for n in doc_json.get("content") or [] if n.get("type") == "page"]


def envelope_from_blocks(
    blocks: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a drafting document envelope from block dicts."""
    normalized = as_edit_blocks(blocks)
    drafting: list[dict[str, Any]] = []
    for i, block in enumerate(normalized):
        drafting.append(
            {
                "id": block_id_of(block),
                "type": block.get("type") or "paragraph",
                "index": i,
                "level": block.get("level"),
                "html": block.get("html")
                or _html_for_text(
                    block.get("type") or "paragraph",
                    block_text_of(block),
                    block.get("level"),
                ),
                "plain_text": block_text_of(block),
            }
        )
    result: dict[str, Any] = {"blocks": drafting}
    if metadata:
        result["metadata"] = metadata
    return result
