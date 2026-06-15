"""TipTap document helpers: block extraction, language detection, IDs."""

from __future__ import annotations

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


def extract_blocks_from_tiptap(doc_json: dict[str, Any]) -> list[dict[str, Any]]:
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

    for child in doc_json.get("content") or []:
        traverse(child, is_top_level=True)

    return blocks


def find_block_by_id(doc_json: dict[str, Any], block_id: str) -> dict[str, Any] | None:
    def search(node: dict[str, Any]) -> dict[str, Any] | None:
        attrs = node.get("attrs") or {}
        if attrs.get("block_id") == block_id:
            return node
        for child in node.get("content") or []:
            found = search(child)
            if found:
                return found
        return None

    for child in doc_json.get("content") or []:
        found = search(child)
        if found:
            return found
    return None


def generate_block_id(prefix: str = "b") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}"


def validate_tiptap_structure(doc_json: dict[str, Any]) -> bool:
    return (
        isinstance(doc_json, dict)
        and doc_json.get("type") == "doc"
        and isinstance(doc_json.get("content"), list)
    )


def find_page_by_index(doc_json: dict[str, Any], page_index: int) -> dict[str, Any] | None:
    pages = [n for n in doc_json.get("content") or [] if n.get("type") == "page"]
    if 0 <= page_index < len(pages):
        return pages[page_index]
    return None


def get_all_pages(doc_json: dict[str, Any]) -> list[dict[str, Any]]:
    return [n for n in doc_json.get("content") or [] if n.get("type") == "page"]
