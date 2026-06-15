"""Deterministic TipTap operation application."""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any

from app.services.drafting.editing.schemas import normalize_operation_type
from app.services.drafting.editing.tiptap import (
    extract_blocks_from_tiptap,
    find_page_by_index,
    generate_block_id,
    get_all_pages,
)

logger = logging.getLogger(__name__)


def apply_operations(doc_json: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, Any]:
    result = deepcopy(doc_json)

    if "_block_id_to_uuid" in doc_json:
        result["_block_id_to_uuid"] = doc_json["_block_id_to_uuid"].copy()

    original_uuid_map: dict[str, str] = {}

    def extract_original_uuids(node: dict[str, Any]) -> None:
        attrs = node.get("attrs") or {}
        bid = attrs.get("block_id")
        ouuid = attrs.get("original_uuid")
        if bid and ouuid:
            original_uuid_map[bid] = ouuid
        for child in node.get("content") or []:
            extract_original_uuids(child)

    for node in result.get("content") or []:
        extract_original_uuids(node)

    existing_blocks = {b["block_id"] for b in extract_blocks_from_tiptap(doc_json)}

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
            try:
                if op_type == "merge_pages":
                    result = _apply_merge_pages(
                        result, op.get("page_index_1", 0), op.get("page_index_2", 1)
                    )
                elif op_type == "delete_page":
                    result = _apply_delete_page(result, op.get("page_index", 0))
                elif op_type == "move_page":
                    result = _apply_move_page(
                        result, op.get("page_index", 0), op.get("target_index", 0)
                    )
            except Exception as exc:
                logger.error("Page operation %s failed: %s", op.get("op_id"), exc)
            continue

        if not block_id:
            logger.warning("Operation %s missing block_id", op.get("op_id"))
            continue

        if block_id not in existing_blocks and not block_id.startswith("auto_"):
            logger.warning("Block %s not in document, skipping op %s", block_id, op.get("op_id"))
            continue

        try:
            if op_type == "replace":
                result = _apply_replace(result, block_id, payload)
            elif op_type == "insert":
                result = _apply_insert(result, block_id, payload)
            elif op_type == "remove":
                result = _apply_remove(result, block_id)
            elif op_type == "change_heading":
                result = _apply_change_heading(result, block_id, payload)
            else:
                logger.warning("Unknown operation type: %s", op_type)
        except Exception as exc:
            logger.error("Operation %s failed: %s", op.get("op_id"), exc)

    def restore_original_uuids(node: dict[str, Any]) -> None:
        attrs = node.get("attrs") or {}
        bid = attrs.get("block_id")
        if bid and bid in original_uuid_map and "original_uuid" not in attrs:
            attrs["original_uuid"] = original_uuid_map[bid]
        for child in node.get("content") or []:
            restore_original_uuids(child)

    for node in result.get("content") or []:
        restore_original_uuids(node)

    if result.get("type") != "doc" or "content" not in result:
        logger.error("Document structure corrupted after operations")
        return doc_json

    return result


def _block_matches(
    node: dict[str, Any],
    block_id: str,
    *,
    auto_counter: list[int] | None = None,
    is_top_level: bool = False,
) -> bool:
    attrs = node.get("attrs") or {}
    node_block_id = attrs.get("block_id")
    node_type = node.get("type")

    if node_block_id == block_id:
        return True

    if block_id.startswith("auto_") and not node_block_id and auto_counter is not None:
        try:
            target = int(block_id.split("_", 1)[1])
        except (ValueError, IndexError):
            return False
        if is_top_level and node_type in ("heading", "paragraph"):
            current = auto_counter[0]
            auto_counter[0] += 1
            return current == target

    return False


def _apply_replace(doc_json: dict[str, Any], block_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    new_text = payload.get("new_text", "")
    lines = new_text.strip().split("\n")
    is_markdown_list = len(lines) > 1 and all(
        re.match(r"^\s*[-*]\s+", line.strip()) for line in lines if line.strip()
    )
    auto_counter = [0]

    def replace_in_node(
        node: dict[str, Any],
        parent_list: list[dict[str, Any]],
        node_index: int,
        is_top_level: bool = False,
    ) -> bool:
        node_type = node.get("type")
        if _block_matches(node, block_id, auto_counter=auto_counter, is_top_level=is_top_level):
            if node_type == "paragraph" and is_markdown_list:
                list_items = []
                for line in lines:
                    line = line.strip()
                    if line.startswith(("- ", "* ")):
                        item_text = line[2:].strip()
                        list_items.append(
                            {
                                "type": "listItem",
                                "attrs": {"block_id": generate_block_id()},
                                "content": [
                                    {
                                        "type": "paragraph",
                                        "content": [{"type": "text", "text": item_text}],
                                    }
                                ],
                            }
                        )
                if list_items:
                    parent_list[node_index] = {"type": "bulletList", "content": list_items}
                    return True

            if node_type in ("paragraph", "heading"):
                node["content"] = [{"type": "text", "text": new_text}]
                return True
            if node_type == "listItem":
                for child in node.get("content") or []:
                    if child.get("type") == "paragraph":
                        child["content"] = [{"type": "text", "text": new_text}]
                        return True

        for i, child in enumerate(node.get("content") or []):
            child_top = (
                is_top_level
                and not (node.get("attrs") or {}).get("block_id")
                and node_type not in ("listItem", "bulletList", "orderedList", "pageBreak")
            ) or node_type == "page"
            if replace_in_node(child, node.get("content"), i, child_top):
                return True
        return False

    for i, child in enumerate(doc_json.get("content") or []):
        if replace_in_node(child, doc_json["content"], i, is_top_level=True):
            break
    return doc_json


def _apply_insert(doc_json: dict[str, Any], block_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    new_text = payload.get("new_text", "")
    position = payload.get("position", "after")
    new_block = {
        "type": "paragraph",
        "attrs": {"block_id": generate_block_id()},
        "content": [{"type": "text", "text": new_text}],
    }
    auto_counter = [0]

    def insert_relative(
        node: dict[str, Any],
        parent_list: list[dict[str, Any]],
        index: int,
        is_top_level: bool = False,
    ) -> bool:
        if _block_matches(node, block_id, auto_counter=auto_counter, is_top_level=is_top_level):
            insert_at = index if position == "before" else index + 1
            parent_list.insert(insert_at, deepcopy(new_block))
            return True

        node_type = node.get("type")
        for i, child in enumerate(node.get("content") or []):
            child_top = (
                is_top_level
                and not (node.get("attrs") or {}).get("block_id")
                and node_type not in ("listItem", "bulletList", "orderedList", "pageBreak")
            ) or node_type == "page"
            if insert_relative(child, node.get("content"), i, child_top):
                return True
        return False

    for i, child in enumerate(doc_json.get("content") or []):
        if insert_relative(child, doc_json["content"], i, is_top_level=True):
            break
    return doc_json


def _apply_remove(doc_json: dict[str, Any], block_id: str) -> dict[str, Any]:
    auto_counter = [0]

    def remove_node(
        node: dict[str, Any],
        parent_list: list[dict[str, Any]],
        index: int,
        is_top_level: bool = False,
    ) -> bool:
        if _block_matches(node, block_id, auto_counter=auto_counter, is_top_level=is_top_level):
            parent_list.pop(index)
            return True

        node_type = node.get("type")
        for i, child in enumerate(node.get("content") or []):
            child_top = (
                is_top_level
                and not (node.get("attrs") or {}).get("block_id")
                and node_type not in ("listItem", "bulletList", "orderedList", "pageBreak")
            ) or node_type == "page"
            if remove_node(child, node.get("content"), i, child_top):
                return True
        return False

    for i, child in enumerate(doc_json.get("content") or []):
        if remove_node(child, doc_json["content"], i, is_top_level=True):
            break
    return doc_json


def _apply_change_heading(
    doc_json: dict[str, Any], block_id: str, payload: dict[str, Any]
) -> dict[str, Any]:
    new_level = payload.get("level", 1)
    new_text = payload.get("new_text", "")

    def change_heading_in_node(node: dict[str, Any]) -> bool:
        attrs = node.get("attrs") or {}
        if attrs.get("block_id") == block_id and node.get("type") == "heading":
            node["attrs"]["level"] = new_level
            if new_text:
                node["content"] = [{"type": "text", "text": new_text}]
            return True
        for child in node.get("content") or []:
            if change_heading_in_node(child):
                return True
        return False

    for child in doc_json.get("content") or []:
        change_heading_in_node(child)
    return doc_json


def _apply_merge_pages(doc_json: dict[str, Any], page_index_1: int, page_index_2: int) -> dict[str, Any]:
    page1 = find_page_by_index(doc_json, page_index_1)
    page2 = find_page_by_index(doc_json, page_index_2)
    if not page1 or not page2:
        return doc_json
    page1["content"] = (page1.get("content") or []) + (page2.get("content") or [])
    content = doc_json.get("content") or []
    for i in range(len(content) - 1, -1, -1):
        if content[i] is page2:
            if i > 0 and content[i - 1].get("type") == "pageBreak":
                content.pop(i - 1)
                content.pop(i - 1)
            else:
                content.pop(i)
            break
    return doc_json


def _apply_delete_page(doc_json: dict[str, Any], page_index: int) -> dict[str, Any]:
    page = find_page_by_index(doc_json, page_index)
    if not page:
        return doc_json
    content = doc_json.get("content") or []
    for i in range(len(content) - 1, -1, -1):
        if content[i] is page:
            if i > 0 and content[i - 1].get("type") == "pageBreak":
                content.pop(i - 1)
                content.pop(i - 1)
            else:
                content.pop(i)
            break
    return doc_json


def _apply_move_page(doc_json: dict[str, Any], page_index: int, target_index: int) -> dict[str, Any]:
    pages = get_all_pages(doc_json)
    if not (0 <= page_index < len(pages) and 0 <= target_index < len(pages)):
        return doc_json
    if page_index == target_index:
        return doc_json

    page_to_move = pages[page_index]
    content = doc_json.get("content") or []
    page_break_to_move = None
    for i in range(len(content) - 1, -1, -1):
        if content[i] is page_to_move:
            if i > 0 and content[i - 1].get("type") == "pageBreak":
                page_break_to_move = content[i - 1]
                content.pop(i - 1)
                content.pop(i - 1)
            else:
                content.pop(i)
            break

    pages_after = [n for n in content if n.get("type") == "page"]
    page_break = page_break_to_move or {"type": "pageBreak"}
    if target_index < len(pages_after):
        target_page = pages_after[target_index]
        for i, node in enumerate(content):
            if node is target_page:
                content.insert(i, page_break)
                content.insert(i, page_to_move)
                break
    else:
        content.extend([page_break, page_to_move])
    return doc_json


def compute_simple_diff(before: dict[str, Any], after: dict[str, Any]) -> str:
    before_blocks = {b["block_id"]: b["text"] for b in extract_blocks_from_tiptap(before)}
    after_blocks = {b["block_id"]: b["text"] for b in extract_blocks_from_tiptap(after)}
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
