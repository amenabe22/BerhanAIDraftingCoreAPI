"""Operation JSON schemas and validation."""

from __future__ import annotations

import uuid
from typing import Any

import jsonschema

OPERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op_id": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "replace",
                            "insert",
                            "remove",
                            "change_heading",
                            "rewrite",
                            "summarize",
                            "expand",
                            "simplify",
                            "clarify",
                            "strengthen",
                            "weaken",
                            "merge_pages",
                            "delete_page",
                            "move_page",
                        ],
                    },
                    "block_id": {"type": "string"},
                    "page_index": {"type": "integer"},
                    "page_index_1": {"type": "integer"},
                    "page_index_2": {"type": "integer"},
                    "target_index": {"type": "integer"},
                    "payload": {
                        "type": "object",
                        "properties": {
                            "new_text": {"type": "string"},
                            "level": {"type": "integer"},
                            "position": {"type": "string"},
                        },
                    },
                },
                "required": ["op_id", "type", "payload"],
            },
            "minItems": 1,
        }
    },
    "required": ["operations"],
}

OPS_REQUIRING_TEXT = frozenset(
    {
        "replace",
        "insert",
        "rewrite",
        "summarize",
        "expand",
        "simplify",
        "clarify",
        "strengthen",
        "weaken",
    }
)

SEMANTIC_OPS = frozenset(
    {"rewrite", "summarize", "expand", "simplify", "clarify", "strengthen", "weaken"}
)


def normalize_operation_type(op_type: str) -> str:
    if op_type in SEMANTIC_OPS:
        return "replace"
    return op_type


def validate_operations(json_data: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        jsonschema.validate(instance=json_data, schema=OPERATION_SCHEMA)
        for op in json_data.get("operations") or []:
            op_type = op.get("type")
            payload = op.get("payload") or {}

            if op_type == "merge_pages":
                if "page_index_1" not in op or "page_index_2" not in op:
                    return False, "merge_pages requires page_index_1 and page_index_2"
            elif op_type == "delete_page":
                if "page_index" not in op:
                    return False, "delete_page requires page_index"
            elif op_type == "move_page":
                if "page_index" not in op or "target_index" not in op:
                    return False, "move_page requires page_index and target_index"
            elif "block_id" not in op:
                return False, f"Operation type '{op_type}' requires block_id"

            if op_type in OPS_REQUIRING_TEXT and not payload.get("new_text"):
                return False, f"Operation type '{op_type}' requires new_text in payload"
            if op_type == "change_heading" and "level" not in payload:
                return False, "change_heading requires level in payload"
        return True, None
    except jsonschema.ValidationError as exc:
        return False, str(exc)
    except jsonschema.SchemaError as exc:
        return False, f"Schema error: {exc}"


def validate_block_exists(doc_json: dict[str, Any], block_id: str) -> bool:
    from app.services.drafting.editing.tiptap import find_block_by_id

    return find_block_by_id(doc_json, block_id) is not None


def generate_fallback_operation(
    doc_json: dict[str, Any],
    relevant_blocks: list[dict[str, Any]],
    reason: str = "LLM failure or invalid response",
) -> dict[str, Any]:
    del reason
    for block in relevant_blocks:
        block_id = block.get("block_id")
        if block_id and validate_block_exists(doc_json, block_id):
            break
    return {"operations": []}


def ensure_op_ids(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for op in operations:
        if not op.get("op_id"):
            op = {**op, "op_id": uuid.uuid4().hex[:8]}
        normalized.append(op)
    return normalized
