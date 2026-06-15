"""Unit tests for operation schema validation."""

from app.services.drafting.editing.schemas import (
    ensure_op_ids,
    normalize_operation_type,
    validate_operations,
)


def test_normalize_rewrite_to_replace():
    assert normalize_operation_type("rewrite") == "replace"


def test_validate_replace_operation():
    data = {
        "operations": [
            {
                "op_id": "abc",
                "type": "replace",
                "block_id": "b1",
                "payload": {"new_text": "Updated"},
            }
        ]
    }
    ok, err = validate_operations(data)
    assert ok is True
    assert err is None


def test_validate_rejects_missing_block_id():
    data = {
        "operations": [
            {"op_id": "abc", "type": "replace", "payload": {"new_text": "x"}},
        ]
    }
    ok, err = validate_operations(data)
    assert ok is False
    assert "block_id" in (err or "")


def test_validate_insert_requires_new_text():
    data = {
        "operations": [
            {"op_id": "abc", "type": "insert", "block_id": "b1", "payload": {"position": "after"}},
        ]
    }
    ok, err = validate_operations(data)
    assert ok is False


def test_validate_remove_empty_payload():
    data = {
        "operations": [
            {"op_id": "abc", "type": "remove", "block_id": "b1", "payload": {}},
        ]
    }
    ok, _ = validate_operations(data)
    assert ok is True


def test_ensure_op_ids():
    ops = [{"type": "remove", "block_id": "b1", "payload": {}}]
    out = ensure_op_ids(ops)
    assert len(out[0]["op_id"]) == 8
