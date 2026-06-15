"""Unit tests for deterministic apply_operations."""

from copy import deepcopy

from app.services.drafting.editing.ops import apply_operations, compute_simple_diff
from app.services.drafting.editing.tiptap import extract_blocks_from_tiptap
from tests.fixtures.tiptap_docs import make_en_doc, make_multi_page_doc


def _replace_op(block_id: str, new_text: str) -> dict:
    return {
        "op_id": "op1",
        "type": "replace",
        "block_id": block_id,
        "payload": {"new_text": new_text},
    }


def test_replace_block(en_doc):
    result = apply_operations(en_doc, [_replace_op("b2", "Thirty-five hours per week.")])
    blocks = extract_blocks_from_tiptap(result)
    b2 = next(b for b in blocks if b["block_id"] == "b2")
    assert "Thirty-five" in b2["text"]
    assert len(blocks) == 3


def test_remove_block(en_doc):
    result = apply_operations(
        en_doc,
        [{"op_id": "op1", "type": "remove", "block_id": "b1", "payload": {}}],
    )
    blocks = extract_blocks_from_tiptap(result)
    assert len(blocks) == 2
    assert all(b["block_id"] != "b1" for b in blocks)


def test_insert_after(en_doc):
    result = apply_operations(
        en_doc,
        [
            {
                "op_id": "op1",
                "type": "insert",
                "block_id": "b2",
                "payload": {"new_text": "Overtime requires approval.", "position": "after"},
            }
        ],
    )
    blocks = extract_blocks_from_tiptap(result)
    assert len(blocks) == 4
    texts = [b["text"] for b in blocks]
    idx_b2 = next(i for i, b in enumerate(blocks) if b["block_id"] == "b2")
    assert "Overtime" in blocks[idx_b2 + 1]["text"]


def test_insert_before(en_doc):
    result = apply_operations(
        en_doc,
        [
            {
                "op_id": "op1",
                "type": "insert",
                "block_id": "b3",
                "payload": {"new_text": "Prior clause.", "position": "before"},
            }
        ],
    )
    blocks = extract_blocks_from_tiptap(result)
    idx_b3 = next(i for i, b in enumerate(blocks) if b["block_id"] == "b3")
    assert blocks[idx_b3 - 1]["text"] == "Prior clause."


def test_preserve_unrelated_blocks(en_doc):
    original = deepcopy(en_doc)
    result = apply_operations(en_doc, [_replace_op("b2", "Changed only b2.")])
    blocks = extract_blocks_from_tiptap(result)
    b1 = next(b for b in blocks if b["block_id"] == "b1")
    b3 = next(b for b in blocks if b["block_id"] == "b3")
    orig_blocks = extract_blocks_from_tiptap(original)
    assert b1["text"] == next(b for b in orig_blocks if b["block_id"] == "b1")["text"]
    assert b3["text"] == next(b for b in orig_blocks if b["block_id"] == "b3")["text"]


def test_skip_invalid_block_id(en_doc):
    result = apply_operations(en_doc, [_replace_op("nonexistent", "nope")])
    assert extract_blocks_from_tiptap(result) == extract_blocks_from_tiptap(en_doc)


def test_multi_page_insert(multi_page_doc):
    result = apply_operations(
        multi_page_doc,
        [
            {
                "op_id": "op1",
                "type": "insert",
                "block_id": "b2",
                "payload": {"new_text": "Inserted on page one.", "position": "after"},
            }
        ],
    )
    blocks = extract_blocks_from_tiptap(result)
    assert any("Inserted on page one" in b["text"] for b in blocks)


def test_compute_simple_diff(en_doc):
    patched = apply_operations(en_doc, [_replace_op("b2", "New hours.")])
    diff = compute_simple_diff(en_doc, patched)
    assert "Modified block b2" in diff
