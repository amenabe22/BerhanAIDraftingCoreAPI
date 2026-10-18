"""Unit tests for drafting-block / legacy TipTap helpers."""

from app.services.drafting.editing.tiptap import (
    detect_language,
    extract_blocks_from_tiptap,
    resolve_document_blocks,
    tiptap_to_drafting_blocks,
    validate_document_structure,
    validate_tiptap_structure,
)
from tests.fixtures.tiptap_docs import (
    make_amharic_doc,
    make_en_doc,
    make_multi_page_doc,
    make_tiptap_en_doc,
)


def test_resolve_blocks_from_envelope(en_doc):
    blocks = resolve_document_blocks(en_doc)
    assert [b["block_id"] for b in blocks] == ["b1", "b2", "b3"]
    assert "forty" in blocks[1]["text"]


def test_resolve_multi_page(multi_page_doc):
    blocks = resolve_document_blocks(multi_page_doc)
    ids = [b["block_id"] for b in blocks]
    assert "b1" in ids and "b4" in ids


def test_validate_document_structure(en_doc):
    assert validate_document_structure(en_doc) is True
    assert validate_document_structure({"blocks": []}) is False
    assert validate_tiptap_structure({"type": "paragraph"}) is False


def test_amharic_blocks(amharic_doc):
    blocks = resolve_document_blocks(amharic_doc)
    assert detect_language(blocks[0]["text"]) == "am"


def test_legacy_tiptap_extract_and_convert():
    tip = make_tiptap_en_doc()
    legacy = extract_blocks_from_tiptap(tip)
    drafting = tiptap_to_drafting_blocks(tip)
    assert len(legacy) == 3
    assert len(drafting) == 3
    assert drafting[0]["id"] == "b1"
    assert "Employment" in drafting[0]["plain_text"]
