"""Unit tests for TipTap helpers."""

from app.services.drafting.editing.tiptap import (
    detect_language,
    extract_blocks_from_tiptap,
    generate_block_id,
    validate_tiptap_structure,
)
from tests.fixtures.tiptap_docs import make_amharic_doc, make_en_doc, make_multi_page_doc


def test_detect_language_english():
    assert detect_language("Hello world") == "en"


def test_detect_language_amharic():
    assert detect_language("ሰላም") == "am"


def test_extract_blocks_order_and_ids(en_doc):
    blocks = extract_blocks_from_tiptap(en_doc)
    assert [b["block_id"] for b in blocks] == ["b1", "b2", "b3"]
    assert blocks[0]["type"] == "heading"
    assert "forty hours" in blocks[1]["text"]


def test_extract_blocks_multi_page(multi_page_doc):
    blocks = extract_blocks_from_tiptap(multi_page_doc)
    assert len(blocks) == 4
    assert blocks[0]["block_id"] == "b1"
    assert blocks[-1]["block_id"] == "b4"


def test_generate_block_id_format():
    bid = generate_block_id()
    assert bid.startswith("b")
    assert len(bid) == 9


def test_validate_tiptap_structure(en_doc):
    assert validate_tiptap_structure(en_doc) is True
    assert validate_tiptap_structure({"type": "paragraph"}) is False


def test_amharic_doc_blocks(amharic_doc):
    blocks = extract_blocks_from_tiptap(amharic_doc)
    assert len(blocks) == 3
    assert detect_language(blocks[0]["text"]) == "am"
