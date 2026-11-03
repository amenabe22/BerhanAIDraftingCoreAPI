from app.services.drafting.editing.live_document import (
    augment_message_with_live_document,
    plain_text_from_tiptap,
)


def test_augment_includes_live_document_body():
    doc = {
        "type": "doc",
        "content": [
            {
                "type": "heading",
                "attrs": {"block_id": "h1"},
                "content": [{"type": "text", "text": "Employment Agreement"}],
            },
            {
                "type": "paragraph",
                "attrs": {"block_id": "p1"},
                "content": [
                    {"type": "text", "text": "Party A and Party B agree to the following terms."}
                ],
            },
        ],
    }
    out = augment_message_with_live_document("is this doc legal?", doc)
    assert "CURRENT OPEN DOCUMENT" in out
    assert "Employment Agreement" in out
    assert "Party A and Party B" in out
    assert plain_text_from_tiptap(doc).startswith("Employment Agreement")
