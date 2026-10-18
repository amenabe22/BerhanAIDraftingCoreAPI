"""Shared drafting-block fixtures (legacy TipTap helpers retained)."""

from __future__ import annotations

from typing import Any


def make_block(
    block_id: str,
    text: str,
    *,
    block_type: str = "paragraph",
    level: int = 1,
    index: int = 0,
) -> dict[str, Any]:
    """Canonical drafting block."""
    if block_type == "heading":
        html = f"<h{level}>{text}</h{level}>"
        return {
            "id": block_id,
            "type": "heading",
            "index": index,
            "level": level,
            "html": html,
            "plain_text": text,
        }
    html = f"<p>{text}</p>"
    return {
        "id": block_id,
        "type": block_type,
        "index": index,
        "html": html,
        "plain_text": text,
    }


def make_en_doc() -> dict[str, Any]:
    return {
        "blocks": [
            make_block("b1", "Employment Agreement", block_type="heading", level=1, index=0),
            make_block("b2", "The Employee shall work forty hours per week.", index=1),
            make_block("b3", "The Employer may terminate this agreement with notice.", index=2),
        ]
    }


def make_amharic_doc() -> dict[str, Any]:
    return {
        "blocks": [
            make_block("b1", 'የቅጥር ስምምነት', block_type="heading", level=1, index=0),
            make_block("b2", 'ሰራተኛው በሳምንት አርባ ሰዓት ይሰራል።', index=1),
            make_block("b3", 'አሰሪው ስምምነቱን በማስታወቂያ ሊያቋርጥ ይችላል።', index=2),
        ]
    }


def make_multi_page_doc() -> dict[str, Any]:
    return {
        "blocks": [
            make_block("b1", "Page One Title", block_type="heading", level=1, index=0),
            make_block("b2", "Content on page one.", index=1),
            {
                "id": "pb1",
                "type": "pageBreak",
                "index": 2,
                "html": "",
                "plain_text": "",
            },
            make_block("b3", "Page Two Title", block_type="heading", level=1, index=3),
            make_block("b4", "Content on page two.", index=4),
        ]
    }


def make_single_block_doc() -> dict[str, Any]:
    return {"blocks": [make_block("b1", "Only block.", index=0)]}


def make_empty_content_doc() -> dict[str, Any]:
    return {"blocks": []}


def make_tiptap_block(
    block_id: str,
    text: str,
    *,
    block_type: str = "paragraph",
    level: int = 1,
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "type": block_type,
        "attrs": {"block_id": block_id},
        "content": [{"type": "text", "text": text}],
    }
    if block_type == "heading":
        node["attrs"]["level"] = level
    return node


def make_tiptap_en_doc() -> dict[str, Any]:
    return {
        "type": "doc",
        "content": [
            make_tiptap_block("b1", "Employment Agreement", block_type="heading", level=1),
            make_tiptap_block("b2", "The Employee shall work forty hours per week."),
            make_tiptap_block("b3", "The Employer may terminate this agreement with notice."),
        ],
    }
