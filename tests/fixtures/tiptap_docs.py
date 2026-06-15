"""Shared TipTap document fixtures for semantic edit tests."""

from __future__ import annotations

from typing import Any


def make_block(
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


def make_en_doc() -> dict[str, Any]:
    return {
        "type": "doc",
        "content": [
            make_block("b1", "Employment Agreement", block_type="heading", level=1),
            make_block("b2", "The Employee shall work forty hours per week."),
            make_block("b3", "The Employer may terminate this agreement with notice."),
        ],
    }


def make_amharic_doc() -> dict[str, Any]:
    return {
        "type": "doc",
        "content": [
            make_block("b1", "የቅጥር ስምምነት", block_type="heading", level=1),
            make_block("b2", "ሰራተኛው በሳምንት አርባ ሰዓት ይሰራል።"),
            make_block("b3", "አሰሪው ስምምነቱን በማስታወቂያ ሊያቋርጥ ይችላል።"),
        ],
    }


def make_multi_page_doc() -> dict[str, Any]:
    return {
        "type": "doc",
        "content": [
            {
                "type": "page",
                "content": [
                    make_block("b1", "Page One Title", block_type="heading", level=1),
                    make_block("b2", "Content on page one."),
                ],
            },
            {"type": "pageBreak"},
            {
                "type": "page",
                "content": [
                    make_block("b3", "Page Two Title", block_type="heading", level=1),
                    make_block("b4", "Content on page two."),
                ],
            },
        ],
    }


def make_single_block_doc() -> dict[str, Any]:
    return {"type": "doc", "content": [make_block("b1", "Only block.")]}


def make_empty_content_doc() -> dict[str, Any]:
    return {"type": "doc", "content": []}
