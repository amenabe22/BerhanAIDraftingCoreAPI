"""Legacy TipTap export helpers — thin wrappers over blocks_text."""

from __future__ import annotations

from typing import Any

from app.services.export.blocks_text import blocks_to_markdown, validate_blocks_document


def tiptap_to_markdown(tiptap_doc: dict[str, Any] | list[Any]) -> str:
    """Convert TipTap or drafting blocks to markdown (legacy name)."""
    return blocks_to_markdown(tiptap_doc)


def validate_tiptap_document(document: dict[str, Any] | list[Any]) -> None:
    """Validate TipTap or drafting blocks document (legacy name)."""
    validate_blocks_document(document)
