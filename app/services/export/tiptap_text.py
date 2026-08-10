"""TipTap JSON → markdown-ish text for PDF/DOCX export."""

from __future__ import annotations

from typing import Any


def _extract_text(content_list: list[Any] | None) -> str:
    if not content_list:
        return ""
    parts: list[str] = []
    for item in content_list:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append(item.get("text") or "")
        elif "content" in item:
            parts.append(_extract_text(item.get("content")))
    return "".join(parts)


def tiptap_to_markdown(tiptap_doc: dict[str, Any]) -> str:
    """Convert TipTap doc JSON to markdown-formatted text."""
    lines: list[str] = []

    def process_node(node: dict[str, Any]) -> None:
        node_type = node.get("type", "")
        content = node.get("content") or []

        if node_type == "doc":
            for child in content:
                if isinstance(child, dict):
                    process_node(child)
            return

        if node_type == "page":
            for child in content:
                if isinstance(child, dict):
                    process_node(child)
            lines.append("")
            return

        if node_type == "pageBreak":
            lines.append("")
            lines.append("---")
            lines.append("")
            return

        if node_type == "heading":
            level = int((node.get("attrs") or {}).get("level") or 1)
            level = max(1, min(level, 6))
            text = _extract_text(content).strip()
            if text:
                lines.append(f"{'#' * level} {text}")
                lines.append("")
            return

        if node_type == "paragraph":
            text = _extract_text(content).strip()
            if text:
                lines.append(text)
                lines.append("")
            return

        if node_type == "bulletList":
            for item in content:
                if isinstance(item, dict) and item.get("type") == "listItem":
                    item_text = _extract_text(item.get("content")).strip()
                    if item_text:
                        lines.append(f"- {item_text}")
            lines.append("")
            return

        if node_type == "orderedList":
            idx = 1
            for item in content:
                if isinstance(item, dict) and item.get("type") == "listItem":
                    item_text = _extract_text(item.get("content")).strip()
                    if item_text:
                        lines.append(f"{idx}. {item_text}")
                        idx += 1
            lines.append("")
            return

        if node_type == "blockquote":
            text = _extract_text(content).strip()
            if text:
                for part in text.split("\n"):
                    lines.append(f"> {part}")
                lines.append("")
            return

        # Fallback: recurse into children
        for child in content:
            if isinstance(child, dict):
                process_node(child)

    if not isinstance(tiptap_doc, dict):
        raise ValueError("document must be a TipTap JSON object")

    process_node(tiptap_doc)
    # Collapse excessive blank lines
    out: list[str] = []
    blank = 0
    for line in lines:
        if not line.strip():
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(line)
    return "\n".join(out).strip()


def validate_tiptap_document(document: dict[str, Any]) -> None:
    if not isinstance(document, dict):
        raise ValueError("document must be an object")
    if document.get("type") != "doc":
        raise ValueError('document.type must be "doc"')
    content = document.get("content")
    if not isinstance(content, list) or len(content) == 0:
        raise ValueError("document.content must be a non-empty array")
