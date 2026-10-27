"""Deterministic TipTap page-count enforcement.

IMPORTANT: TipTap Pro Pages paginates by content *height* (Letter format), not by
structural `page` / `pageBreak` nodes. Those nodes are organizational only and are
stripped before the editor renders. Visual page count ≈ total_words / ~300.

This module:
1. Clamps structural page nodes (legacy / LLM output shape)
2. Clamps total word budget to what TipTap Letter can actually show
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Calibrated to TipTap Pro Pages Letter @ 96dpi with default header/footer chrome.
# Content area ≈ 744px; 14px Verdana @ ~1.5 line-height ≈ 35 lines; legal prose
# with headings ≈ 280–320 words before a new visual page appears.
DEFAULT_WORDS_PER_VISUAL_PAGE = 300
DEFAULT_WORDS_PER_VISUAL_PAGE_MAX = 340

_SIGNATURE_NODE_RE = re.compile(
    r"(in witness whereof|signature|executed|በዚህ መሠረት|ፈጽመዋል)",
    re.IGNORECASE,
)


def _extract_text(node: Dict[str, Any]) -> str:
    parts: List[str] = []

    def walk(n: Any) -> None:
        if not isinstance(n, dict):
            return
        if n.get("type") == "text":
            parts.append(n.get("text") or "")
            return
        for child in n.get("content") or []:
            walk(child)

    walk(node)
    return " ".join(parts)


def count_words_in_node(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    text = _extract_text(node)
    return len(text.split()) if text.strip() else 0


def count_document_words(document: Dict[str, Any]) -> int:
    if not isinstance(document, dict):
        return 0
    total = 0
    for node in document.get("content") or []:
        total += count_words_in_node(node)
    return total


def estimate_tiptap_visual_pages(
    document: Dict[str, Any],
    *,
    words_per_page: int = DEFAULT_WORDS_PER_VISUAL_PAGE,
) -> int:
    """Estimate how many TipTap Letter pages the document will paint."""
    words = count_document_words(document)
    if words <= 0:
        return 0
    wpp = max(1, words_per_page)
    return max(1, (words + wpp - 1) // wpp)


def visual_word_budget(
    num_pages: int,
    *,
    words_per_page: int = DEFAULT_WORDS_PER_VISUAL_PAGE,
    words_per_page_max: int = DEFAULT_WORDS_PER_VISUAL_PAGE_MAX,
) -> Tuple[int, int]:
    """Return (target_words, max_words) for N TipTap visual pages.

    Hard max stays at n * words_per_page (not words_per_page_max). Using the
    higher per-page max as a multiplier pushed ceil(words/wpp) to N+1 and the
    editor showed an extra page. words_per_page_max is reserved as a soft
    ceiling for logging / validation warnings only.
    """
    n = max(1, int(num_pages))
    wpp = max(1, words_per_page)
    # Aim slightly under so headings/spacing don't tip into page N+1
    target = max(1, int(n * wpp * 0.9))
    max_w = n * wpp
    _ = words_per_page_max  # kept for API compat / callers
    return target, max_w


def _is_signature_node(node: Dict[str, Any]) -> bool:
    text = _extract_text(node)
    if not text.strip():
        return False
    if _SIGNATURE_NODE_RE.search(text):
        return True
    if node.get("type") == "heading":
        lower = text.lower()
        if "execution" in lower or "signature" in lower:
            return True
    return False


def _extract_pages(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = document.get("content") or []
    return [n for n in content if isinstance(n, dict) and n.get("type") == "page"]


def _find_signature_start(page: Dict[str, Any]) -> Optional[int]:
    content = page.get("content") or []
    for i, node in enumerate(content):
        if isinstance(node, dict) and _is_signature_node(node):
            return i
    return None


def _collect_signature_from_pages(
    pages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Pull signature/execution nodes from the last pages that contain them."""
    for page in reversed(pages):
        start = _find_signature_start(page)
        if start is None:
            continue
        content = page.get("content") or []
        return copy.deepcopy(content[start:])
    return []


def _kept_pages_have_signature(pages: List[Dict[str, Any]]) -> bool:
    for page in pages:
        if _find_signature_start(page) is not None:
            return True
    return False


def _reassemble(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for i, page in enumerate(pages):
        content.append(page)
        if i < len(pages) - 1:
            content.append({"type": "pageBreak"})
    return content


def _flatten_block_nodes(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Hoist page children (and top-level blocks) into a flat block list."""
    blocks: List[Dict[str, Any]] = []
    for node in document.get("content") or []:
        if not isinstance(node, dict):
            continue
        if node.get("type") == "page":
            for child in node.get("content") or []:
                if isinstance(child, dict) and child.get("type") != "pageBreak":
                    blocks.append(child)
        elif node.get("type") == "pageBreak":
            continue
        else:
            blocks.append(node)
    return blocks


def _pack_blocks_into_pages(
    blocks: List[Dict[str, Any]],
    n: int,
) -> List[Dict[str, Any]]:
    """Distribute blocks across n page nodes (roughly equal by word count)."""
    n = max(1, n)
    if not blocks:
        return [{"type": "page", "content": []}]

    total_words = sum(count_words_in_node(b) for b in blocks) or 1
    target_per_page = max(1, total_words // n)

    pages: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    current_words = 0

    for i, block in enumerate(blocks):
        w = count_words_in_node(block)
        # Start a new page when over target and we still have pages left to fill
        pages_left_after = n - len(pages) - 1
        blocks_left_including = len(blocks) - i
        if (
            current
            and pages_left_after > 0
            and current_words >= target_per_page
            and blocks_left_including > pages_left_after
        ):
            pages.append({"type": "page", "content": current})
            current = []
            current_words = 0
        current.append(block)
        current_words += w

    if current:
        pages.append({"type": "page", "content": current})

    # If we somehow made fewer pages, that's fine (under-budget).
    # If more than n (shouldn't), merge overflow into last.
    if len(pages) > n:
        head = pages[: n - 1]
        tail_blocks: List[Dict[str, Any]] = []
        for p in pages[n - 1 :]:
            tail_blocks.extend(p.get("content") or [])
        head.append({"type": "page", "content": tail_blocks})
        pages = head

    return pages


def clamp_document_to_word_budget(
    document: Dict[str, Any],
    max_words: int,
) -> Tuple[Dict[str, Any], int]:
    """
    Trim trailing non-signature blocks so total words <= max_words.
    Preserves signature/execution block when present.
    Returns (document, final_word_count).
    """
    if not isinstance(document, dict) or document.get("type") != "doc":
        return document, 0
    if not isinstance(max_words, int) or max_words < 1:
        return document, count_document_words(document)

    blocks = _flatten_block_nodes(document)
    if not blocks:
        return document, 0

    signature_start = None
    for i, block in enumerate(blocks):
        if _is_signature_node(block):
            signature_start = i
            break

    signature_blocks: List[Dict[str, Any]] = []
    body_blocks = blocks
    if signature_start is not None:
        signature_blocks = copy.deepcopy(blocks[signature_start:])
        body_blocks = blocks[:signature_start]

    signature_words = sum(count_words_in_node(b) for b in signature_blocks)
    body_budget = max(0, max_words - signature_words)

    kept: List[Dict[str, Any]] = []
    used = 0
    for block in body_blocks:
        w = count_words_in_node(block)
        if kept and used + w > body_budget:
            break
        if not kept and w > body_budget:
            # Keep at least one body block even if over (avoid empty docs)
            kept.append(copy.deepcopy(block))
            used += w
            break
        kept.append(copy.deepcopy(block))
        used += w

    kept.extend(signature_blocks)

    pages = _extract_pages(document)
    n_pages = len(pages) if pages else 1
    packed = _pack_blocks_into_pages(kept, n_pages)

    out = copy.deepcopy(document)
    out["content"] = _reassemble(packed)
    final_words = count_document_words(out)
    if final_words < count_document_words(document):
        logger.info(
            f"Clamped document words from {count_document_words(document)} to "
            f"{final_words} (max_budget={max_words})"
        )
    return out, final_words


def clamp_document_to_pages(
    document: Dict[str, Any],
    n: int,
    *,
    words_per_page: int = DEFAULT_WORDS_PER_VISUAL_PAGE,
    words_per_page_max: int = DEFAULT_WORDS_PER_VISUAL_PAGE_MAX,
) -> Tuple[Dict[str, Any], int]:
    """
    Cap document to at most n *visual* TipTap Letter pages.

    1) Structural page-node clamp (LLM output shape)
    2) Word-budget clamp so TipTap height pagination ≈ n pages

    Returns (document, estimated_visual_page_count).
    """
    if not isinstance(document, dict) or document.get("type") != "doc":
        return document, 0
    if not isinstance(n, int) or n < 1:
        return document, estimate_tiptap_visual_pages(document, words_per_page=words_per_page)

    pages = _extract_pages(document)
    page_count = len(pages)
    working = document

    if page_count > n:
        kept = copy.deepcopy(pages[:n])
        dropped = pages[n:]

        if not _kept_pages_have_signature(kept):
            signature_nodes = _collect_signature_from_pages(dropped)
            if signature_nodes:
                last = kept[-1]
                last_content = list(last.get("content") or [])
                last_content.extend(signature_nodes)
                last["content"] = last_content
                logger.info(
                    "Relocated signature/execution block onto last kept page after clamp "
                    f"({page_count} -> {n})"
                )

        working = copy.deepcopy(document)
        working["content"] = _reassemble(kept)
        logger.info(f"Clamped structural pages from {page_count} to {n}")

    _, max_words = visual_word_budget(
        n,
        words_per_page=words_per_page,
        words_per_page_max=words_per_page_max,
    )
    before_words = count_document_words(working)
    if before_words > max_words:
        working, _ = clamp_document_to_word_budget(working, max_words)

    est = estimate_tiptap_visual_pages(working, words_per_page=words_per_page)
    return working, est


def resolve_target_pages(requirements: Optional[Dict[str, Any]]) -> Optional[int]:
    """Return explicit num_pages when valid (int or digit string), else None."""
    if not requirements:
        return None
    from app.services.drafting.compliance_rules import coerce_num_pages

    return coerce_num_pages(requirements.get("num_pages"))
