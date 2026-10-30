"""In-memory block retrieval using Cohere embed + rerank (multilingual, no BM25)."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_LEADING_SECTION_NUM_RE = re.compile(
    r"^(\d+)\s*[.):\-–—]\s*(.*)$",
    re.UNICODE,
)
_WS_RE = re.compile(r"\s+", re.UNICODE)

_PREFILTER_THRESHOLD = 50


def _get_cohere_client():
    import cohere

    if not settings.COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY is required for semantic edit retrieval")
    return cohere.Client(api_key=settings.COHERE_API_KEY, base_url=settings.COHERE_API_URL)


def _block_document_text(block: dict[str, Any]) -> str:
    parts: list[str] = []
    block_type = block.get("type") or "paragraph"
    parts.append(f"[{block_type}]")
    text = (block.get("text") or "").strip()
    if text:
        parts.append(text)
    return "\n".join(parts)


def _keyword_fallback_rank(instruction: str, blocks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Offline-safe fallback when Cohere is unavailable."""
    tokens = {t.lower() for t in instruction.split() if len(t) > 2}
    scored: list[tuple[float, dict[str, Any]]] = []
    for block in blocks:
        text = (block.get("text") or "").lower()
        score = sum(1 for t in tokens if t in text)
        scored.append((score, block))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [b for s, b in scored[:top_k] if s > 0] or blocks[:top_k]


def rank_blocks_for_instruction(
    instruction: str,
    blocks: list[dict[str, Any]],
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Rank document blocks against an edit instruction.

    Uses Cohere rerank-multilingual-v3.0 when API key is present; falls back to
    keyword overlap for tests/offline runs.
    """
    if not blocks:
        return []

    limit = top_k or settings.EDIT_RERANK_TOP_K
    if not instruction.strip():
        return blocks[:limit]

    if not settings.COHERE_API_KEY:
        return _keyword_fallback_rank(instruction, blocks, limit)

    try:
        client = _get_cohere_client()
        candidates = blocks
        if len(blocks) > _PREFILTER_THRESHOLD:
            candidates = _prefilter_with_embeddings(client, instruction, blocks, _PREFILTER_THRESHOLD)

        documents = [_block_document_text(b) for b in candidates]
        response = client.rerank(
            model=settings.COHERE_RERANK_MODEL,
            query=instruction,
            documents=documents,
            top_n=min(limit, len(documents)),
        )
        ranked: list[dict[str, Any]] = []
        for item in response.results:
            idx = item.index
            if 0 <= idx < len(candidates):
                block = dict(candidates[idx])
                block["rerank_score"] = item.relevance_score
                ranked.append(block)
        return ranked
    except Exception as exc:
        logger.warning("Cohere rerank failed, using keyword fallback: %s", exc)
        return _keyword_fallback_rank(instruction, blocks, limit)


def _prefilter_with_embeddings(
    client: Any,
    instruction: str,
    blocks: list[dict[str, Any]],
    keep: int,
) -> list[dict[str, Any]]:
    texts = [_block_document_text(b) for b in blocks]
    try:
        query_emb = client.embed(
            texts=[instruction],
            model=settings.COHERE_EMBEDDING_MODEL,
            input_type="search_query",
            truncate="END",
        ).embeddings[0]
        doc_embs = client.embed(
            texts=texts,
            model=settings.COHERE_EMBEDDING_MODEL,
            input_type="search_document",
            truncate="END",
        ).embeddings

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        scored = [(cosine(query_emb, emb), block) for emb, block in zip(doc_embs, blocks, strict=False)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored[:keep]]
    except Exception as exc:
        logger.warning("Embedding prefilter failed: %s", exc)
        return blocks[:keep]


def neighbor_blocks(
    all_blocks: list[dict[str, Any]],
    target_ids: set[str],
    window: int = 1,
) -> list[dict[str, Any]]:
    """Return target blocks plus immediate neighbors for edit context."""
    id_to_index = {b["block_id"]: i for i, b in enumerate(all_blocks)}
    indices: set[int] = set()
    for bid in target_ids:
        if bid in id_to_index:
            idx = id_to_index[bid]
            for j in range(max(0, idx - window), min(len(all_blocks), idx + window + 1)):
                indices.add(j)
    return [all_blocks[i] for i in sorted(indices)]


def _heading_level(block: dict[str, Any]) -> int:
    attrs = block.get("attrs") or {}
    level = attrs.get("level")
    try:
        return int(level) if level is not None else 2
    except (TypeError, ValueError):
        return 2


def expand_section_blocks(
    all_blocks: list[dict[str, Any]],
    seed_ids: set[str],
) -> set[str]:
    """Expand heading seeds to the full section (heading + body until next peer heading).

    For each seed block that is a heading, include every following block until a
    heading of the same or higher level (lower/equal level number). Non-heading
    seeds are kept as-is. Order of ``all_blocks`` must be document order.
    """
    if not all_blocks or not seed_ids:
        return set(seed_ids)

    id_to_index = {b["block_id"]: i for i, b in enumerate(all_blocks)}
    expanded: set[str] = set(seed_ids)

    for bid in list(seed_ids):
        idx = id_to_index.get(bid)
        if idx is None:
            continue
        block = all_blocks[idx]
        if block.get("type") != "heading":
            continue
        level = _heading_level(block)
        for j in range(idx + 1, len(all_blocks)):
            nxt = all_blocks[j]
            if nxt.get("type") == "heading" and _heading_level(nxt) <= level:
                break
            nxt_id = nxt.get("block_id")
            if nxt_id:
                expanded.add(nxt_id)

    return expanded


_BODY_TYPES = frozenset({"paragraph", "bulletList", "orderedList", "listItem", "blockquote"})


def _word_count(text: str) -> int:
    return len([t for t in (text or "").split() if t])


def _attach_section_parents(all_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate each block with parent_heading_id / parent_heading_text (document order)."""
    current_heading_id: str | None = None
    current_heading_text: str = ""
    enriched: list[dict[str, Any]] = []
    for block in all_blocks:
        b = dict(block)
        if b.get("type") == "heading":
            current_heading_id = b.get("block_id")
            current_heading_text = (b.get("text") or "").strip()
            b["parent_heading_id"] = None
            b["parent_heading_text"] = ""
            b["is_section_heading"] = True
        else:
            b["parent_heading_id"] = current_heading_id
            b["parent_heading_text"] = current_heading_text
            b["is_section_heading"] = False
        text = (b.get("text") or "").strip()
        b["char_len"] = len(text)
        b["word_count"] = _word_count(text)
        enriched.append(b)
    return enriched


def build_document_skeleton(all_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build outline + per-block length features for dimensional locate.

    Features are relative (median / quartile), not language-specific keywords.
    """
    enriched = _attach_section_parents(all_blocks)
    body = [
        b
        for b in enriched
        if b.get("type") != "heading" and (b.get("type") in _BODY_TYPES or b.get("char_len", 0) > 0)
    ]
    lengths = sorted(b["char_len"] for b in body if b["char_len"] > 0)
    median = lengths[len(lengths) // 2] if lengths else 0
    q1 = lengths[max(0, len(lengths) // 4)] if lengths else 0

    for b in enriched:
        cl = b.get("char_len") or 0
        if median > 0 and b.get("type") != "heading":
            b["relative_thinness"] = round(cl / median, 3)
            b["below_median"] = cl <= median
            b["bottom_quartile"] = cl <= q1 if q1 > 0 else cl <= median
        else:
            b["relative_thinness"] = 1.0 if b.get("type") == "heading" else 0.0
            b["below_median"] = False
            b["bottom_quartile"] = False

    outline = [
        {
            "block_id": b["block_id"],
            "level": _heading_level(b),
            "title": (b.get("text") or "")[:120],
        }
        for b in enriched
        if b.get("type") == "heading"
    ]

    feature_rows = [
        {
            "block_id": b["block_id"],
            "type": b.get("type"),
            "char_len": b["char_len"],
            "word_count": b["word_count"],
            "parent_heading_id": b.get("parent_heading_id"),
            "parent_heading_text": (b.get("parent_heading_text") or "")[:80],
            "relative_thinness": b.get("relative_thinness"),
            "below_median": b.get("below_median"),
            "bottom_quartile": b.get("bottom_quartile"),
            "preview": (b.get("text") or "")[:160],
        }
        for b in enriched
        if b.get("type") != "heading"
    ]

    return {
        "outline": outline,
        "stats": {
            "block_count": len(enriched),
            "heading_count": len(outline),
            "body_count": len(body),
            "median_body_chars": median,
            "q1_body_chars": q1,
        },
        "body_features": feature_rows,
        "enriched_blocks": enriched,
    }


def structural_candidate_blocks(
    all_blocks: list[dict[str, Any]],
    *,
    thinness: str = "below_median",
    max_targets: int | None = None,
) -> list[dict[str, Any]]:
    """Select body blocks by relative length features (language-agnostic).

    ``thinness``:
      - ``below_median``: body blocks at or below median char length
      - ``bottom_quartile``: body blocks at or below Q1
      - ``all_body``: all non-heading body blocks under a section heading
    """
    skeleton = build_document_skeleton(all_blocks)
    enriched = skeleton["enriched_blocks"]
    limit = max_targets if max_targets is not None else getattr(
        settings, "EDIT_STRUCTURAL_MAX_TARGETS", 24
    )

    mode = (thinness or "below_median").strip().lower()
    selected: list[dict[str, Any]] = []
    for b in enriched:
        if b.get("type") == "heading":
            continue
        if not (b.get("char_len") or 0):
            continue
        # Prefer content under a heading (skip orphan preamble stubs when outline exists)
        if skeleton["outline"] and not b.get("parent_heading_id"):
            continue
        if mode == "all_body":
            selected.append(b)
        elif mode == "bottom_quartile":
            if b.get("bottom_quartile"):
                selected.append(b)
        else:  # below_median default
            if b.get("below_median"):
                selected.append(b)

    if not selected and mode != "all_body":
        # Soften: if nothing is "thin" relative to median (uniform short doc), take all bodies
        selected = [
            b
            for b in enriched
            if b.get("type") != "heading"
            and (b.get("char_len") or 0) > 0
            and (not skeleton["outline"] or b.get("parent_heading_id"))
        ]

    return selected[:limit] if limit else selected


def merge_candidate_blocks(
    *groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Stable union of candidate lists by block_id (first occurrence wins)."""
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for group in groups:
        for b in group:
            bid = b.get("block_id")
            if not bid or bid in seen:
                continue
            seen.add(bid)
            merged.append(b)
    return merged


def _norm_match_text(value: str) -> str:
    return _WS_RE.sub(" ", (value or "").strip().lower())


def _split_section_title(title: str) -> tuple[str | None, str]:
    """Return (leading_number, remainder) for titles like ``3. Profit sharing``."""
    raw = (title or "").strip()
    m = _LEADING_SECTION_NUM_RE.match(raw)
    if not m:
        return None, raw
    return m.group(1), (m.group(2) or "").strip()


def match_named_section_headings(
    instruction: str,
    outline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match outline headings referenced in an instruction (language-agnostic).

    Detects numbered titles (``3. …``, ``4) …``) and substantial title substrings
    so multi-section follow-ups like ``expand 3. X, 4. Y and 8. Z`` resolve to
    heading seeds instead of a single semantic hit.
    """
    instr = _norm_match_text(instruction)
    if not instr or not outline:
        return []

    matched: list[dict[str, Any]] = []
    seen: set[str] = set()
    for heading in outline:
        bid = heading.get("block_id")
        title = heading.get("title") or ""
        if not bid or bid in seen:
            continue
        norm_title = _norm_match_text(title)
        num, remainder = _split_section_title(title)
        norm_remainder = _norm_match_text(remainder)

        hit = False
        if norm_title and len(norm_title) >= 4 and norm_title in instr:
            hit = True
        elif norm_remainder and len(norm_remainder) >= 4 and norm_remainder in instr:
            hit = True
        elif num:
            # Numbered reference: "4." / "4)" / "section 4" near this heading's words
            num_patterns = (
                rf"(?<!\d){re.escape(num)}\s*[.):\-–—]",
                rf"(?<!\d){re.escape(num)}(?!\d)",
            )
            number_mentioned = any(re.search(p, instr) for p in num_patterns)
            if number_mentioned and norm_remainder:
                # Prefer titles whose distinctive tokens also appear (avoids bare "3" noise)
                tokens = [t for t in norm_remainder.split() if len(t) >= 3]
                if not tokens or any(t in instr for t in tokens):
                    hit = True
            elif number_mentioned and not norm_remainder:
                hit = True

        if hit:
            seen.add(bid)
            matched.append(dict(heading))

    return matched
