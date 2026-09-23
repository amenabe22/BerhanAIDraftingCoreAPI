"""In-memory block retrieval using Cohere embed + rerank (multilingual, no BM25)."""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

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
