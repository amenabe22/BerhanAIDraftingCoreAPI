"""Targeted legal knowledge retrieval for compliance: query gen, search + RRF, LLM rerank."""

import re

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from qdrant_client.models import FieldCondition, Filter, MatchAny

from app.config import settings
from app.logging_config import get_logger
from app.retrieval import get_legal_kb_vector_store

log = get_logger("knowledge_retrieval")

# RRF constant (standard value)
RRF_K = 60

# Clause query: cap synonyms and total length so we never pass huge article-number lists to search
MAX_SYNONYM_TERMS = 20
MAX_CLAUSE_QUERY_CHARS = 1200
# Terms that are only digits or "Article N" (optional "Article") count as article-like; allow a few, skip the rest
MAX_ARTICLE_LIKE_TERMS = 3
_RE_ARTICLE_NUMBER = re.compile(r"^(?:Article\s*)?(\d+)$", re.I)

# Cache for available sources (in-process, no TTL for simplicity)
_available_sources_cache: list[str] | None = None


def _compliance_llm(model: str | None = None, temperature: float | None = None) -> ChatOpenAI:
    """LLM for compliance query gen and rerank (fast model, deterministic)."""
    temp = (
        settings.COMPLIANCE_ANALYSIS_TEMPERATURE
        if temperature is None
        else temperature
    )
    # Query gen / rerank stay on the fast default model, not the reasoning analysis model.
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENROUTER_API_KEY,
        model=model or settings.GEMINI_MODEL,
        temperature=temp,
        streaming=False,
        model_kwargs={"seed": settings.COMPLIANCE_ANALYSIS_SEED},
    )


def _is_article_like(term: str) -> bool:
    """True if term is only a number or 'Article N' (used to cap article-number dumping)."""
    return bool(_RE_ARTICLE_NUMBER.match(term.strip()))


def build_clause_legal_query(clause_text: str, implications: str) -> str:
    """Turn clause + implications into a legal-concept query plus dynamic synonyms for retrieval (no raw clause dump)."""
    clause_snippet = (clause_text or "").strip()[:400]
    impl_snippet = (implications or "").strip()[:300]
    if not clause_snippet and not impl_snippet:
        return ""
    prompt = f"""You are helping to find relevant Ethiopian law for a contract clause.

Clause (excerpt): {clause_snippet}
Legal implications (excerpt): {impl_snippet}

Output exactly two lines:
Line 1: One short search query that would find the Ethiopian legal provisions governing this topic. Describe the legal concept, not the clause wording.
Line 2: Up to 15–20 synonyms and related terms (comma-separated). Include: alternative phrasings, terms used in Ethiopian statutes, statute or proclamation names. You may include 1–3 specific article references (e.g. "Article 29", "Article 3325") only if directly relevant to this clause. Do NOT list article numbers in sequence (e.g. do not output Article 3325, 3326, 3327, ...). Output only conceptual terms and a few key article references.

Output only these two lines, no other text."""
    try:
        llm = _compliance_llm()
        response = llm.invoke(prompt)
        text = (response.content if hasattr(response, "content") else str(response)).strip()
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()][:2]
        query_line = lines[0] if lines else ""
        synonyms_line = lines[1] if len(lines) > 1 else ""
        # Combine query + synonyms; cap terms and filter out long article-number runs
        parts = [query_line]
        query_lower = query_line.lower()
        article_like_count = 0
        for term in (t.strip() for t in synonyms_line.split(",") if t.strip()):
            if len(parts) > MAX_SYNONYM_TERMS:
                break
            if not term or term.lower() in query_lower:
                continue
            if _is_article_like(term):
                article_like_count += 1
                if article_like_count > MAX_ARTICLE_LIKE_TERMS:
                    continue
            parts.append(term)
        result = " ".join(parts).strip()
        if len(result) > MAX_CLAUSE_QUERY_CHARS:
            result = result[:MAX_CLAUSE_QUERY_CHARS].rsplit(" ", 1)[0]
        if result:
            log.info(
                "clause_legal_query",
                extra={"event": "compliance_clause_query", "legal_query": result},
            )
        return result or ""
    except Exception as e:
        log.warning(
            "build_clause_legal_query failed",
            extra={"event": "clause_query_error", "error": str(e)},
        )
        return ""


def generate_targeted_queries(doc_type: str, summary: str) -> list[str]:
    """Generate 2–4 targeted search queries from document type and summary. One LLM call."""
    llm = _compliance_llm()
    prompt = f"""You are helping to find relevant Ethiopian law for a compliance analysis.

Document type: {doc_type or "unknown"}
Document summary (first part): {summary[:800] if summary else "N/A"}

Output exactly 2 to 4 short search queries that would find the most relevant Ethiopian legal provisions (Civil Code, proclamations, regulations) for this document. Each query should be one line, specific (e.g. "Ethiopian Civil Code contract liability limitation", "Labour Proclamation termination notice"). Output ONLY the queries, one per line, no numbering or bullets."""
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    queries = [q.strip() for q in text.strip().splitlines() if q.strip()][:4]
    if not queries:
        # Fallback: one generic query
        queries = [f"Ethiopian law {doc_type or 'contract'} compliance"]
    log.info(
        "targeted_queries",
        extra={
            "event": "compliance_queries",
            "doc_type": doc_type,
            "count": len(queries),
            "queries": queries,
        },
    )
    return queries


def _doc_key(doc: Document) -> tuple[str, str]:
    """Stable key for deduplication: (document_id, item_id). Uses fallbacks for Legal KB payload variants."""
    m = getattr(doc, "metadata", None) or {}
    nested = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
    doc_id = (
        m.get("document_id") or nested.get("document_id") or m.get("source_file") or m.get("source")
    ) or ""
    item_id = (
        m.get("item_id") or nested.get("item_id") or m.get("article_id") or m.get("article_number")
    ) or ""
    return (str(doc_id), str(item_id))


def _rrf_merge(ranked_lists: list[list[Document]], top_k: int) -> list[Document]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion. Dedupe by (document_id, item_id)."""
    scores: dict[tuple[str, str], float] = {}
    doc_map: dict[tuple[str, str], Document] = {}

    for rank_list in ranked_lists:
        for rank, doc in enumerate(rank_list, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            if key not in doc_map:
                doc_map[key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda k: -scores[k])
    return [doc_map[k] for k in sorted_keys[:top_k]]


def search_legal_knowledge(
    queries: list[str],
    source_filter: list[str] | None = None,
    top_k_per_query: int = 12,
    rrf_top_k: int = 15,
) -> list[Document]:
    """Run vector search for each query, merge with RRF, return top rrf_top_k. Optional filter by source."""
    store = get_legal_kb_vector_store()
    filter_obj = None
    if source_filter:
        # Qdrant: filter by document_id (or source_file) in payload
        filter_obj = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchAny(any=source_filter),
                )
            ]
        )
        # If collection uses "source_file" instead, we could add alternative key

    ranked_lists: list[list[Document]] = []
    for q in queries:
        try:
            if filter_obj:
                docs = store.similarity_search(q, k=top_k_per_query, filter=filter_obj)
            else:
                docs = store.similarity_search(q, k=top_k_per_query)
        except TypeError:
            # Store may not support filter in similarity_search; fall back to unfiltered
            docs = store.similarity_search(q, k=top_k_per_query)
        ranked_lists.append(docs)

    merged = _rrf_merge(ranked_lists, rrf_top_k)
    log.info(
        "search_legal_knowledge",
        extra={
            "event": "compliance_retrieve",
            "query_count": len(queries),
            "result_count": len(merged),
        },
    )
    return merged


def rerank_with_llm(query: str, chunks: list[Document], top_k: int) -> list[Document]:
    """Rerank chunks by relevance to query using an LLM. Returns top_k in order."""
    if not chunks:
        return []
    if len(chunks) <= top_k:
        return chunks

    model = (
        settings.COMPLIANCE_RERANKER_MODEL
        or settings.COMPLIANCE_ANALYSIS_MODEL
        or settings.GEMINI_MODEL
    )
    llm = _compliance_llm(model=model)

    # Number chunks 1..n for LLM to reference
    numbered = [f"[{i}] {d.page_content[:400]}" for i, d in enumerate(chunks, start=1)]
    block = "\n\n".join(numbered)
    prompt = f"""Given the query and the numbered passages below, output the numbers of the {top_k} most relevant passages in order of relevance (most first). Output only the numbers separated by commas, e.g. 3, 7, 1, 9, 2.

Query: {query}

Passages:
{block}

Top {top_k} numbers (comma-separated):"""

    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    order: list[int] = []
    for part in text.strip().replace(",", " ").split():
        try:
            idx = int(part)
            if 1 <= idx <= len(chunks) and idx not in order:
                order.append(idx)
        except ValueError:
            continue
        if len(order) >= top_k:
            break

    if not order:
        return chunks[:top_k]
    return [chunks[i - 1] for i in order]


def get_available_source_files() -> list[str]:
    """Return distinct source names (document_id or source_file) from Legal KB. Cached."""
    global _available_sources_cache
    if _available_sources_cache is not None:
        return _available_sources_cache

    from app.retrieval import _get_qdrant_client

    client = _get_qdrant_client()
    collection = settings.QDRANT_LEGAL_KNOWLEDGE_COLLECTION
    seen: set[str] = set()
    try:
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points or []:
                payload = p.payload or {}
                doc_id = (
                    payload.get("document_id")
                    or payload.get("source_file")
                    or payload.get("source")
                )
                if doc_id and isinstance(doc_id, str):
                    seen.add(doc_id.strip())
            if offset is None:
                break
    except Exception as e:
        log.warning(
            "get_available_source_files failed", extra={"event": "sources_error", "error": str(e)}
        )
        return []

    _available_sources_cache = sorted(seen)
    log.info(
        "available_sources",
        extra={"event": "sources_loaded", "count": len(_available_sources_cache)},
    )
    return _available_sources_cache
