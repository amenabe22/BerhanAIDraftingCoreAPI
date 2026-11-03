"""Federal Supreme Court cassation (ሰበር) retrieval tools for the legal agents.

Two tools over the cassation collection, alongside the statute tool
(``search_legal_knowledge``) in :mod:`app.retrieval`:

- ``search_cassation_decisions`` — dense recall → Cohere rerank → per-case
  diversity, so one long decision cannot crowd out every other holding.
- ``get_cassation_case`` — exact lookup by case number (headnote, holding and
  the body passages most relevant to the requested focus).

Every passage is emitted under a ``[Cassation: <document_id> | Case <n> | <title>]``
header; ``_parse_citations`` in :mod:`app.main` and the grounding verifier
depend on that exact shape.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import settings
from app.logging_config import get_logger
from app.retrieval import (
    _FlatPayloadQdrantVectorStore,
    _get_cohere_client,
    _get_embeddings,
    _get_qdrant_client,
)

log = get_logger("retrieval.cassation")

_SECTION_ORDER = {"headnote": 0, "holding": 1, "body": 2}
_CASE_NUMBER_RE = re.compile(r"\d{3,7}")

NO_CASSATION_HITS = (
    "NO_CASSATION_PASSAGES: No cassation decision matched this query. Rewrite it with the "
    "legal issue in Amharic legal terms (e.g. parties' roles, the right or act in dispute) "
    "and search again once, or answer from statutes and say no ሰበር holding was found."
)


def get_cassation_kb_vector_store() -> _FlatPayloadQdrantVectorStore:
    return _FlatPayloadQdrantVectorStore(
        client=_get_qdrant_client(),
        collection_name=settings.QDRANT_CASSATION_COLLECTION,
        embedding=_get_embeddings(),
        content_payload_key=settings.QDRANT_CONTENT_PAYLOAD_KEY,
        metadata_payload_key=settings.QDRANT_METADATA_PAYLOAD_KEY,
    )


def _section(meta: dict) -> str:
    return str(meta.get("section") or meta.get("part") or "body").lower()


def format_cassation_block(doc: Document) -> str:
    m = doc.metadata or {}
    header = (
        f"[Cassation: {m.get('document_id') or ''} | Case {m.get('case_number') or ''} | "
        f"{m.get('title') or ''}]"
    )
    subject = str(m.get("subject") or "").strip()
    section = _section(m)
    label = f"({section}{' · ' + subject if subject else ''})"
    return f"{header}\n{label}\n{(doc.page_content or '').strip()}"


def _rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    if not docs:
        return []
    try:
        client = _get_cohere_client()
        response = client.rerank(
            model=settings.COHERE_RERANK_MODEL,
            query=query,
            documents=[(d.page_content or "")[:2500] for d in docs],
            top_n=min(top_n, len(docs)),
        )
        out: list[Document] = []
        for item in response.results:
            d = docs[item.index]
            d.metadata = {**(d.metadata or {}), "_rerank_score": float(item.relevance_score)}
            out.append(d)
        return out
    except Exception as e:
        log.warning(
            "cassation rerank failed; using vector order",
            extra={"event": "cassation_rerank_error", "error": str(e)},
        )
        return docs[:top_n]


def _diversify_by_case(
    ranked: list[Document], *, max_cases: int, per_case: int, min_score: float
) -> list[Document]:
    """Keep the best passages per case so several holdings reach the model."""
    kept: list[Document] = []
    per: dict[str, int] = {}
    for d in ranked:
        m = d.metadata or {}
        score = m.get("_rerank_score")
        if score is not None and float(score) < min_score:
            continue
        case = str(m.get("case_number") or m.get("document_id") or "")
        if case not in per and len(per) >= max_cases:
            continue
        if per.get(case, 0) >= per_case:
            continue
        per[case] = per.get(case, 0) + 1
        kept.append(d)
    return kept


def search_cassation(query: str) -> list[Document]:
    store = get_cassation_kb_vector_store()
    candidates = store.similarity_search(query, k=settings.RETRIEVAL_CASSATION_FETCH_K)
    ranked = _rerank(query, candidates, settings.RETRIEVAL_CASSATION_FETCH_K)
    selected = _diversify_by_case(
        ranked,
        max_cases=settings.RETRIEVAL_CASSATION_MAX_CASES,
        per_case=settings.RETRIEVAL_CASSATION_PER_CASE,
        min_score=settings.RETRIEVAL_CASSATION_MIN_RERANK_SCORE,
    )
    log.info(
        "cassation search",
        extra={
            "event": "cassation_search",
            "query": query,
            "candidate_count": len(candidates),
            "result_count": len(selected),
            "cases": [str((d.metadata or {}).get("case_number") or "") for d in selected],
        },
    )
    return selected


def _scroll_case(case_number: str) -> list[Document]:
    client = _get_qdrant_client()
    case_filter = Filter(must=[FieldCondition(key="case_number", match=MatchValue(value=case_number))])
    docs: list[Document] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.QDRANT_CASSATION_COLLECTION,
            scroll_filter=case_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points or []:
            payload = dict(p.payload or {})
            content = str(payload.pop(settings.QDRANT_CONTENT_PAYLOAD_KEY, "") or "").strip()
            if content:
                docs.append(Document(page_content=content, metadata=payload))
        if offset is None:
            break
    return docs


def _chunk_index(doc: Document) -> int:
    try:
        return int((doc.metadata or {}).get("chunk_index") or 0)
    except (TypeError, ValueError):
        return 0


def get_cassation_case(case_number: str, focus: str = "") -> list[Document]:
    """Headnote + holding + the body passages most relevant to *focus* (or to the holding)."""
    match = _CASE_NUMBER_RE.search(case_number or "")
    if not match:
        return []
    docs = _scroll_case(match.group(0))
    if not docs:
        return []
    summary = [d for d in docs if _section(d.metadata) in ("headnote", "holding")]
    body = [d for d in docs if _section(d.metadata) not in ("headnote", "holding")]
    anchor = (focus or "").strip() or "\n".join(d.page_content for d in summary)[:2000]
    if anchor and len(body) > settings.RETRIEVAL_CASSATION_CASE_BODY_CHUNKS:
        body = _rerank(anchor, body, settings.RETRIEVAL_CASSATION_CASE_BODY_CHUNKS)
    else:
        body = body[: settings.RETRIEVAL_CASSATION_CASE_BODY_CHUNKS]
    summary.sort(key=lambda d: _SECTION_ORDER.get(_section(d.metadata), 9))
    body.sort(key=_chunk_index)
    return summary + body


def _render(docs: list[Document], empty_notice: str) -> str:
    if not docs:
        return empty_notice
    return "\n\n".join(format_cassation_block(d) for d in docs)


class _SearchCassationArgs(BaseModel):
    query: str = Field(
        description=(
            "Self-contained description of the legal issue (who did what, which right or act is "
            "disputed). Amharic legal terms retrieve best because decisions are in Amharic."
        )
    )


class _GetCassationCaseArgs(BaseModel):
    case_number: str = Field(description="ሰበር case / file number, e.g. '186626'.")
    focus: str = Field(
        default="",
        description="Optional issue to focus on within the decision (e.g. the user's question).",
    )


def _run_search(query: str) -> str:
    try:
        return _render(search_cassation(query), NO_CASSATION_HITS)
    except Exception as e:
        log.warning("cassation search failed", extra={"event": "cassation_search_error", "error": str(e)})
        return "CASSATION_SEARCH_UNAVAILABLE: answer from statutes and say case law could not be searched."


def _run_get_case(case_number: str, focus: str = "") -> str:
    try:
        docs = get_cassation_case(case_number, focus)
    except Exception as e:
        log.warning("cassation case lookup failed", extra={"event": "cassation_case_error", "error": str(e)})
        return "CASSATION_LOOKUP_UNAVAILABLE: the case could not be loaded right now."
    return _render(
        docs,
        f"CASE_NOT_FOUND: no cassation decision with number {case_number} is in the collection. "
        "Use search_cassation_decisions with the legal issue instead.",
    )


def get_cassation_search_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_run_search,
        name="search_cassation_decisions",
        args_schema=_SearchCassationArgs,
        description=(
            "Search Ethiopian Federal Supreme Court cassation (ሰበር) decisions — binding "
            "interpretations of law by the court. Use for questions about court decisions, "
            "case law, precedent, how courts have ruled, or concrete fact patterns / disputes, "
            "and alongside search_legal_knowledge when both the statute and its judicial "
            "interpretation matter. Returns [Cassation: …] passages (headnote, holding, body)."
        ),
    )


def get_cassation_case_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_run_get_case,
        name="get_cassation_case",
        args_schema=_GetCassationCaseArgs,
        description=(
            "Load a specific cassation (ሰበር) decision by its case number — use whenever the user "
            "names a case number or the conversation is about a specific decision. Returns the "
            "headnote, holding and the most relevant reasoning passages as [Cassation: …] blocks."
        ),
    )


def get_cassation_tools() -> list[StructuredTool]:
    return [get_cassation_search_tool(), get_cassation_case_tool()]
