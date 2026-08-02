from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools.retriever import create_retriever_tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import settings
from app.logging_config import get_logger

log = get_logger("retrieval")


class _FlatPayloadQdrantVectorStore(QdrantVectorStore):
    """Maps a flat Qdrant payload to a LangChain Document.

    The content key is tried in priority order so the store works with both
    collection schemas:
      - old schema:  ``text``        (block_id / type / doc_id)
      - new schema:  ``plain_text``  (section_id / section_type / section_title / …)
    Any remaining payload keys are placed in metadata.
    """

    # Ordered list of payload keys to try for page_content.
    # First non-empty value wins.
    _CONTENT_KEY_PRIORITY: tuple[str, ...] = ("text", "plain_text", "page_content", "content")

    @classmethod
    def _document_from_point(
        cls,
        scored_point,
        collection_name: str,
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> Document:
        payload = scored_point.payload or {}

        # Try the configured key first, then the priority fallback list
        page_content = payload.get(content_payload_key, "") or ""
        if not page_content:
            for fallback_key in cls._CONTENT_KEY_PRIORITY:
                if fallback_key != content_payload_key and payload.get(fallback_key):
                    page_content = payload[fallback_key]
                    break

        metadata = {k: v for k, v in payload.items() if k != content_payload_key}
        metadata["_id"] = scored_point.id
        metadata["_collection_name"] = collection_name
        # Ensure common citation fields exist so document_prompt templates never error
        metadata.setdefault("document_id", "")
        metadata.setdefault("item_id", "")
        metadata.setdefault("title", "")
        # doc_blocks old-schema fields
        metadata.setdefault("block_id", "")
        metadata.setdefault("type", "")
        metadata.setdefault("doc_id", "")
        # doc_blocks new-schema fields
        metadata.setdefault("section_id", "")
        metadata.setdefault("section_type", "")
        metadata.setdefault("section_title", "")
        return Document(page_content=page_content, metadata=metadata)


def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        check_compatibility=False,
        verify=settings.QDRANT_VERIFY_SSL,
    )


def _get_embeddings() -> CohereEmbeddings:
    if not settings.COHERE_API_KEY:
        raise ValueError(
            "COHERE_API_KEY is required for embeddings. "
            "Use the same embedding model that was used to index the Qdrant collection (e.g. embed-multilingual-v3.0, 1024 dimensions)."
        )
    return CohereEmbeddings(
        model=settings.COHERE_EMBEDDING_MODEL,
        cohere_api_key=settings.COHERE_API_KEY,
    )


class _LoggingRetriever(BaseRetriever):
    """Wraps a retriever and logs each query and result as JSON."""

    retriever: BaseRetriever

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        log.info(
            "qdrant query",
            extra={"event": "rag_retrieve", "query": query, "phase": "query"},
        )
        docs = self.retriever.invoke(query)
        citations = []
        for d in docs:
            m = getattr(d, "metadata", None) or {}
            citations.append(
                {
                    "document_id": m.get("document_id"),
                    "item_id": m.get("item_id"),
                    "title": (m.get("title") or "")[:200],
                    "content_preview": ((d.page_content or "")[:150]).replace("\n", " "),
                }
            )
        log.info(
            "qdrant results",
            extra={
                "event": "rag_retrieve",
                "phase": "results",
                "query": query,
                "doc_count": len(docs),
                "citations": citations,
            },
        )
        return docs


def _get_cohere_client():
    """Return a Cohere client for reranking. Raises if COHERE_API_KEY is absent."""
    import cohere

    if not settings.COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY is required for chat retrieval reranking")
    return cohere.Client(api_key=settings.COHERE_API_KEY, base_url=settings.COHERE_API_URL)


class _RoutingRetriever(BaseRetriever):
    """Semantic retrieval pipeline for the legal knowledge chat path.

    Each query goes through three stages:
      1. Instrument router — reads ``expect_kb_gap`` and ``forbidden_primary``
         boolean signals for the grounding verifier. The ``query_suffix`` is
         intentionally ignored; it is no longer used to mutate the query.
      2. LLM query expansion — ``expand_chat_query`` produces 2-3 semantically
         diverse sub-queries. Combined with the original query, all are sent to
         Qdrant and merged via RRF.
      3. Cohere cross-encoder rerank — the merged candidate pool is reranked
         against the *original* query (not any enriched variant) to produce the
         final top-k results.

    Fallback chain (each stage degrades gracefully):
      - Expansion fails → single-query RRF path.
      - RRF/search fails → ``self.retriever.invoke(query)`` (legacy dense path).
      - Cohere key absent or rerank fails → top-k from RRF pass is returned.
    """

    retriever: BaseRetriever

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        from app.services.legal.instrument_router import route
        from app.services.drafting.knowledge_retrieval import (
            expand_chat_query,
            search_legal_knowledge,
        )

        # ── Stage 1: routing signals (grounding verifier metadata only) ──────
        decision = route(query)
        log.info(
            "routing_decision",
            extra={
                "event": "instrument_route",
                "original_query": query,
                "expect_kb_gap": decision.expect_kb_gap,
                "forbidden_primary": decision.forbidden_primary,
            },
        )

        # ── Stage 2: LLM query expansion + RRF ───────────────────────────────
        sub_queries = expand_chat_query(query)
        all_queries = [query] + sub_queries  # original always first

        try:
            rrf_top_k = settings.RETRIEVAL_LEGAL_FETCH_K * len(all_queries)
            candidates = search_legal_knowledge(
                all_queries,
                top_k_per_query=settings.RETRIEVAL_LEGAL_FETCH_K,
                rrf_top_k=rrf_top_k,
            )
        except Exception as e:
            log.warning(
                "search_legal_knowledge failed, falling back to dense retriever",
                extra={"event": "instrument_route", "error": str(e)},
            )
            # Dense fallback: skip rerank entirely and return results directly
            return self.retriever.invoke(query)

        if not candidates:
            return candidates

        # ── Stage 3: Cohere cross-encoder rerank ─────────────────────────────
        top_k = settings.RETRIEVAL_LEGAL_RERANK_TOP_K
        try:
            cohere_client = _get_cohere_client()
            documents = [doc.page_content for doc in candidates]
            response = cohere_client.rerank(
                model=settings.COHERE_RERANK_MODEL,
                query=query,  # always the ORIGINAL query, never any enriched variant
                documents=documents,
                top_n=min(top_k, len(documents)),
            )
            reranked = [candidates[item.index] for item in response.results]
            log.info(
                "routing_retrieval_done",
                extra={
                    "event": "instrument_route",
                    "phase": "reranked",
                    "candidate_count": len(candidates),
                    "result_count": len(reranked),
                    "expect_kb_gap": decision.expect_kb_gap,
                },
            )
            return reranked
        except Exception as e:
            log.warning(
                "Cohere rerank failed, returning RRF top results",
                extra={"event": "instrument_route", "error": str(e)},
            )
            result = candidates[:top_k]
            log.info(
                "routing_retrieval_done",
                extra={
                    "event": "instrument_route",
                    "phase": "rrf_fallback",
                    "result_count": len(result),
                    "expect_kb_gap": decision.expect_kb_gap,
                },
            )
            return result


def get_document_blocks_by_doc_id(doc_id: str) -> list[dict]:
    """Load document blocks from doc_blocks by doc_id. Returns list of dicts with block_id, text, type (and doc_id). Supports both payload schemas (text/block_id/type and plain_text/section_id/section_type)."""
    client = _get_qdrant_client()
    doc_filter = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    raw: list[tuple[int | str, dict]] = []  # (sort_key, block_dict)
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.QDRANT_DEFAULT_COLLECTION,
            scroll_filter=doc_filter,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points or []:
            payload = p.payload or {}
            text = payload.get("text") or payload.get("plain_text")
            if text is None:
                continue
            text = str(text).strip()
            if not text:
                continue
            block_id = payload.get("block_id") or payload.get("section_id") or str(p.id)
            block_type = payload.get("type") or payload.get("section_type") or "paragraph"
            sort_key: int | str = payload.get("index", payload.get("order", p.id))
            if isinstance(sort_key, str):
                try:
                    sort_key = int(sort_key)
                except (ValueError, TypeError):
                    sort_key = str(p.id)
            raw.append(
                (
                    sort_key,
                    {
                        "block_id": block_id,
                        "text": text,
                        "type": block_type,
                        "doc_id": doc_id,
                    },
                )
            )
        if offset is None:
            break
    raw.sort(
        key=lambda x: (
            0 if isinstance(x[0], int) else 1,
            x[0] if isinstance(x[0], int) else str(x[0]),
        )
    )
    return [b for _, b in raw]


def get_document_text_by_doc_id(doc_id: str) -> str:
    """Load full document text from doc_blocks by doc_id. Uses get_document_blocks_by_doc_id and joins block text. Returns empty string if none found."""
    blocks = get_document_blocks_by_doc_id(doc_id)
    return "\n\n".join(b.get("text", "") for b in blocks if b.get("text"))


def get_legal_kb_vector_store() -> _FlatPayloadQdrantVectorStore:
    """Return the Legal KB vector store for direct similarity_search (e.g. compliance)."""
    client = _get_qdrant_client()
    embeddings = _get_embeddings()
    return _FlatPayloadQdrantVectorStore(
        client=client,
        collection_name=settings.QDRANT_LEGAL_KNOWLEDGE_COLLECTION,
        embedding=embeddings,
        content_payload_key=settings.QDRANT_CONTENT_PAYLOAD_KEY,
        metadata_payload_key=settings.QDRANT_METADATA_PAYLOAD_KEY,
    )


def get_retriever_tool():
    try:
        vector_store = get_legal_kb_vector_store()
    except Exception as e:
        raise ValueError(
            f"Cannot connect to Qdrant collection '{settings.QDRANT_LEGAL_KNOWLEDGE_COLLECTION}': {e}"
        ) from e
    inner = vector_store.as_retriever(search_kwargs={"k": settings.RETRIEVAL_LEGAL_TOP_K})
    # Stack: plain retriever → logging → routing/query-enrichment
    logging_retriever = _LoggingRetriever(retriever=inner)
    retriever = _RoutingRetriever(retriever=logging_retriever)
    # Include document_id (source/book), item_id, title so the LLM can cite them.
    # The [Source: …] header is parsed by _parse_citations in app/main.py and by
    # the grounding verifier — do not change this format without updating both.
    document_prompt = PromptTemplate.from_template(
        "[Source: {document_id} | Article {item_id} | {title}]\n{page_content}"
    )
    return create_retriever_tool(
        retriever,
        name="search_legal_knowledge",
        description=(
            "Search the Ethiopian legal knowledge base for relevant articles, provisions, and "
            "legal content. ALWAYS call this tool before making any legal statement or conclusion. "
            "Use for ANY legal question — contracts, obligations, property, family law, "
            "inheritance, employment, criminal procedure, business / commercial law, "
            "constitutional rights, tax, and all other areas of Ethiopian law. "
            "On follow-up turns that extend or add legal claims, call this tool again. "
            "Returns document_id (source/instrument name), item_id (article number), "
            "title, and article content — cite these in your answer."
        ),
        document_prompt=document_prompt,
    )


def get_doc_blocks_retriever_tool(doc_id: str):
    """Retriever for the doc_blocks collection filtered to a single document.

    Passes a Qdrant ``must`` filter on ``doc_id`` so only blocks from the
    caller's document are retrieved — other users' documents are never touched.

    Each point has a flat payload:
      text      – the paragraph / clause text
      block_id  – e.g. "b31"
      type      – e.g. "paragraph", "heading"
      doc_id    – UUID of the parent document
    """
    client = _get_qdrant_client()
    embeddings = _get_embeddings()
    try:
        vector_store = _FlatPayloadQdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_DEFAULT_COLLECTION,
            embedding=embeddings,
            content_payload_key="text",
            metadata_payload_key="metadata",
        )
    except Exception as e:
        raise ValueError(
            f"Cannot connect to Qdrant collection '{settings.QDRANT_DEFAULT_COLLECTION}': {e}"
        ) from e
    # Filter strictly to this document — no cross-document leakage
    doc_filter = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    inner = vector_store.as_retriever(
        search_kwargs={"k": settings.RETRIEVAL_DOC_TOP_K, "filter": doc_filter}
    )
    retriever = _LoggingRetriever(retriever=inner)
    document_prompt = PromptTemplate.from_template(
        "[Doc: {doc_id} | Block: {block_id}{section_id} | {section_title}{type}{section_type}]\n{page_content}"
    )
    return create_retriever_tool(
        retriever,
        name="search_user_documents",
        description=(
            "Search the loaded document for relevant clauses, paragraphs, or sections. "
            "Call this tool FIRST for ANY user message that could relate to the document — "
            "including vague requests like 'summarize', 'what is this about', 'explain this', 'what does it say', "
            "as well as specific questions about parties, dates, obligations, terms, conditions, penalties, "
            "scope, purpose, rights, warranties, or any other document content. "
            "When the user's intent is unclear, default to calling this tool before asking for clarification. "
            "Returns doc_id, block_id, block type, and the text content."
        ),
        document_prompt=document_prompt,
    )
