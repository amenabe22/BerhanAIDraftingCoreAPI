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


def get_retriever_tool():
    client = _get_qdrant_client()
    embeddings = _get_embeddings()
    try:
        vector_store = _FlatPayloadQdrantVectorStore(
            client=client,
            collection_name=settings.QDRANT_LEGAL_KNOWLEDGE_COLLECTION,
            embedding=embeddings,
            content_payload_key=settings.QDRANT_CONTENT_PAYLOAD_KEY,
            metadata_payload_key=settings.QDRANT_METADATA_PAYLOAD_KEY,
        )
    except Exception as e:
        raise ValueError(
            f"Cannot connect to Qdrant collection '{settings.QDRANT_LEGAL_KNOWLEDGE_COLLECTION}': {e}"
        ) from e
    inner = vector_store.as_retriever(search_kwargs={"k": settings.RETRIEVAL_LEGAL_TOP_K})
    retriever = _LoggingRetriever(retriever=inner)
    # Include document_id (source/book), item_id, title so the LLM can cite them
    document_prompt = PromptTemplate.from_template(
        "[Source: {document_id} | Article {item_id} | {title}]\n{page_content}"
    )
    return create_retriever_tool(
        retriever,
        name="search_legal_knowledge",
        description="Search the legal knowledge base for relevant articles, provisions, and legal content. Use this for ANY legal question — including contracts, obligations, property, family law, inheritance, employment, criminal procedure, business law, constitutional rights, and all other areas of Ethiopian law. Returns document_id (source name), item_id (article number), title, and article content.",
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
