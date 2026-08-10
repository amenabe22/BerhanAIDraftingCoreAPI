"""Legal KB retrieval for document generation using the core service Qdrant stack."""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.logging_config import get_logger

log = get_logger("generation_rag")


class GenerationRAGService:
    """Retrieve legal knowledge chunks for generation prompts."""

    async def retrieve_relevant_knowledge(
        self,
        requirements: dict[str, Any],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not settings.ENABLE_GENERATION_RAG:
            return []

        k = top_k or settings.GENERATION_KNOWLEDGE_TOP_K
        document_type = requirements.get("document_type", "contract")
        query_parts = [str(document_type)]
        for key in ("title", "summary", "parties", "purpose"):
            val = requirements.get(key)
            if val:
                query_parts.append(str(val)[:200])
        query = " ".join(query_parts).strip() or "Ethiopian legal contract"
        log.info("generation_rag_query", extra={"query": query[:120], "top_k": k})

        try:
            from app.retrieval import get_legal_kb_vector_store

            store = get_legal_kb_vector_store()
            docs = store.similarity_search(query, k=k)
        except Exception as exc:
            log.warning("generation_rag_failed", extra={"error": str(exc)})
            return []

        chunks: list[dict[str, Any]] = []
        for doc in docs:
            meta = dict(getattr(doc, "metadata", None) or {})
            chunks.append(
                {
                    "page_content": getattr(doc, "page_content", "") or "",
                    "metadata": meta,
                }
            )
        return chunks

    def extract_citations(self, knowledge_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for chunk in knowledge_chunks:
            meta = chunk.get("metadata") or {}
            source = (
                meta.get("document_id")
                or meta.get("source")
                or meta.get("title")
                or "legal_knowledge"
            )
            citations.append(
                {
                    "source": source,
                    "item_id": meta.get("item_id") or meta.get("article"),
                    "title": meta.get("title"),
                    "preview": (chunk.get("page_content") or "")[:240],
                }
            )
        return citations

    def extract_law_references(self, knowledge_chunks: list[dict[str, Any]]) -> list[str]:
        refs: list[str] = []
        seen: set[str] = set()
        for chunk in knowledge_chunks:
            meta = chunk.get("metadata") or {}
            for key in ("document_id", "source", "title"):
                val = meta.get(key)
                if val and str(val) not in seen:
                    seen.add(str(val))
                    refs.append(str(val))
        return refs

    @staticmethod
    def format_knowledge_for_prompt(knowledge_chunks: list[dict[str, Any]]) -> str:
        if not knowledge_chunks:
            return ""
        parts: list[str] = []
        for i, chunk in enumerate(knowledge_chunks, 1):
            meta = chunk.get("metadata") or {}
            header = meta.get("document_id") or meta.get("title") or f"Source {i}"
            item = meta.get("item_id")
            label = f"{header}" + (f" | Article {item}" if item else "")
            parts.append(f"[{label}]\n{chunk.get('page_content', '')}")
        return "\n\n".join(parts)
