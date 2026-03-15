"""Main ingestion service for processing and storing documents."""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from app.services.ingestion.html_parser import HTMLParser
from app.services.ingestion.chunker import Chunker
from app.vector_store.embeddings import get_embedding
from app.vector_store.qdrant_client import (
    ensure_collection,
    upsert_block_embedding,
    get_collection_name,
)
from app.database.connection import get_db
from app.models.document import Document, DocumentStatus
from app.models.block import Block
from app.utils.logger import get_logger
from app.utils.section_utils import normalize_text, generate_synonyms
import json

logger = get_logger(__name__)


class IngestionService:
    """Service for ingesting documents into the system."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize ingestion service.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.html_parser = HTMLParser()
        self.chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    async def ingest_document(
        self,
        doc_id: UUID,
        sections: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a document with sections.

        Args:
            doc_id: Document UUID
            sections: List of section dictionaries with keys: section_id, section_type, html

        Returns:
            Dictionary with ingestion results: status, doc_id, collection_name, chunks_ingested, sections_processed
        """
        try:
            # Parse sections from structured input
            parsed_sections = self.html_parser.extract_sections_from_structured(sections)

            if not parsed_sections:
                raise ValueError("No valid sections found in input")

            # Ensure Qdrant collection exists and get collection name
            coll_name = await ensure_collection(doc_id=doc_id, collection_name=collection_name)
            
            # Ensure document record exists
            document = await self._ensure_document(doc_id, coll_name)

            # Update status to processing
            await self._update_document_status(doc_id, DocumentStatus.PROCESSING)

            # Build section_index: {normalized_title: section_id}
            section_index = {}
            for section in parsed_sections:
                section_id = section.get("section_id")
                section_title = section.get("section_title")
                if section_id and section_title:
                    norm_title = normalize_text(section_title)
                    section_index[norm_title] = section_id
            
            # Store section_index in document record (gracefully handle if column doesn't exist yet)
            if section_index:
                try:
                    async for db in get_db():
                        await db.execute(
                            "UPDATE documents SET section_index = $1::jsonb, updated_at = $2 WHERE id = $3",
                            json.dumps(section_index),
                            datetime.utcnow(),
                            doc_id,
                        )
                        break
                    logger.info(f"[INGEST] section_index created: {doc_id} {section_index}")
                except Exception as e:
                    # Gracefully handle if section_index column doesn't exist (migration not run yet)
                    if "section_index" in str(e).lower() or "does not exist" in str(e).lower():
                        logger.warning(f"[INGEST] section_index column not found. Run migration 003_add_section_index_to_documents. Continuing without section_index.")
                    else:
                        # Re-raise if it's a different error
                        raise

            # Chunk sections
            chunks = self.chunker.chunk_sections(parsed_sections)

            # Process chunks: generate embeddings and store
            chunks_ingested = 0
            global_chunk_index = 0
            async for db in get_db():
                for chunk in chunks:
                    try:
                        # Build enhanced text for embedding that includes section context
                        # This improves searchability, especially for minimal/empty sections
                        section_id = chunk.get("section_id", "")
                        section_type = chunk.get("section_type", "")
                        section_title = chunk.get("section_title")  # From parsed section
                        plain_text = chunk.get("plain_text", "")
                        
                        # Build embedding text with section context
                        embedding_parts = []
                        
                        # Include section title if available (helps with section-specific searches)
                        if section_title:
                            embedding_parts.append(section_title)
                        
                        # Include section_id for better cross-language matching
                        # Section IDs are always in English, so they help bridge language gaps
                        if section_id:
                            embedding_parts.append(f"Section: {section_id}")
                        
                        # Include section_type for context
                        if section_type:
                            embedding_parts.append(f"Type: {section_type}")
                        
                        # Add the actual content
                        if plain_text:
                            embedding_parts.append(plain_text)
                        elif not section_title:
                            # For minimal/empty sections, at least include section_id
                            # This ensures they can be found even with no content
                            embedding_parts.append(f"Section {section_id}")
                        
                        embedding_text = " ".join(embedding_parts)
                        
                        # Generate embedding with enhanced text
                        embedding = await get_embedding(embedding_text)

                        # Generate section synonyms
                        section_synonyms = generate_synonyms(section_title) if section_title else []
                        
                        # Prepare payload for Qdrant (include section_title and section_synonyms for metadata filtering)
                        payload = {
                            "doc_id": str(doc_id),
                            "section_id": section_id,
                            "section_type": section_type,
                            "section_title": section_title,  # Added for metadata filtering
                            "section_synonyms": section_synonyms,  # NEW: synonyms for better matching
                            "chunk_index": chunk["chunk_index"],
                            "total_chunks": chunk["total_chunks"],
                            "section_index": chunk["section_index"],
                            "html": chunk["html"],
                            "plain_text": plain_text,
                        }

                        # Upsert to Qdrant
                        chunk_uuid = UUID(chunk["chunk_id"])
                        await upsert_block_embedding(
                            doc_id=doc_id,
                            block_id=chunk_uuid,
                            embedding=embedding,
                            payload=payload,
                            collection_name=collection_name,
                        )

                        # Store block record in Postgres with sequential index
                        await self._store_block(
                            db=db,
                            block_id=chunk_uuid,
                            doc_id=doc_id,
                            index=global_chunk_index,
                            block_type=chunk["section_type"],
                            html=chunk["html"],
                            plain_text=chunk["plain_text"],
                            embedding_id=str(chunk_uuid),
                        )

                        chunks_ingested += 1
                        global_chunk_index += 1

                    except Exception as e:
                        logger.error(f"Failed to process chunk {chunk.get('chunk_id')}: {e}", exc_info=True)
                        # Continue with other chunks
                        continue

                break  # Exit after first connection

            # Update document status to completed
            await self._update_document_status(doc_id, DocumentStatus.COMPLETED)

            logger.info(
                f"Successfully ingested document {doc_id}: {chunks_ingested} chunks from {len(parsed_sections)} sections"
            )

            return {
                "status": "success",
                "doc_id": str(doc_id),
                "collection_name": coll_name,
                "chunks_ingested": chunks_ingested,
                "sections_processed": len(parsed_sections),
            }

        except Exception as e:
            from app.utils.error_handler import is_ssl_error
            if is_ssl_error(e):
                logger.error(f"Failed to ingest document {doc_id}: SSL connection error")
            else:
                logger.error(f"Failed to ingest document {doc_id}: {e}", exc_info=True)
            # Update document status to failed
            try:
                await self._update_document_status(doc_id, DocumentStatus.FAILED)
            except Exception:
                pass  # Ignore errors when updating status
            raise

    async def _ensure_document(self, doc_id: UUID, collection_name: str) -> Document:
        """Ensure document record exists, create if not."""
        async for db in get_db():
            # Check if document exists
            row = await db.fetchrow(
                "SELECT id, collection_name, status FROM documents WHERE id = $1",
                doc_id,
            )

            if row:
                # Update collection_name if different
                if row["collection_name"] != collection_name:
                    await db.execute(
                        "UPDATE documents SET collection_name = $1, updated_at = $2 WHERE id = $3",
                        collection_name,
                        datetime.utcnow(),
                        doc_id,
                    )
                document = Document(
                    id=row["id"],
                    collection_name=row["collection_name"],
                    status=DocumentStatus(row["status"]),
                )
                break
            else:
                # Create new document
                await db.execute(
                    """
                    INSERT INTO documents (id, collection_name, status, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    doc_id,
                    collection_name,
                    DocumentStatus.PENDING.value,
                    datetime.utcnow(),
                    datetime.utcnow(),
                )

                document = Document(
                    id=doc_id,
                    collection_name=collection_name,
                    status=DocumentStatus.PENDING,
                )
                break

        return document

    async def _update_document_status(self, doc_id: UUID, status: DocumentStatus) -> None:
        """Update document status."""
        async for db in get_db():
            await db.execute(
                "UPDATE documents SET status = $1, updated_at = $2 WHERE id = $3",
                status.value,
                datetime.utcnow(),
                doc_id,
            )
            break

    async def _store_block(
        self,
        db,
        block_id: UUID,
        doc_id: UUID,
        index: int,
        block_type: str,
        html: str,
        plain_text: str,
        embedding_id: str,
    ) -> None:
        """Store block record in Postgres."""
        await db.execute(
            """
            INSERT INTO blocks (
                id, doc_id, index, type, html, plain_text, embedding_id,
                version, last_modified_by, last_modified_at, schema_valid
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (id) DO UPDATE SET
                html = EXCLUDED.html,
                plain_text = EXCLUDED.plain_text,
                embedding_id = EXCLUDED.embedding_id,
                version = blocks.version + 1,
                last_modified_at = EXCLUDED.last_modified_at
            """,
            block_id,
            doc_id,
            index,
            block_type,
            html,
            plain_text,
            embedding_id,
            1,  # version
            "ingestion_service",  # last_modified_by
            datetime.utcnow(),
            True,  # schema_valid
        )

    async def re_embed_block(
        self,
        doc_id: UUID,
        block_id: UUID,
        new_html: str,
        new_plain_text: str,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Re-embed a block after edit, preserving metadata.
        
        Args:
            doc_id: Document UUID
            block_id: Block UUID
            new_html: Updated HTML content
            new_plain_text: Updated plain text content
            collection_name: Optional collection name
        """
        try:
            # Get existing block metadata from Qdrant to preserve section info
            from app.vector_store.qdrant_client import get_qdrant_client, get_collection_name
            client = await get_qdrant_client()
            coll_name = get_collection_name(doc_id=doc_id, collection_name=collection_name)
            
            # Retrieve existing point to get metadata
            try:
                points = await client.retrieve(
                    collection_name=coll_name,
                    ids=[str(block_id)],
                )
                
                existing_payload = {}
                if points and len(points) > 0:
                    # Handle both list and single point response
                    point = points[0] if isinstance(points, list) else points
                    if hasattr(point, 'payload'):
                        existing_payload = point.payload or {}
                    elif isinstance(point, dict):
                        existing_payload = point.get('payload', {})
            except Exception as e:
                logger.warning(f"Could not retrieve existing block metadata: {e}, using defaults")
                existing_payload = {}
            
            # Preserve section metadata
            section_id = existing_payload.get("section_id", "")
            section_type = existing_payload.get("section_type", "")
            section_title = existing_payload.get("section_title")
            section_synonyms = existing_payload.get("section_synonyms", [])
            
            # Build embedding text with section context (same as ingestion)
            embedding_parts = []
            if section_title:
                embedding_parts.append(section_title)
            if section_id:
                embedding_parts.append(f"Section: {section_id}")
            if section_type:
                embedding_parts.append(f"Type: {section_type}")
            if new_plain_text:
                embedding_parts.append(new_plain_text)
            elif not section_title:
                embedding_parts.append(f"Section {section_id}")
            
            embedding_text = " ".join(embedding_parts)
            
            # Generate new embedding
            embedding = await get_embedding(embedding_text)
            
            # Prepare payload (preserve all metadata)
            payload = {
                "doc_id": str(doc_id),
                "section_id": section_id,
                "section_type": section_type,
                "section_title": section_title,
                "section_synonyms": section_synonyms,  # Preserve synonyms
                "chunk_index": existing_payload.get("chunk_index", 0),
                "total_chunks": existing_payload.get("total_chunks", 1),
                "section_index": existing_payload.get("section_index", 0),
                "html": new_html,
                "plain_text": new_plain_text,
            }
            
            # Upsert to Qdrant
            await upsert_block_embedding(
                doc_id=doc_id,
                block_id=block_id,
                embedding=embedding,
                payload=payload,
                collection_name=collection_name,
            )
            
            # Update block in Postgres
            async for db in get_db():
                await db.execute(
                    """
                    UPDATE blocks 
                    SET html = $1, 
                        plain_text = $2, 
                        version = version + 1,
                        last_modified_at = $3,
                        last_modified_by = $4
                    WHERE id = $5 AND doc_id = $6
                    """,
                    new_html,
                    new_plain_text,
                    datetime.utcnow(),
                    "llm_agent_v1",
                    block_id,
                    doc_id,
                )
                break
            
            logger.info(f"[PERSIST] re-embed_complete block_id={block_id}")
            
        except Exception as e:
            logger.error(f"Failed to re-embed block {block_id}: {e}", exc_info=True)
            raise
