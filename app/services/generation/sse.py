"""SSE event emitter for document generation (replaces WebSocket streaming service)."""

from __future__ import annotations

import asyncio
import json
from typing import Any


class SSEEmitter:
    """Push generation events onto an asyncio.Queue for StreamingResponse."""

    def __init__(self, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        self._queue = queue

    async def emit(self, event: dict[str, Any]) -> None:
        await self._queue.put(event)

    async def close(self) -> None:
        await self._queue.put(None)

    async def analyzing(self, thread_id: str) -> None:
        await self.emit({"type": "analyzing", "thread_id": thread_id})

    async def status(self, message: str) -> None:
        await self.emit({"type": "status", "message": message})

    async def clarification_needed(
        self,
        thread_id: str,
        *,
        questions: list[str],
        response_message: str,
        ready_to_generate: bool = False,
    ) -> None:
        await self.emit(
            {
                "type": "clarification_needed",
                "thread_id": thread_id,
                "questions": questions,
                "response_message": response_message,
                "ready_to_generate": ready_to_generate,
            }
        )

    async def ready_to_generate(self, thread_id: str, response_message: str = "") -> None:
        await self.emit(
            {
                "type": "ready_to_generate",
                "thread_id": thread_id,
                "response_message": response_message,
            }
        )

    async def generation_progress(
        self,
        thread_id: str,
        progress: float,
        stage: str,
        message: str,
    ) -> None:
        await self.emit(
            {
                "type": "generation_progress",
                "thread_id": thread_id,
                "progress": progress,
                "stage": stage,
                "message": message,
            }
        )

    async def node_stream(self, thread_id: str, node: dict[str, Any], node_index: int) -> None:
        """Stream a drafting block (legacy event name node_stream)."""
        await self.emit(
            {
                "type": "node_stream",
                "thread_id": thread_id,
                "block": node,
                "node": node,  # alias for transitional clients
                "node_index": node_index,
            }
        )

    async def token(self, content: str) -> None:
        if content:
            await self.emit({"type": "token", "content": content})

    async def citations(
        self,
        thread_id: str,
        citations: list[dict[str, Any]],
        laws_referenced: list[str],
        message: str = "",
    ) -> None:
        await self.emit(
            {
                "type": "citations",
                "thread_id": thread_id,
                "citations": citations,
                "laws_referenced": laws_referenced,
                "message": message,
            }
        )

    async def document_generated(
        self,
        thread_id: str,
        document: dict[str, Any] | list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        blocks: list[Any] | None = None,
    ) -> None:
        """Emit completed drafting document as blocks (+ optional metadata)."""
        resolved_blocks: list[Any]
        resolved_meta = dict(metadata or {})
        if blocks is not None:
            resolved_blocks = list(blocks)
        elif isinstance(document, list):
            resolved_blocks = document
        elif isinstance(document, dict) and isinstance(document.get("blocks"), list):
            resolved_blocks = document["blocks"]
            if isinstance(document.get("metadata"), dict):
                resolved_meta = {**document["metadata"], **resolved_meta}
        else:
            # Legacy TipTap inbound — convert for wire format
            from app.services.drafting.editing.tiptap import tiptap_to_drafting_blocks

            resolved_blocks = tiptap_to_drafting_blocks(document or {})
            if isinstance(document, dict) and isinstance(document.get("metadata"), dict):
                resolved_meta = {**document["metadata"], **resolved_meta}

        payload: dict[str, Any] = {
            "type": "document_generated",
            "thread_id": thread_id,
            "blocks": resolved_blocks,
            "metadata": resolved_meta,
        }
        # Transitional envelope for callers that still read document
        payload["document"] = {"blocks": resolved_blocks, "metadata": resolved_meta}
        await self.emit(payload)

    async def error(self, message: str) -> None:
        await self.emit({"type": "error", "message": message})


def format_sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"
