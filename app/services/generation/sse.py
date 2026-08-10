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
        await self.emit(
            {
                "type": "node_stream",
                "thread_id": thread_id,
                "node": node,
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
        document: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.emit(
            {
                "type": "document_generated",
                "thread_id": thread_id,
                "document": document,
                "metadata": metadata or {},
            }
        )

    async def error(self, message: str) -> None:
        await self.emit({"type": "error", "message": message})


def format_sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"
