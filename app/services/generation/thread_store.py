"""In-memory conversation store keyed by thread_id (no Postgres)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any
from uuid import uuid4


class ConversationStatus(str, Enum):
    COLLECTING_INFO = "collecting_info"
    READY = "ready"
    GENERATING = "generating"
    COMPLETED = "completed"
    ERROR = "error"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ThreadState:
    thread_id: str
    user_messages: list[dict[str, Any]] = field(default_factory=list)
    system_responses: list[dict[str, Any]] = field(default_factory=list)
    extracted_requirements: dict[str, Any] = field(default_factory=dict)
    document_structure: dict[str, Any] | None = None
    status: ConversationStatus = ConversationStatus.COLLECTING_INFO
    updated_at: str = field(default_factory=_utcnow)


class ThreadStore:
    """Process-local store for multi-turn generation sessions."""

    def __init__(self) -> None:
        self._threads: dict[str, ThreadState] = {}
        self._lock = Lock()

    def create(self, thread_id: str | None = None) -> ThreadState:
        tid = thread_id or str(uuid4())
        state = ThreadState(thread_id=tid)
        with self._lock:
            self._threads[tid] = state
        return state

    def get(self, thread_id: str) -> ThreadState | None:
        with self._lock:
            return self._threads.get(thread_id)

    def get_or_create(self, thread_id: str | None) -> ThreadState:
        if thread_id:
            existing = self.get(thread_id)
            if existing:
                return existing
            return self.create(thread_id)
        return self.create()

    def add_user_message(self, thread_id: str, message: str) -> None:
        state = self._require(thread_id)
        state.user_messages.append(
            {"role": "user", "content": message, "timestamp": _utcnow()}
        )
        state.updated_at = _utcnow()

    def add_system_response(self, thread_id: str, response: str) -> None:
        state = self._require(thread_id)
        state.system_responses.append(
            {"role": "assistant", "content": response, "timestamp": _utcnow()}
        )
        state.updated_at = _utcnow()

    def update_requirements(self, thread_id: str, requirements: dict[str, Any]) -> None:
        state = self._require(thread_id)
        state.extracted_requirements = {**state.extracted_requirements, **requirements}
        state.updated_at = _utcnow()

    def set_requirements(self, thread_id: str, requirements: dict[str, Any]) -> None:
        state = self._require(thread_id)
        state.extracted_requirements = requirements
        state.updated_at = _utcnow()

    def update_document(self, thread_id: str, document: dict[str, Any]) -> None:
        state = self._require(thread_id)
        state.document_structure = document
        state.updated_at = _utcnow()

    def update_status(self, thread_id: str, status: ConversationStatus) -> None:
        state = self._require(thread_id)
        state.status = status
        state.updated_at = _utcnow()

    def history(self, state: ThreadState) -> list[dict[str, Any]]:
        all_messages: list[tuple[str, dict[str, Any]]] = []
        for msg in state.user_messages:
            if isinstance(msg, dict):
                all_messages.append(("user", msg))
        for msg in state.system_responses:
            if isinstance(msg, dict):
                all_messages.append(("assistant", msg))
        all_messages.sort(key=lambda x: x[1].get("timestamp", ""))
        return [
            {"role": role, "content": msg.get("content", "")}
            for role, msg in all_messages
        ]

    def _require(self, thread_id: str) -> ThreadState:
        state = self.get(thread_id)
        if not state:
            raise ValueError(f"Thread {thread_id} not found")
        return state


# Shared singleton for the process
thread_store = ThreadStore()
