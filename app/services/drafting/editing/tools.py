"""LangChain tools for hybrid doc-agent (chat + semantic edit)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from langgraph.config import get_config, get_stream_writer
from pydantic import BaseModel, Field

from app.services.drafting.editing.agent import SemanticEditAgent

logger = logging.getLogger(__name__)


class ApplyDocumentEditInput(BaseModel):
    instruction: str = Field(
        description=(
            "Natural-language edit instruction describing what to change in the "
            "document (e.g. 'shorten the Definitions section', 'change payment "
            "terms to 30 days'). Use conversation context for follow-ups like "
            "'do that' by restating the concrete edit."
        )
    )


def _run_async(coro):
    """Run an async coroutine from a sync tool context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Already inside an event loop (unlikely for ToolNode sync path) — run in a thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _emit_custom(event: dict[str, Any]) -> None:
    try:
        writer = get_stream_writer()
        if writer is not None:
            writer(event)
    except Exception:
        # Outside a LangGraph stream (unit tests) — ignore.
        pass


def _apply_document_edit_impl(instruction: str) -> str:
    """Apply a semantic edit to the TipTap doc_json from runnable config."""
    instruction = (instruction or "").strip()
    if not instruction:
        return json.dumps(
            {
                "ok": False,
                "no_effective_change": True,
                "reason": "Empty edit instruction.",
            }
        )

    try:
        cfg = get_config().get("configurable", {}) or {}
    except Exception:
        cfg = {}

    doc_json = cfg.get("doc_json")
    if not isinstance(doc_json, dict) or doc_json.get("type") != "doc":
        return json.dumps(
            {
                "ok": False,
                "no_effective_change": True,
                "reason": "No editable document was provided for this turn.",
            }
        )

    doc_id = cfg.get("doc_id")
    document_language = cfg.get("document_language")

    _emit_custom({"type": "edit_started", "instruction": instruction[:200]})

    agent = SemanticEditAgent()
    result = _run_async(
        agent.edit_document(
            doc_json=doc_json,
            instruction=instruction,
            doc_id=doc_id,
            document_language=document_language,
        )
    )

    # Keep config in sync for subsequent tool calls in the same turn.
    if isinstance(result.get("document"), dict) and not result.get("no_effective_change"):
        cfg["doc_json"] = result["document"]

    edit_payload = {
        "type": "edit_result",
        "document": result.get("document"),
        "operations": result.get("operations") or [],
        "diff": result.get("diff") or "",
        "no_effective_change": bool(result.get("no_effective_change")),
        "no_effective_change_reason": result.get("no_effective_change_reason"),
        "operation_id": result.get("operation_id"),
        "error": result.get("error"),
        "metrics": result.get("metrics") or {},
    }
    _emit_custom(edit_payload)

    # Compact tool message for the LLM (full doc goes via SSE to the client).
    summary = {
        "ok": not bool(result.get("no_effective_change") or result.get("error")),
        "no_effective_change": bool(result.get("no_effective_change")),
        "reason": result.get("no_effective_change_reason") or result.get("error"),
        "diff": (result.get("diff") or "")[:800],
        "operations_count": len(result.get("operations") or []),
        "operation_id": result.get("operation_id"),
    }
    return json.dumps(summary)


def create_apply_document_edit_tool() -> StructuredTool:
    """Create the apply_document_edit tool (reads doc_json from LangGraph config)."""
    return StructuredTool.from_function(
        func=_apply_document_edit_impl,
        name="apply_document_edit",
        description=(
            "Apply an edit to the user's open document. Call this when the user "
            "wants the document changed (rewrite, shorten, add, remove, fix, etc.), "
            "including follow-ups that confirm a prior suggestion. Do not call for "
            "pure questions or explanations. Pass a clear, self-contained instruction."
        ),
        args_schema=ApplyDocumentEditInput,
    )
