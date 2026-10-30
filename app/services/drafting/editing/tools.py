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


class ClarificationOptionInput(BaseModel):
    id: str = Field(description="Stable option id, e.g. shorter")
    label: str = Field(description="Short user-visible label")


class ClarificationQuestionInput(BaseModel):
    id: str = Field(description="Stable question id, e.g. q1")
    prompt: str = Field(description="Short question shown to the user")
    kind: str = Field(
        default="single_choice",
        description="single_choice (radio) or text (free-form)",
    )
    options: list[ClarificationOptionInput] = Field(
        default_factory=list,
        description="2-5 options when kind is single_choice",
    )
    allow_other: bool = Field(
        default=True,
        description="When true, UI shows an Other write-in for single_choice",
    )


class AskClarifyingQuestionsInput(BaseModel):
    response_message: str = Field(
        default="",
        description="Short friendly intro shown above the form (1-2 sentences).",
    )
    questions: list[ClarificationQuestionInput] = Field(
        description="1-3 clarifying questions. Prefer single_choice with options.",
        min_length=1,
        max_length=3,
    )


def _run_async(coro):
    """Run an async coroutine from a sync ToolNode safely.

    Doc-agent graphs stream via ``asyncio.to_thread``; tools therefore run on a
    worker thread that must not nest ``asyncio.run`` against a polluted/closed
    loop. Always isolate the edit agent in a fresh thread + ``asyncio.run``.
    """
    import concurrent.futures

    def _runner():
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_runner).result()


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
            "Apply an edit to the user's open document. Call when the user wants "
            "the document changed: rewrite, shorten, expand, enrich, clarify, "
            "strengthen, add, remove, fix, more detail / legal-technical content, "
            "including follow-ups that confirm a prior suggestion. Do not call only "
            "for pure informational questions with no document-change intent "
            "(e.g. what does this mean?). Pass a clear, self-contained instruction."
        ),
        args_schema=ApplyDocumentEditInput,
    )


def _ask_clarifying_questions_impl(
    response_message: str = "",
    questions: list[Any] | None = None,
) -> str:
    """Emit structured clarification_needed for the FE form UI."""
    from app.services.generation.clarification_questions import normalize_questions

    raw_questions: list[Any] = []
    for q in questions or []:
        if hasattr(q, "model_dump"):
            raw_questions.append(q.model_dump())
        elif isinstance(q, dict):
            raw_questions.append(q)
        else:
            raw_questions.append(q)

    normalized = normalize_questions(raw_questions, max_questions=3)
    if not normalized:
        return json.dumps(
            {
                "ok": False,
                "reason": "No valid clarifying questions were provided.",
            }
        )

    intro = (response_message or "").strip()
    _emit_custom(
        {
            "type": "clarification_needed",
            "response_message": intro,
            "response": intro,
            "questions": normalized,
            "ready_to_generate": False,
        }
    )
    return json.dumps(
        {
            "ok": True,
            "awaiting_user": True,
            "questions_count": len(normalized),
            "message": (
                "Clarifying questions were shown to the user as a form. "
                "Wait for their next message with answers — do not invent answers."
            ),
        }
    )


def create_ask_clarifying_questions_tool() -> StructuredTool:
    """Create ask_clarifying_questions — structured radio/text form for the client."""
    return StructuredTool.from_function(
        func=_ask_clarifying_questions_impl,
        name="ask_clarifying_questions",
        description=(
            "Ask the user 1-3 clarifying questions with selectable options when you "
            "need a choice before editing or answering. Prefer single_choice with "
            "2-5 short options and allow_other true. Use text kind only for open "
            "answers (names, amounts). Do NOT ask clarifying questions as plain "
            "prose in your reply when options are natural — call this tool instead. "
            "Do not call this when the user already asked to apply a clear edit."
        ),
        args_schema=AskClarifyingQuestionsInput,
    )
