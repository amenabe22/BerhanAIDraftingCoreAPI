"""Pydantic models for semantic document edit API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EditRequest(BaseModel):
    """Request body for POST /drafting/edit — mirrors EditingAgent.edit_document args."""

    doc_json: dict[str, Any] = Field(..., description="Full TipTap JSON document")
    instruction: str = Field(..., min_length=1, description="Natural language edit instruction")
    doc_id: str | None = Field(default=None, description="Document UUID for tracking")
    document_language: Literal["en", "am", "om"] | None = Field(
        default=None, description="Output language for generated text (en, am, om)"
    )
    compliance_context: dict[str, Any] | None = Field(
        default=None, description="Optional compliance editor_fix context (phase-2)"
    )


class EditResult(BaseModel):
    """Response body — mirrors EditingAgent.edit_document return dict."""

    document: dict[str, Any]
    diff: str
    operations: list[dict[str, Any]]
    metrics: dict[str, Any] = Field(default_factory=dict)
    operation_id: str
    no_effective_change: bool = False
    no_effective_change_reason: str | None = None
    error: str | None = None
