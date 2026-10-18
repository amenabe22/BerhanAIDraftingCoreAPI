"""Pydantic models for semantic document edit API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.models.drafting.blocks import DraftingBlock


class EditRequest(BaseModel):
    """Request body for POST /drafting/edit — mirrors EditingAgent.edit_document args."""

    blocks: list[DraftingBlock] | list[dict[str, Any]] | None = Field(
        default=None,
        description="Drafting blocks ({id, type, index, level?, html, plain_text})",
    )
    doc_json: dict[str, Any] | None = Field(
        default=None,
        description="Legacy TipTap JSON or blocks envelope {blocks:[...]}",
    )
    instruction: str = Field(..., min_length=1, description="Natural language edit instruction")
    doc_id: str | None = Field(default=None, description="Document UUID for tracking")
    document_language: Literal["en", "am", "om"] | None = Field(
        default=None, description="Output language for generated text (en, am, om)"
    )
    compliance_context: dict[str, Any] | None = Field(
        default=None, description="Optional compliance editor_fix context (phase-2)"
    )

    @model_validator(mode="after")
    def _require_blocks_or_doc(self) -> EditRequest:
        if self.blocks is None and self.doc_json is None:
            raise ValueError("Provide blocks or doc_json")
        return self


class EditResult(BaseModel):
    """Response body — mirrors EditingAgent.edit_document return dict."""

    document: dict[str, Any] = Field(
        description="Patched drafting document envelope {blocks:[...], metadata?:...}"
    )
    blocks: list[dict[str, Any]] | None = Field(
        default=None, description="Patched drafting blocks (convenience mirror)"
    )
    diff: str
    operations: list[dict[str, Any]]
    metrics: dict[str, Any] = Field(default_factory=dict)
    operation_id: str
    no_effective_change: bool = False
    no_effective_change_reason: str | None = None
    error: str | None = None
