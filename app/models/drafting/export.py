"""Request/response models for drafting-block export."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.models.drafting.blocks import DraftingBlock


class ExportRequest(BaseModel):
    blocks: list[DraftingBlock] | list[dict[str, Any]] | None = Field(
        default=None,
        description="Drafting blocks ({id, type, index, level?, html, plain_text})",
    )
    document: dict[str, Any] | None = Field(
        default=None,
        description="Blocks envelope {blocks:[...]} or legacy TipTap {type:doc, content:[...]}",
    )
    formats: list[Literal["pdf", "docx"]] = Field(
        default_factory=lambda: ["pdf", "docx"],
        description="Formats to generate and upload",
    )
    filename: str | None = Field(
        default=None,
        description="Base filename without extension",
    )

    @model_validator(mode="after")
    def _require_blocks_or_document(self) -> ExportRequest:
        if self.blocks is None and self.document is None:
            raise ValueError("Provide blocks or document")
        return self


class ExportResponse(BaseModel):
    filename: str
    pdf_url: str | None = None
    docx_url: str | None = None
    keys: dict[str, str] = Field(default_factory=dict)
