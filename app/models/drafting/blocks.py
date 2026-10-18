"""Canonical drafting block models (TipTap-free wire format)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class DraftingBlock(BaseModel):
    """Single drafting content block exchanged with clients / WordCraft."""

    id: str = Field(..., min_length=1, description="Stable block UUID string.")
    type: str = Field(..., min_length=1, description="Block type (heading, paragraph, …).")
    index: int = Field(..., ge=0, description="0-based order in the document.")
    level: int | None = Field(
        default=None,
        ge=1,
        le=6,
        description="Heading level when type is heading.",
    )
    html: str = Field(default="", description="HTML representation of the block.")
    plain_text: str = Field(default="", description="Plain-text content of the block.")

    @field_validator("html", "plain_text", mode="before")
    @classmethod
    def _coerce_str(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)


class DraftingDocument(BaseModel):
    """Document envelope for generation / edit / export."""

    blocks: list[DraftingBlock] = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
