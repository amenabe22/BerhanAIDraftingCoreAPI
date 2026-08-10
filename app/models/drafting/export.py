"""Request/response models for TipTap export."""

from typing import Literal

from pydantic import BaseModel, Field


class ExportRequest(BaseModel):
    document: dict = Field(..., description="TipTap JSON document ({type: doc, content: [...]})")
    formats: list[Literal["pdf", "docx"]] = Field(
        default_factory=lambda: ["pdf", "docx"],
        description="Formats to generate and upload",
    )
    filename: str | None = Field(
        default=None,
        description="Base filename without extension",
    )


class ExportResponse(BaseModel):
    filename: str
    pdf_url: str | None = None
    docx_url: str | None = None
    keys: dict[str, str] = Field(default_factory=dict)
