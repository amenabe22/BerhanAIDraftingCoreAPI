"""Request models for document generation SSE endpoints."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from app.llm import SupportedModel
from app.services.generation.language import coerce_language_input


class Language(str, Enum):
    amharic = "am"
    english = "en"
    oromo = "om"


# Accept aliases like amh/amharic/oro before enum validation
LanguageField = Annotated[Language | None, BeforeValidator(coerce_language_input)]


class GenerateRequest(BaseModel):
    """Legacy conversational generation request."""

    message: str
    thread_id: str | None = None
    language: LanguageField = None
    model: SupportedModel | None = None
    enable_reasoning: bool = False
    num_pages: int | None = Field(default=None, ge=1, le=20)
    document_type: str | None = None
    file_url: str | None = None
    action: Literal["start", "message", "finalize"] = "start"


class Party(BaseModel):
    """Party entry per drafting-metadata.schema.json (name + role required)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)

    @field_validator("name", "role")
    @classmethod
    def strip_nonempty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("must be a non-empty string")
        return v


class DraftMetadata(BaseModel):
    """Canonical drafting metadata (JSON Schema draft 2020-12, camelCase).

    See ``app/schemas/drafting_metadata.schema.json``.
    Required: ``title``, ``parties``, ``governingLaw``.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "title": "Mutual Non-Disclosure Agreement",
                    "parties": [
                        {"name": "Acme PLC", "role": "Disclosing Party"},
                        {"name": "Beta LLC", "role": "Receiving Party"},
                    ],
                    "governingLaw": "Ethiopian law",
                    "jurisdiction": "Ethiopian courts",
                    "numPages": 2,
                    "purpose": "Protect confidential business information",
                }
            ]
        },
    )

    title: str = Field(..., min_length=1)
    parties: list[Party] = Field(..., min_length=1)
    governing_law: str = Field(..., min_length=1, alias="governingLaw")
    num_pages: int | None = Field(default=None, ge=1, le=20, alias="numPages")
    purpose: str | None = None
    jurisdiction: str | None = None

    @field_validator("title", "governing_law")
    @classmethod
    def strip_required_strings(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("must be a non-empty string")
        return v

    @field_validator("purpose", "jurisdiction")
    @classmethod
    def strip_optional_strings(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None


class StructuredGenerateRequest(BaseModel):
    """One-shot agent drafting payload (no clarification loop).

    ``metadata`` must follow ``drafting_metadata.schema.json`` (camelCase).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "doc_type": "NDA",
                    "type": ["pdf", "docx"],
                    "language": "am",
                    "instructions": "Keep it practical under Ethiopian law.",
                    "metadata": {
                        "title": "Mutual Non-Disclosure Agreement",
                        "parties": [
                            {"name": "Acme PLC", "role": "Disclosing Party"},
                            {"name": "Beta LLC", "role": "Receiving Party"},
                        ],
                        "governingLaw": "Ethiopian law",
                        "jurisdiction": "Ethiopian courts",
                        "numPages": 2,
                        "purpose": "Protect confidential business information",
                    },
                    "filename": "nda-acme-beta",
                }
            ]
        }
    )

    doc_type: str = Field(..., min_length=1, description="e.g. NDA, Contract, Employment Contract")
    type: list[Literal["pdf", "docx"]] = Field(
        ...,
        min_length=1,
        description="Output formats to upload to Contabo after generation",
    )
    metadata: DraftMetadata = Field(
        ...,
        description=(
            "Drafting metadata JSON object (camelCase). "
            "Required keys: title, parties, governingLaw. "
            "Schema: app/schemas/drafting_metadata.schema.json"
        ),
    )
    instructions: str | None = Field(
        default=None,
        description="Free-text agent brief (optional; facts belong in metadata)",
    )
    language: LanguageField = Field(
        default=Language.english,
        description='Document language: "en" | "am" | "om" (aliases: amh, oro, oromo, afaan). Pins output language.',
    )
    model: SupportedModel | None = None
    enable_reasoning: bool = False
    file_url: str | None = None
    filename: str | None = None
    thread_id: str | None = None

    @field_validator("doc_type")
    @classmethod
    def strip_doc_type(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("doc_type is required")
        return v

    @field_validator("type")
    @classmethod
    def normalize_formats(cls, v: list[str]) -> list[str]:
        allowed = {"pdf", "docx"}
        out: list[str] = []
        for item in v:
            if item not in allowed:
                raise ValueError('type items must be "pdf" or "docx"')
            if item not in out:
                out.append(item)
        if not out:
            raise ValueError("type must include at least one of pdf or docx")
        return out
