"""Map structured agent drafting payloads → generation requirements."""

from __future__ import annotations

from typing import Any

from app.models.drafting.generate import Party, StructuredGenerateRequest
from app.services.generation.language import language_label, normalize_language_code


def _format_parties(parties: list[Party] | None) -> str | None:
    if not parties:
        return None
    parts: list[str] = []
    for p in parties:
        bit = f"{p.name.strip()} ({p.role.strip()})"
        parts.append(bit)
    return "; ".join(parts) if parts else None


def build_requirements(request: StructuredGenerateRequest) -> dict[str, Any]:
    """Build the requirements dict consumed by PromptBuilder / GenerationAgent."""
    language = normalize_language_code(
        request.language.value if request.language else "en"
    )
    meta = request.metadata

    reqs: dict[str, Any] = {
        "document_type": request.doc_type,
        "language": language,
        "title": meta.title,
        "governing_law": meta.governing_law,
    }

    if meta.num_pages is not None:
        reqs["num_pages"] = meta.num_pages
    if meta.jurisdiction:
        reqs["jurisdiction"] = meta.jurisdiction
    if meta.purpose:
        reqs["purpose"] = meta.purpose
        reqs["summary"] = meta.purpose[:500]

    parties_str = _format_parties(meta.parties)
    if parties_str:
        reqs["parties"] = parties_str

    brief = (request.instructions or "").strip()
    if brief:
        reqs["instructions"] = brief
        if "summary" not in reqs:
            reqs["summary"] = brief[:500]

    return reqs


def build_synthetic_prompt(request: StructuredGenerateRequest, requirements: dict[str, Any]) -> str:
    """Human-readable brief seeded into conversation history for the LLM."""
    lang = normalize_language_code(requirements.get("language", "en"))
    lines = [
        f"Generate a complete {requirements.get('document_type', request.doc_type)} document.",
        (
            f"OUTPUT LANGUAGE (mandatory): {language_label(lang)}. "
            f'The TipTap metadata.language field MUST be "{lang}". '
            "Write all generated legal text in that language."
        ),
    ]
    if request.instructions and request.instructions.strip():
        lines.append(f"Instructions: {request.instructions.strip()}")

    skip = {"document_type", "language", "instructions", "summary"}
    for key, value in requirements.items():
        if key in skip or value is None or value == "":
            continue
        label = key.replace("_", " ").title()
        lines.append(f"{label}: {value}")

    lines.append("Proceed to generate the full document now. Do not ask clarifying questions.")
    return "\n".join(lines)
