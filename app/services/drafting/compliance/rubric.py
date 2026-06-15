"""Versioned compliance rubric: fixed checklist per document type with deterministic weights."""

from __future__ import annotations

from typing import Literal

RubricStatus = Literal[
    "PRESENT",
    "PARTIAL",
    "MISSING",
    "NON_COMPLIANT",
    "NOT_APPLICABLE",
]

RUBRIC_VERSION = "1"

# Penalty points per rubric item outcome (applied once per checklist item).
STATUS_PENALTIES: dict[str, int] = {
    "PRESENT": 0,
    "NOT_APPLICABLE": 0,
    "PARTIAL": 4,
    "MISSING": 8,
    "NON_COMPLIANT": 12,
}

# Document types used in applies_to (case-insensitive match against detected type).
_ALL_TYPES = (
    "Contract",
    "Employment Agreement",
    "NDA",
    "MOU",
    "Lease",
    "Partnership",
    "Service Agreement",
)

RUBRIC_ITEMS: list[dict] = [
    {
        "id": "jurisdiction_governing_law",
        "title": "Governing law and jurisdiction",
        "category": "mandatory_law",
        "applies_to": list(_ALL_TYPES),
        "description": "Explicit Ethiopian governing law; no foreign governing law or jurisdiction.",
    },
    {
        "id": "contract_type_parties",
        "title": "Contract type and parties",
        "category": "structure",
        "applies_to": list(_ALL_TYPES),
        "description": "Document type is clear; parties are identified with roles.",
    },
    {
        "id": "parties_obligations",
        "title": "Parties' obligations",
        "category": "substance",
        "applies_to": list(_ALL_TYPES),
        "description": "Core obligations of each party are stated with sufficient clarity.",
    },
    {
        "id": "liability_limitation",
        "title": "Liability and limitation of liability",
        "category": "risk",
        "applies_to": ["Contract", "Service Agreement", "Partnership", "MOU", "Lease"],
        "description": "Liability scope and any caps/exclusions are addressed where appropriate.",
    },
    {
        "id": "termination_notice",
        "title": "Termination and notice",
        "category": "lifecycle",
        "applies_to": list(_ALL_TYPES),
        "description": "Termination grounds, notice periods, and post-termination effects.",
    },
    {
        "id": "confidentiality",
        "title": "Confidentiality",
        "category": "risk",
        "applies_to": ["NDA", "Employment Agreement", "Service Agreement", "Partnership", "MOU"],
        "description": "Confidential information definition, obligations, and exceptions.",
    },
    {
        "id": "ip_assignment",
        "title": "Intellectual property and assignment",
        "category": "substance",
        "applies_to": ["Employment Agreement", "Service Agreement", "Partnership", "Contract"],
        "description": "IP ownership, assignment, and work-product rules where relevant.",
    },
    {
        "id": "dispute_resolution",
        "title": "Dispute resolution and venue",
        "category": "mandatory_law",
        "applies_to": list(_ALL_TYPES),
        "description": "Dispute mechanism and Ethiopian courts/arbitration venue.",
    },
    {
        "id": "mandatory_ethiopian_law",
        "title": "Mandatory Ethiopian law compliance",
        "category": "mandatory_law",
        "applies_to": list(_ALL_TYPES),
        "description": "No prohibited terms (foreign law, at-will employment, outdated code refs, etc.).",
    },
    {
        "id": "standard_clauses",
        "title": "Standard clauses for document type",
        "category": "completeness",
        "applies_to": list(_ALL_TYPES),
        "description": "Typical clauses for this document type are present or reasonably covered.",
    },
    {
        "id": "unfair_terms",
        "title": "Unfair or one-sided terms",
        "category": "risk",
        "applies_to": list(_ALL_TYPES),
        "description": "Materially unfair, unconscionable, or heavily one-sided provisions.",
    },
]

# Derived from COMPLIANCE_CHECKLIST narrative in analysis_agent (Ethiopian framework summary).
RUBRIC_ETHIOPIAN_FRAMEWORK = """
Ethiopian legal framework:
- Ethiopian Commercial Code (Proclamation No. 1243/2021)
- Ethiopian Civil Code (1960)
- Ethiopian Labor Law (Proclamation No. 1156/2019)
- Ethiopian Investment Law (Proclamation No. 1180/2020)
- Ethiopian Constitution (1995)

Critical prohibitions:
- NO foreign governing law - MUST use Ethiopian law
- NO foreign jurisdiction clauses - MUST use Ethiopian courts
- NO outdated Commercial Code references (166/1960) - use Proclamation No. 1243/2021
- NO at-will employment (illegal under Proclamation No. 1156/2019)
- Mandatory labor benefits and notice periods per Ethiopian Labor Law
""".strip()

VALID_STATUSES = frozenset(STATUS_PENALTIES.keys())


def normalize_document_type(document_type: str | None) -> str:
    """Map detected/provided type to a rubric applies_to label."""
    if not document_type:
        return "Contract"
    dt = document_type.strip()
    lower = dt.lower()
    if "employment" in lower or "labour" in lower or "labor" in lower:
        return "Employment Agreement"
    if "nda" in lower or "non-disclosure" in lower or "confidential" in lower and "agreement" in lower:
        return "NDA"
    if "mou" in lower or "memorandum" in lower:
        return "MOU"
    if "lease" in lower or "rental" in lower or "tenant" in lower:
        return "Lease"
    if "partnership" in lower or "joint venture" in lower or "jv" in lower:
        return "Partnership"
    if "service" in lower and "agreement" in lower:
        return "Service Agreement"
    return dt if dt in _ALL_TYPES else "Contract"


def get_rubric_items_for_document_type(document_type: str | None) -> list[dict]:
    """Return rubric checklist items applicable to the given document type."""
    norm = normalize_document_type(document_type)
    return [item for item in RUBRIC_ITEMS if norm in item["applies_to"]]


def format_rubric_for_prompt(items: list[dict]) -> str:
    """Format rubric ids for inclusion in the analysis prompt."""
    lines = []
    for item in items:
        lines.append(
            f'- id="{item["id"]}" | {item["title"]} ({item["category"]}): {item["description"]}'
        )
    return "\n".join(lines)
