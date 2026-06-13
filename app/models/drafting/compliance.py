"""Pydantic models for compliance analysis request/response.

Request accepts doc_id (from doc collection); document is loaded from Qdrant with block_id, text, type per block.
"""

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ComplianceAnalysisRequest(BaseModel):
    """Input for POST /drafting/compliance/analyze. Document is loaded from the doc collection (Qdrant) by doc_id, with block_id and type per block for context."""

    doc_id: str = Field(description="Document ID in the doc collection (e.g. Qdrant doc_blocks).")
    language: str = Field(default="en", description="Response language (e.g. en, am).")
    check_level: Literal["quick", "standard", "deep"] = Field(
        default="quick",
        description="Analysis depth: quick (fastest), standard, or deep (most thorough).",
    )


# ---------------------------------------------------------------------------
# Citations and internal structures
# ---------------------------------------------------------------------------


class LegalCitation(BaseModel):
    """Single citation to Ethiopian law (Source | Article | Title + excerpt)."""

    document_id: str = Field(description="Source name / book (e.g. Civil Code).")
    item_id: str = Field(description="Article or provision id.")
    title: str = Field(default="", description="Section or article title.")
    excerpt: str = Field(default="", description="Relevant text excerpt.")


class LegalIssue(BaseModel):
    """A compliance or legal issue identified in the document."""

    issue_id: str = Field(description="Unique id for this issue.")
    description: str = Field(description="Description of the issue.")
    severity: str = Field(description="LOW | MEDIUM | HIGH | CRITICAL")
    block_id: str | None = Field(default=None, description="Block id if tied to a clause.")
    citations: list[LegalCitation] = Field(default_factory=list)


class EditorFixLegalBasis(BaseModel):
    """Legal basis for an editor fix action."""

    source: str = Field(description="Legal source name (e.g. ethiopian-labor-proclamation).")
    article: str = Field(default="", description="Article or section reference.")
    rationale: str = Field(default="", description="Why this law requires the fix.")


class EditorFixSpec(BaseModel):
    """Structured edit spec for the semantic editor. Separate from human-readable analysis."""

    action: Literal["replace"] = Field(description="Edit action type.")
    block_id: str | None = Field(default=None, description="Target document block id.")
    clause_reference: str = Field(default="", description="Human-readable clause label.")
    current_text: str = Field(description="Current clause or block text.")
    problem_summary: str = Field(description="Brief summary of the compliance problem.")
    offending_phrases: list[str] = Field(default_factory=list)
    legal_requirement: str = Field(description="What the law requires in this area.")
    rewrite_directive: str = Field(description="Imperative instruction for rewriting the clause.")
    remove_phrases: list[str] = Field(default_factory=list)
    add_elements: list[str] = Field(default_factory=list)
    suggested_text: str = Field(description="Concrete draft replacement text.")
    placeholder_policy: Literal["use_bracketed_placeholders_when_values_unknown"] = (
        "use_bracketed_placeholders_when_values_unknown"
    )
    legal_basis: list[EditorFixLegalBasis] = Field(default_factory=list)
    document_language: str = Field(default="en")
    severity: str = Field(description="e.g. medium_risk | high_risk | critical_risk")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the fix spec.")


class ClauseAnalysis(BaseModel):
    """Analysis of a single clause or section."""

    clause_id: str = Field(description="Unique id for this clause.")
    text: str = Field(description="Clause text or summary.")
    risk_level: str = Field(description="LOW | MEDIUM | HIGH | CRITICAL")
    implications: str = Field(default="", description="Legal implications.")
    block_id: str | None = Field(default=None, description="Mapped block id if from TipTap.")
    citations: list[LegalCitation] = Field(default_factory=list)
    ethiopian_law_implications: list[str] = Field(
        default_factory=list,
        description="Specific Ethiopian law implications for this clause (populated for non-LOW risk clauses).",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations to address risk in this clause (populated for non-LOW risk clauses).",
    )
    editor_fix: EditorFixSpec | None = Field(
        default=None,
        description="Structured edit spec for semantic editor. Populated for MEDIUM+ (warning and above) clauses; null for LOW.",
    )


class EthiopianLawCompliance(BaseModel):
    """Summary of compliance with Ethiopian law."""

    summary: str = Field(default="", description="Brief compliance summary.")
    applicable_laws: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class ComplianceAnalysisResponse(BaseModel):
    """Full response from compliance analysis."""

    document_type: str = Field(description="Detected or provided document type.")
    overall_risk_level: str = Field(description="LOW | MEDIUM | HIGH | CRITICAL")
    risk_score: float = Field(description="Numeric risk score 0–100. Higher = riskier.")
    compliance_score: float = Field(
        default=0.0,
        description="Compliance health score 0–100 (100 − risk_score). Higher = healthier.",
    )
    summary: str = Field(description="Executive summary of the analysis.")
    clauses: list[ClauseAnalysis] = Field(default_factory=list)
    issues_by_block_id: dict[str, list[LegalIssue]] = Field(
        default_factory=dict,
        description="Issues keyed by block_id.",
    )
    ethiopian_law_compliance: EthiopianLawCompliance = Field(
        default_factory=EthiopianLawCompliance,
    )
    recommendations: list[str] = Field(default_factory=list)
    should_sign: bool | None = Field(default=None, description="Recommendation to sign or not.")
    critical_issues: list[LegalIssue] = Field(default_factory=list)
    missing_clauses: list[str] = Field(
        default_factory=list,
        description="Clauses that may be missing for this document type.",
    )
    citations: list[LegalCitation] = Field(
        default_factory=list,
        description="Global citations used in the analysis.",
    )
    score_breakdown: dict = Field(
        default_factory=dict,
        description=(
            "Audit trail showing exactly how risk_score was computed: "
            "clause_counts, issue_counts, missing_clauses_count, penalty totals, "
            "raw_penalty, max_penalty, risk_score, compliance_score, overall_risk_level."
        ),
    )
