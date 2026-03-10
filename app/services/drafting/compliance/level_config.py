"""Compliance check level limits: quick, standard, deep."""

from typing import TypedDict


class ComplianceLevelLimits(TypedDict):
    doc_char_limit: int
    blocks_limit: int
    block_char_limit: int
    legal_context_limit: int
    initial_limit: int
    rerank_top: int
    top_k_per_query: int
    max_clauses_citations: int
    per_clause_top_k: int
    per_clause_rrf_k: int
    per_clause_rerank_k: int


COMPLIANCE_LEVEL_CONFIG: dict[str, ComplianceLevelLimits] = {
    "quick": ComplianceLevelLimits(
        doc_char_limit=12_000,
        blocks_limit=50,
        block_char_limit=200,
        legal_context_limit=15_000,
        initial_limit=25,
        rerank_top=10,
        top_k_per_query=12,
        max_clauses_citations=20,
        per_clause_top_k=8,
        per_clause_rrf_k=10,
        per_clause_rerank_k=3,
    ),
    "standard": ComplianceLevelLimits(
        doc_char_limit=28_000,
        blocks_limit=100,
        block_char_limit=300,
        legal_context_limit=25_000,
        initial_limit=40,
        rerank_top=15,
        top_k_per_query=16,
        max_clauses_citations=35,
        per_clause_top_k=10,
        per_clause_rrf_k=12,
        per_clause_rerank_k=5,
    ),
    "deep": ComplianceLevelLimits(
        doc_char_limit=50_000,
        blocks_limit=200,
        block_char_limit=400,
        legal_context_limit=40_000,
        initial_limit=60,
        rerank_top=20,
        top_k_per_query=20,
        max_clauses_citations=50,
        per_clause_top_k=12,
        per_clause_rrf_k=15,
        per_clause_rerank_k=8,
    ),
}


def get_compliance_limits(level: str) -> ComplianceLevelLimits:
    """Return limits for the given check level. Unknown levels fall back to quick."""
    return COMPLIANCE_LEVEL_CONFIG.get(level.strip().lower(), COMPLIANCE_LEVEL_CONFIG["quick"])
