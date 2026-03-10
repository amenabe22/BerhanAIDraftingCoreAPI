"""Deterministic compliance risk scoring engine.

Converts LLM-classified clause/issue severities into a transparent, auditable
risk_score (0–100) and compliance_score (0–100). No LLM involvement — pure math.

Methodology
-----------
Based on the weighted-penalty model used by Pactly, IntelAgree, and IntelliAgree,
adapted for clause-level severity classification outputs:

  raw_penalty = Σ clause_penalties
              + Σ issue_penalties
              + missing_clauses_count × MISSING_CLAUSE_PENALTY
              + should_sign_penalty (flat, when should_sign is False)
              + concern_count × CONCERN_PENALTY

  risk_score       = min(raw_penalty / max_penalty × 100, 100.0)
  compliance_score = 100.0 − risk_score
  overall_risk_level derived from risk_score thresholds

Thresholds
----------
  risk_score >= 70  → CRITICAL
  risk_score >= 45  → HIGH
  risk_score >= 20  → MEDIUM
  risk_score <  20  → LOW

All weights and thresholds are module-level constants so they are easy to tune
without touching business logic.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Penalty weights
# ---------------------------------------------------------------------------

# Per-clause risk level → penalty points
CLAUSE_PENALTIES: dict[str, int] = {
    "CRITICAL": 20,
    "HIGH": 10,
    "MEDIUM": 5,
    "LOW": 0,
}

# Per-document-level issue severity → penalty points
ISSUE_PENALTIES: dict[str, int] = {
    "CRITICAL": 15,
    "HIGH": 8,
    "MEDIUM": 3,
    "LOW": 1,
}

# Flat penalty per missing clause identified by the LLM
MISSING_CLAUSE_PENALTY: int = 3

# Flat bonus added when should_sign is explicitly False
SHOULD_NOT_SIGN_PENALTY: int = 10

# Flat penalty per Ethiopian law concern listed
CONCERN_PENALTY: int = 2

# Default normalization ceiling.
# Can be overridden via settings.COMPLIANCE_SCORE_MAX_PENALTY.
DEFAULT_MAX_PENALTY: int = 150

# ---------------------------------------------------------------------------
# Risk-level thresholds (risk_score → label)
# ---------------------------------------------------------------------------

RISK_LEVEL_THRESHOLDS: list[tuple[int, str]] = [
    (70, "CRITICAL"),
    (45, "HIGH"),
    (20, "MEDIUM"),
    (0, "LOW"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_risk_score(
    clauses: list[dict],
    issues: list[dict],
    missing_clauses: list[str],
    should_sign: bool | None,
    concern_count: int,
    *,
    max_penalty: int = DEFAULT_MAX_PENALTY,
) -> dict:
    """Compute a deterministic risk score from LLM-classified document signals.

    Parameters
    ----------
    clauses:
        List of clause dicts, each expected to have a ``risk_level`` key
        (``LOW | MEDIUM | HIGH | CRITICAL``).
    issues:
        List of issue dicts, each expected to have a ``severity`` key.
    missing_clauses:
        List of missing-clause strings identified by the LLM. Only the count
        is used for scoring.
    should_sign:
        LLM recommendation. ``False`` adds a flat penalty; ``True`` or ``None``
        adds nothing.
    concern_count:
        Number of Ethiopian law concerns listed in ``ethiopian_law_compliance``.
    max_penalty:
        Normalization ceiling. Scores are expressed as a fraction of this value.
        Override via ``settings.COMPLIANCE_SCORE_MAX_PENALTY``.

    Returns
    -------
    dict with keys:
        clause_counts, issue_counts, missing_clauses_count,
        should_sign_penalty, concern_count,
        raw_penalty, max_penalty,
        risk_score, compliance_score, overall_risk_level
    """
    # --- clause penalties ---
    clause_counts: dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    clause_raw: int = 0
    for clause in clauses or []:
        level = (clause.get("risk_level") or "LOW").upper().strip()
        if level not in clause_counts:
            level = "LOW"
        clause_counts[level] += 1
        clause_raw += CLAUSE_PENALTIES.get(level, 0)

    # --- issue penalties ---
    issue_counts: dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    issue_raw: int = 0
    for issue in issues or []:
        severity = (issue.get("severity") or "LOW").upper().strip()
        if severity not in issue_counts:
            severity = "LOW"
        issue_counts[severity] += 1
        issue_raw += ISSUE_PENALTIES.get(severity, 0)

    # --- missing clauses ---
    missing_count: int = len(missing_clauses or [])
    missing_raw: int = missing_count * MISSING_CLAUSE_PENALTY

    # --- should_sign bonus ---
    sign_penalty: int = SHOULD_NOT_SIGN_PENALTY if should_sign is False else 0

    # --- Ethiopian law concerns ---
    concern_raw: int = (concern_count or 0) * CONCERN_PENALTY

    # --- aggregate ---
    raw_penalty: int = clause_raw + issue_raw + missing_raw + sign_penalty + concern_raw
    effective_max = max(max_penalty, 1)  # guard against zero division
    risk_score_float = min(raw_penalty / effective_max * 100.0, 100.0)
    risk_score = round(risk_score_float, 2)
    compliance_score = round(100.0 - risk_score, 2)

    # --- derive overall_risk_level ---
    overall_risk_level = _risk_level_from_score(risk_score)

    return {
        "clause_counts": clause_counts,
        "clause_penalty_total": clause_raw,
        "issue_counts": issue_counts,
        "issue_penalty_total": issue_raw,
        "missing_clauses_count": missing_count,
        "missing_clause_penalty_total": missing_raw,
        "should_sign_penalty": sign_penalty,
        "concern_count": concern_count or 0,
        "concern_penalty_total": concern_raw,
        "raw_penalty": raw_penalty,
        "max_penalty": effective_max,
        "risk_score": risk_score,
        "compliance_score": compliance_score,
        "overall_risk_level": overall_risk_level,
    }


def _risk_level_from_score(score: float) -> str:
    """Map a numeric risk_score (0–100) to a severity label."""
    for threshold, label in RISK_LEVEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "LOW"
