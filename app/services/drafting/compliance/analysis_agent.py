"""Compliance analysis agent: extract -> targeted retrieval -> LLM -> per-clause citations -> response."""

import json
import re
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import settings
from app.logging_config import get_logger
from app.models.drafting.compliance import (
    ClauseAnalysis,
    ComplianceAnalysisResponse,
    EditorFixLegalBasis,
    EditorFixSpec,
    EthiopianLawCompliance,
    IssueSeverity,
    LegalCitation,
    LegalIssue,
)
from app.services.drafting.compliance.level_config import get_compliance_limits
from app.services.generation.language import language_label, normalize_language_code
from app.services.drafting.compliance.compliance_cache import (
    get_last_rubric_result,
    store_last_rubric_result,
)
from app.services.drafting.compliance.content_hash import (
    compute_document_content_hash,
    compute_per_block_hashes,
)
from app.services.drafting.compliance.rubric import (
    RUBRIC_ETHIOPIAN_FRAMEWORK,
    RUBRIC_VERSION,
    VALID_STATUSES,
    format_rubric_for_prompt,
    get_rubric_items_for_document_type,
)
from app.services.drafting.compliance.scoring import (
    DEFAULT_MAX_PENALTY,
    compute_risk_score,
    compute_rubric_score,
)
from app.services.drafting.knowledge_retrieval import (
    build_clause_legal_query,
    generate_targeted_queries,
    rerank_with_llm,
    search_legal_knowledge,
)

log = get_logger("compliance_agent")


def _emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    *,
    phase: str,
    percent: int,
    message: str,
) -> None:
    if progress_callback is None:
        return
    p = max(0, min(100, percent))
    progress_callback({"phase": phase, "percent": p, "message": message})


# Checklist for the analysis prompt (aligned with BerhanAdvisorCoreAPI compliance_rules)
COMPLIANCE_CHECKLIST = """
Consider at least: jurisdiction and governing law, contract type, parties' obligations, liability and limitation of liability, termination and notice, confidentiality, IP and assignment, dispute resolution and venue, mandatory law (Ethiopian), missing standard clauses for this document type, unfair terms.

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

Flag as non-compliant: references to Proclamation No. 166/1960 or Ethiopian Commercial Code (1960) in commercial/partnership governing-law clauses; current law is Proclamation No. 1243/2021.

For partnership and commercial agreements, governing law must reference Proclamation No. 1243/2021 (not 166/1960).
Governing law must state laws of Ethiopia; jurisdiction must be Ethiopian courts.
""".strip()


def _extract_text_from_tiptap_node(node: dict) -> str:
    """Recursively extract text from a TipTap content node."""
    if not isinstance(node, dict):
        return ""
    if node.get("type") == "text":
        return node.get("text", "")
    parts = []
    for child in node.get("content", []) or []:
        parts.append(_extract_text_from_tiptap_node(child))
    return "".join(parts)


def extract_blocks_from_tiptap(tiptap_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract blocks (paragraphs/sections) with block_id and text from TipTap JSON.

    Handles multi-page document structure (page nodes wrapping paragraphs/headings)
    and reads all common block-id attribute variants used across the codebase:
    ``block_id`` (BerhanAdvisorCoreAPI), ``id``, and ``blockId``.
    """
    blocks: list[dict[str, Any]] = []

    def _traverse(nodes: list) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_type = node.get("type", "")
            # Recurse into container nodes without extracting them as blocks
            if node_type in ("page", "doc"):
                _traverse(node.get("content") or [])
                continue
            text = _extract_text_from_tiptap_node(node).strip()
            attrs = node.get("attrs") or {}
            # Prefer the explicit block_id attr used by BerhanAdvisorCoreAPI generation
            block_id = (
                attrs.get("block_id")
                or attrs.get("id")
                or attrs.get("blockId")
                or (f"b{len(blocks)}" if text else None)
            )
            if text or node_type in ("heading", "paragraph", "blockquote"):
                blocks.append({
                    "block_id": block_id or f"b{len(blocks)}",
                    "type": node_type,
                    "text": text,
                })

    content = tiptap_json.get("content") or []
    if not content:
        return blocks
    _traverse(content)
    return blocks


def _extract_full_document_text(
    tiptap_json: dict[str, Any] | None, document_text: str | None
) -> str:
    """Return full document text from TipTap or plain document_text."""
    if document_text:
        return document_text.strip()
    if tiptap_json:
        blocks = extract_blocks_from_tiptap(tiptap_json)
        return "\n\n".join(b.get("text", "") for b in blocks if b.get("text"))
    return ""


def _detect_document_type(text: str) -> str:
    """Simple heuristic or LLM-based detection. Returns a short label."""
    text_lower = (text or "")[:2000].lower()
    if "confidential" in text_lower and ("disclose" in text_lower or "nda" in text_lower):
        return "NDA"
    if "memorandum of understanding" in text_lower or "mou" in text_lower:
        return "MOU"
    if "employment" in text_lower and ("contract" in text_lower or "agreement" in text_lower):
        return "Employment Agreement"
    if "service agreement" in text_lower or "terms of service" in text_lower:
        return "Service Agreement"
    if "lease" in text_lower or "tenant" in text_lower:
        return "Lease"
    return "Contract"


def _metadata_get(m: dict, *keys: str, default: str = "") -> str:
    """Get first non-empty value from metadata, including nested metadata and common Legal KB key variants."""
    nested = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
    for key in keys:
        for src in (m, nested):
            v = src.get(key)
            if v is not None and str(v).strip():
                return str(v).strip()
    return default


def _docs_to_citations(docs: list[Document]) -> list[LegalCitation]:
    """Convert LangChain Documents (Legal KB) to LegalCitation list. Uses fallback keys (source_file, article_id, etc.) and skips citations with no document_id, item_id, or excerpt."""
    out: list[LegalCitation] = []
    for d in docs:
        m = getattr(d, "metadata", None) or {}
        document_id = _metadata_get(m, "document_id", "source_file", "source", default="")
        item_id = _metadata_get(m, "item_id", "article_id", "article_number", default="")
        title = (_metadata_get(m, "title", "section_title", default=""))[:500]
        excerpt = ((d.page_content or "").strip())[:500]
        if not document_id and not item_id and not excerpt:
            continue
        out.append(
            LegalCitation(
                document_id=document_id,
                item_id=item_id,
                title=title,
                excerpt=excerpt,
            )
        )
    return out


def _format_legal_context(docs: list[Document]) -> str:
    """Format retrieved chunks for the analysis prompt."""
    parts = []
    for d in docs:
        m = getattr(d, "metadata", None) or {}
        parts.append(
            f"[Source: {m.get('document_id', '')} | Article {m.get('item_id', '')} | {m.get('title', '')}]\n{d.page_content or ''}"
        )
    return "\n\n---\n\n".join(parts)


def _format_blocks_for_prompt(
    blocks: list[dict],
    *,
    blocks_limit: int = 50,
    block_char_limit: int = 200,
) -> str:
    """Format document blocks with block_id for the analysis prompt."""
    if not blocks:
        return "(no structured blocks available)"
    lines: list[str] = []
    for b in blocks[:blocks_limit]:
        block_id = b.get("block_id")
        if not block_id:
            continue
        block_type = b.get("type") or "paragraph"
        text = (b.get("text") or "").strip()
        if not text:
            continue
        excerpt = text[:block_char_limit]
        if len(text) > block_char_limit:
            excerpt += "…"
        anchor_note = (
            " | NON-ANCHORABLE: do not assign issues/clauses to this block"
            if _is_trivial_block_text(text)
            else ""
        )
        lines.append(
            f"[block_id: {block_id} | type: {block_type}{anchor_note}] {excerpt}"
        )
    return "\n".join(lines) if lines else "(no structured blocks available)"


def _valid_block_ids(blocks: list[dict]) -> set[str]:
    return {str(b["block_id"]) for b in blocks if b.get("block_id")}


def _normalize_match_text(text: str) -> str:
    """Lowercase and collapse whitespace for fuzzy block text matching."""
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _significant_tokens(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "but", "to", "for", "of", "in", "on", "at",
        "is", "are", "was", "were", "be", "been", "this", "that", "with", "as", "by",
        "from", "not", "no", "it", "its", "shall", "may", "will", "can", "while",
    }
    tokens = re.findall(r"[a-z0-9]+", _normalize_match_text(text))
    return {t for t in tokens if len(t) > 2 and t not in stop}


def _text_overlap_score(a: str, b: str) -> float:
    ta, tb = _significant_tokens(a), _significant_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _is_trivial_block_text(text: str) -> bool:
    """Blocks too short or label-like to anchor compliance highlights."""
    norm = _normalize_match_text(text)
    if not norm:
        return True
    if len(norm) < 12:
        return True
    words = norm.split()
    if len(words) <= 2:
        return True
    if len(words) <= 3 and norm.rstrip().endswith(":"):
        return True
    return False


def _looks_like_missing_provision(text: str) -> bool:
    t = _normalize_match_text(text)
    markers = (
        "document lacks",
        "document does not",
        "is missing",
        "absence of",
        "missing clause",
        "missing provision",
        "not explicitly include",
        "does not explicitly",
        "without a dedicated",
        "no specific clause",
        "does not include",
        "is not present",
        "is absent",
    )
    return any(m in t for m in markers)


_PROVISION_MARKERS: dict[str, tuple[str, ...]] = {
    "governing_law": (
        "governed by",
        "governing law",
        "construed in accordance with",
        "laws of ethiopia",
        "law of ethiopia",
    ),
    "jurisdiction": (
        "jurisdiction",
        "exclusive jurisdiction",
        "ethiopian courts",
        "courts of ethiopia",
        "submit to the",
    ),
    "dispute_resolution": (
        "dispute resolution",
        "arbitration",
        "mediation",
    ),
    "limitation_of_liability": (
        "limitation of liability",
        "limit liability",
        "liability shall not exceed",
        "cap on liability",
    ),
    "termination": (
        "termination of employment",
        "terminate this agreement",
        "notice of termination",
        "termination by",
    ),
}

_MISSING_CLAIM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "governing_law": ("governing law", "governing-law"),
    "jurisdiction": ("jurisdiction", "venue", "forum selection"),
    "dispute_resolution": ("dispute resolution", "arbitration clause", "mediation clause"),
    "limitation_of_liability": (
        "limitation of liability",
        "liability cap",
        "cap on liability",
    ),
    "termination": ("termination clause", "termination notice", "notice period"),
}


def _document_corpus(full_text: str, blocks: list[dict]) -> str:
    if full_text and full_text.strip():
        return _normalize_match_text(full_text)
    return _normalize_match_text("\n".join(str(b.get("text") or "") for b in blocks))


def _corpus_has_provision(corpus: str, provision_key: str) -> bool:
    patterns = _PROVISION_MARKERS.get(provision_key, ())
    return any(p in corpus for p in patterns)


def _text_claims_missing_provision(text: str, provision_key: str) -> bool:
    if not _looks_like_missing_provision(text):
        return False
    t = _normalize_match_text(text)
    return any(k in t for k in _MISSING_CLAIM_KEYWORDS.get(provision_key, ()))


def _is_false_missing_provision_finding(text: str, corpus: str) -> bool:
    """True when analysis claims a provision is missing but document text contains it."""
    for provision_key in _PROVISION_MARKERS:
        if _text_claims_missing_provision(text, provision_key) and _corpus_has_provision(
            corpus, provision_key
        ):
            return True
    return False


def _block_index(blocks: list[dict], block_id: str | None) -> int | None:
    if not block_id:
        return None
    for i, block in enumerate(blocks):
        if str(block.get("block_id")) == str(block_id):
            return i
    return None


def _reanchor_heading_to_body(block_id: str | None, blocks: list[dict]) -> str | None:
    """Prefer the first substantive paragraph under a section heading for highlights."""
    if not block_id:
        return None
    idx = _block_index(blocks, block_id)
    if idx is None:
        return block_id
    block = blocks[idx]
    if block.get("type") != "heading":
        return block_id
    for j in range(idx + 1, len(blocks)):
        follower = blocks[j]
        if follower.get("type") == "heading":
            break
        follower_text = str(follower.get("text") or "").strip()
        if follower.get("type") == "paragraph" and len(follower_text) >= 30:
            return str(follower.get("block_id"))
    return block_id


def _downgrade_false_positive_clause(clause: dict) -> None:
    clause["risk_level"] = "LOW"
    clause["editor_fix"] = None
    clause["block_id"] = None
    clause["recommendations"] = []
    clause["ethiopian_law_implications"] = []


def _filter_false_missing_provision_findings(
    clauses: list[dict],
    issues: list[dict],
    blocks: list[dict],
    full_text: str,
) -> None:
    """Remove/downgrade LLM 'missing X' findings when the document already contains X."""
    corpus = _document_corpus(full_text, blocks)

    for clause in clauses:
        combined = " ".join(
            part
            for part in (
                clause.get("text"),
                clause.get("implications"),
                (clause.get("editor_fix") or {}).get("problem_summary")
                if isinstance(clause.get("editor_fix"), dict)
                else None,
            )
            if part
        )
        if _is_false_missing_provision_finding(combined, corpus):
            _downgrade_false_positive_clause(clause)

    kept_issues: list[dict] = []
    for issue in issues:
        description = str(issue.get("description") or "")
        if _is_false_missing_provision_finding(description, corpus):
            continue
        kept_issues.append(issue)
    issues[:] = kept_issues


def _filter_missing_clauses_list(missing_clauses: list, corpus: str) -> list:
    kept: list = []
    for entry in missing_clauses or []:
        text = str(entry)
        if _is_false_missing_provision_finding(text, corpus):
            continue
        kept.append(entry)
    return kept


def _validate_block_binding(text: str, block_id: str | None, blocks: list[dict]) -> str | None:
    """Return block_id only when the block text actually supports the clause/issue."""
    if not block_id:
        return None
    block = next((b for b in blocks if str(b.get("block_id")) == str(block_id)), None)
    if not block:
        return None
    block_text = str(block.get("text") or "")
    if _is_trivial_block_text(block_text):
        return None
    score = _text_overlap_score(text, block_text)
    if _looks_like_missing_provision(text) and score < 0.25:
        return None
    if score < 0.15:
        return None
    norm_block = _normalize_match_text(block_text)
    norm_text = _normalize_match_text(text)
    if len(norm_block) >= 20 and norm_block not in norm_text and norm_text not in norm_block:
        if score < 0.35:
            return None
    return str(block_id)


def _match_block_by_text(text: str, blocks: list[dict]) -> str | None:
    """Return block_id whose text best matches the given excerpt."""
    excerpt = (text or "").strip()
    if len(_normalize_match_text(excerpt)) < 8:
        return None

    best_id: str | None = None
    best_score = 0.0
    norm_excerpt = _normalize_match_text(excerpt)

    for b in blocks:
        block_text = str(b.get("text") or "").strip()
        if _is_trivial_block_text(block_text):
            continue

        score = _text_overlap_score(excerpt, block_text)
        norm_block = _normalize_match_text(block_text)
        if len(norm_block) >= 20 and (norm_block in norm_excerpt or norm_excerpt in norm_block):
            score = max(score, 0.85)

        bid = b.get("block_id")
        if bid and score > best_score:
            best_score = score
            best_id = str(bid)

    if best_score >= 0.35 and best_id:
        return best_id
    return None


def _sanitize_clause_and_issue_block_ids(
    clauses: list[dict], issues: list[dict], blocks: list[dict]
) -> None:
    """Drop block_id bindings that point at unrelated or non-anchorable blocks."""
    for clause in clauses:
        text = " ".join(
            part
            for part in (
                clause.get("text"),
                clause.get("implications"),
                (clause.get("editor_fix") or {}).get("problem_summary")
                if isinstance(clause.get("editor_fix"), dict)
                else None,
            )
            if part
        )
        validated = _validate_block_binding(text, clause.get("block_id"), blocks)
        validated = _reanchor_heading_to_body(validated, blocks)
        clause["block_id"] = validated
        editor_fix = clause.get("editor_fix")
        if isinstance(editor_fix, dict) and editor_fix.get("block_id"):
            fix_block = _validate_block_binding(
                text, editor_fix.get("block_id"), blocks
            )
            editor_fix["block_id"] = _reanchor_heading_to_body(fix_block, blocks)

    for issue in issues:
        text = str(issue.get("description") or "")
        validated = _validate_block_binding(text, issue.get("block_id"), blocks)
        issue["block_id"] = _reanchor_heading_to_body(validated, blocks)


def _map_clauses_to_blocks(clauses: list[dict], blocks: list[dict]) -> None:
    """Assign block_id to clauses by matching text/order where possible."""
    if not blocks:
        return
    valid_ids = _valid_block_ids(blocks)
    for c in clauses:
        bid = c.get("block_id")
        if bid and str(bid) not in valid_ids:
            c["block_id"] = None
        if c.get("block_id"):
            continue
        matched = _match_block_by_text(c.get("text") or "", blocks)
        if matched:
            c["block_id"] = matched


def _map_issues_to_blocks(issues: list[dict], blocks: list[dict]) -> None:
    """Assign block_id to issues by matching description text to blocks."""
    if not blocks:
        return
    valid_ids = _valid_block_ids(blocks)
    for issue in issues:
        bid = issue.get("block_id")
        if bid and str(bid) not in valid_ids:
            issue["block_id"] = None
        if issue.get("block_id"):
            continue
        matched = _match_block_by_text(issue.get("description") or "", blocks)
        if matched:
            issue["block_id"] = matched


def _normalize_editor_fix(
    raw: Any,
    clause: dict,
    blocks: list[dict],
    language: str,
) -> EditorFixSpec | None:
    """Validate and enrich LLM editor_fix for MEDIUM+ (warning and above) clauses."""
    risk = (clause.get("risk_level") or "").upper()
    if risk not in ("MEDIUM", "HIGH", "CRITICAL"):
        return None
    if not raw or not isinstance(raw, dict):
        return None

    rewrite_directive = (raw.get("rewrite_directive") or "").strip()
    suggested_text = (raw.get("suggested_text") or "").strip()
    problem_summary = (raw.get("problem_summary") or "").strip()
    if not rewrite_directive or not suggested_text or not problem_summary:
        log.warning(
            "editor_fix_incomplete",
            extra={"event": "editor_fix_incomplete", "clause_id": clause.get("clause_id")},
        )
        return None

    valid_ids = _valid_block_ids(blocks)
    block_id: str | None = None
    clause_bid = clause.get("block_id")
    if clause_bid and str(clause_bid) in valid_ids:
        block_id = str(clause_bid)
    else:
        raw_bid = raw.get("block_id")
        if raw_bid and str(raw_bid) in valid_ids:
            block_id = str(raw_bid)

    current_text = (raw.get("current_text") or clause.get("text") or "").strip()
    if block_id:
        for b in blocks:
            if str(b.get("block_id")) == block_id:
                block_text = (b.get("text") or "").strip()
                if block_text:
                    current_text = block_text
                break

    legal_basis: list[EditorFixLegalBasis] = []
    for lb in raw.get("legal_basis") or []:
        if isinstance(lb, dict) and (lb.get("source") or "").strip():
            legal_basis.append(
                EditorFixLegalBasis(
                    source=str(lb.get("source", "")).strip(),
                    article=str(lb.get("article", "") or "").strip(),
                    rationale=str(lb.get("rationale", "") or "").strip(),
                )
            )
    if not legal_basis:
        for cit in clause.get("citations") or []:
            if not isinstance(cit, dict):
                continue
            doc_id = str(cit.get("document_id", "") or "").strip()
            item_id = str(cit.get("item_id", "") or "").strip()
            if doc_id or item_id:
                legal_basis.append(
                    EditorFixLegalBasis(
                        source=doc_id or "ethiopian-law",
                        article=item_id,
                        rationale=str(cit.get("excerpt") or cit.get("title") or "")[:500],
                    )
                )

    severity_map = {
        "CRITICAL": "critical_risk",
        "HIGH": "high_risk",
        "MEDIUM": "medium_risk",
    }
    severity = severity_map.get(risk, "medium_risk")
    try:
        confidence = float(raw.get("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.8

    placeholder = raw.get("placeholder_policy") or "use_bracketed_placeholders_when_values_unknown"
    if placeholder != "use_bracketed_placeholders_when_values_unknown":
        placeholder = "use_bracketed_placeholders_when_values_unknown"

    try:
        return EditorFixSpec(
            action="replace",
            block_id=block_id,
            clause_reference=str(raw.get("clause_reference") or "").strip(),
            current_text=current_text,
            problem_summary=problem_summary,
            offending_phrases=[str(x) for x in (raw.get("offending_phrases") or []) if x],
            legal_requirement=str(raw.get("legal_requirement") or "").strip(),
            rewrite_directive=rewrite_directive,
            remove_phrases=[str(x) for x in (raw.get("remove_phrases") or []) if x],
            add_elements=[str(x) for x in (raw.get("add_elements") or []) if x],
            suggested_text=suggested_text,
            placeholder_policy=placeholder,
            legal_basis=legal_basis,
            document_language=str(raw.get("document_language") or language or "en"),
            severity=str(raw.get("severity") or severity),
            confidence=confidence,
        )
    except Exception as e:
        log.warning(
            "editor_fix_validation_failed",
            extra={
                "event": "editor_fix_validation_failed",
                "clause_id": clause.get("clause_id"),
                "error": str(e),
            },
        )
        return None


def _risk_level_to_issue_severity(risk_level: str) -> IssueSeverity:
    """Map LLM risk levels (LOW/MEDIUM/HIGH/CRITICAL) to consumer issue severity."""
    level = (risk_level or "LOW").upper()
    if level in ("CRITICAL", "HIGH"):
        return IssueSeverity.ERROR
    if level == "MEDIUM":
        return IssueSeverity.WARNING
    if level == "LOW":
        return IssueSeverity.INFO
    return IssueSeverity.NORMAL


def _determine_issue_type(risk_level: str, description: str) -> str:
    """Infer issue_type from risk level and description text."""
    text = (description or "").lower()
    if "non-compliant" in text or "non compliant" in text:
        return "non_compliant_clause"
    if "missing" in text:
        return "missing_provision"
    if "unclear" in text or "ambiguous" in text or "vague" in text:
        return "ambiguous_clause"
    if "unenforceable" in text:
        return "unenforceable_clause"
    return f"{(risk_level or 'low').lower()}_risk"


def _citations_to_str_list(cits: list) -> list[str]:
    """Convert citation dicts to flat strings for legacy LegalIssue.citations."""
    out: list[str] = []
    for cit in cits or []:
        if isinstance(cit, str) and cit.strip():
            out.append(cit.strip())
            continue
        if not isinstance(cit, dict):
            continue
        doc_id = str(cit.get("document_id", "") or "").strip()
        item_id = str(cit.get("item_id", "") or "").strip()
        title = str(cit.get("title", "") or "").strip()
        excerpt = str(cit.get("excerpt", "") or "").strip()
        head = " | ".join(p for p in (doc_id, item_id, title) if p)
        if head and excerpt:
            out.append(f"{head}: {excerpt}")
        elif head:
            out.append(head)
        elif excerpt:
            out.append(excerpt)
    return out


def _build_analysis_llm() -> ChatOpenAI:
    """Main rubric-evaluation LLM (optional reasoning model, deterministic seed)."""
    model = settings.COMPLIANCE_ANALYSIS_MODEL or settings.GEMINI_MODEL
    extra_body: dict = {}
    effort = (getattr(settings, "COMPLIANCE_REASONING_EFFORT", "") or "").strip().lower()
    if effort and effort not in ("none", "off", ""):
        extra_body["reasoning"] = {"effort": effort}
    seed = getattr(settings, "COMPLIANCE_ANALYSIS_SEED", 7)
    kwargs: dict = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": settings.OPENROUTER_API_KEY,
        "model": model,
        "temperature": getattr(settings, "COMPLIANCE_ANALYSIS_TEMPERATURE", 0.0),
        "max_tokens": getattr(settings, "COMPLIANCE_ANALYSIS_MAX_TOKENS", 32768),
        "streaming": True,
        "seed": seed,
        "model_kwargs": {
            "response_format": {"type": "json_object"},
        },
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    return ChatOpenAI(**kwargs)


def _normalize_rubric_check(raw: dict, *, default_id: str = "") -> dict:
    status = (raw.get("status") or "MISSING").upper().strip()
    if status not in VALID_STATUSES:
        status = "MISSING"
    block_id = raw.get("block_id")
    if block_id is not None:
        block_id = str(block_id).strip() or None
    return {
        "id": str(raw.get("id") or default_id).strip(),
        "status": status,
        "block_id": block_id,
        "rationale": str(raw.get("rationale") or "").strip(),
    }


def _ensure_rubric_checks_complete(
    checks: list[dict],
    rubric_items: list[dict],
) -> list[dict]:
    """Ensure every rubric item has exactly one check entry."""
    by_id = {c["id"]: c for c in checks if c.get("id")}
    out: list[dict] = []
    for item in rubric_items:
        item_id = item["id"]
        if item_id in by_id:
            out.append(by_id[item_id])
        else:
            out.append(
                {
                    "id": item_id,
                    "status": "MISSING",
                    "block_id": None,
                    "rationale": "Not evaluated.",
                }
            )
    return out


def _apply_carryover_guard(
    new_checks: list[dict],
    prior_checks: list[dict] | None,
    prior_block_hashes: dict[str, str],
    current_block_hashes: dict[str, str],
) -> list[dict]:
    """Reuse prior rubric status when the bound block content hash is unchanged."""
    if not prior_checks:
        return new_checks
    prior_by_id = {c["id"]: c for c in prior_checks if c.get("id")}
    out: list[dict] = []
    for check in new_checks:
        item_id = check.get("id")
        prior = prior_by_id.get(item_id)
        if not prior:
            out.append(check)
            continue
        block_id = check.get("block_id") or prior.get("block_id")
        if block_id:
            old_hash = prior_block_hashes.get(str(block_id))
            new_hash = current_block_hashes.get(str(block_id))
            if old_hash and new_hash and old_hash == new_hash:
                carried = dict(check)
                carried["status"] = prior.get("status", check.get("status"))
                carried["block_id"] = block_id
                carried["rationale"] = prior.get("rationale") or check.get("rationale", "")
                carried["carried_over"] = True
                out.append(carried)
                continue
        out.append(check)
    return out


def _format_prior_checks_for_prompt(prior_checks: list[dict]) -> str:
    if not prior_checks:
        return ""
    lines = []
    for c in prior_checks:
        lines.append(
            f'- id="{c.get("id")}" status={c.get("status")} block_id={c.get("block_id")!r} '
            f'rationale="{str(c.get("rationale") or "")[:120]}"'
        )
    return "\n".join(lines)


def _build_analysis_prompt(
    full_text: str,
    blocks: list[dict],
    legal_context: str,
    document_type: str,
    language: str,
    *,
    rubric_items: list[dict] | None = None,
    prior_checks: list[dict] | None = None,
    doc_char_limit: int = 12_000,
    blocks_limit: int = 50,
    block_char_limit: int = 200,
    legal_context_limit: int = 15_000,
) -> str:
    """Build the main analysis prompt with rubric checklist and output schema."""
    items = rubric_items or get_rubric_items_for_document_type(document_type)
    rubric_block = format_rubric_for_prompt(items)
    prior_block = _format_prior_checks_for_prompt(prior_checks or [])
    prior_section = ""
    if prior_block:
        prior_section = f"""
Previous rubric evaluation (baseline — keep each item's status UNLESS the text bound to that item materially changed):
---
{prior_block}
---
"""
    blocks_context = _format_blocks_for_prompt(
        blocks,
        blocks_limit=blocks_limit,
        block_char_limit=block_char_limit,
    )
    lang = normalize_language_code(language)
    if lang == "om":
        lang_rule = (
            f"Write summary, implications, recommendations, problem_summary, "
            f"rewrite_directive, and suggested_text in Afaan Oromoo ({language_label(lang)}). "
            f"Keep law names, article numbers, and citations in their original form. "
            f'Set editor_fix.document_language to "om".'
        )
    elif lang == "am":
        lang_rule = (
            f"Write those narrative fields in Amharic (አማርኛ). "
            f'Set editor_fix.document_language to "am".'
        )
    else:
        lang_rule = (
            "Write those narrative fields in English. "
            'Set editor_fix.document_language to "en".'
        )
    return f"""Analyze the following document for compliance with Ethiopian law. Output valid JSON only.

Document type: {document_type}
Language for response: {lang}
RESPONSE LANGUAGE (CRITICAL): {lang_rule}

RUBRIC EVALUATION (required — authoritative for scoring):
For EACH rubric id below, return exactly one entry in "rubric_checks" with:
- id: the rubric id (must match exactly)
- status: one of PRESENT | PARTIAL | MISSING | NON_COMPLIANT | NOT_APPLICABLE
- block_id: block_id from Document blocks that best supports your judgment, or null for document-level items
- rationale: one short sentence explaining the status

Rubric items (version {RUBRIC_VERSION}):
---
{rubric_block}
---

{RUBRIC_ETHIOPIAN_FRAMEWORK}
{prior_section}
Document blocks (use ONLY these block_id values when tying clauses, issues, or rubric_checks to the document; do not invent block_ids):
---
{blocks_context}
---

Document text (excerpt):
---
{full_text[:doc_char_limit]}
---

Relevant Ethiopian law (use for citations):
---
{legal_context[:legal_context_limit]}
---

When tying clauses, issues, or rubric_checks to the document, set block_id to one of the block_id values listed in Document blocks above. If no block matches, use null — never guess or invent a block_id.

CRITICAL block_id rules:
- block_id must reference the block whose TEXT is being analyzed — never a nearby label (e.g. "AND", "OR", "SERVICE PROVIDER:", "TO EMPLOYER:").
- Blocks marked NON-ANCHORABLE must never receive a block_id on any clause, issue, or rubric check.
- For document-level gaps (missing entire sections such as dispute resolution, limitation of liability, termination notice): set block_id to null and describe the gap in implications/issues/rationale. Do NOT attach these to unrelated blocks.
- Before flagging a provision as MISSING, search the full document text including numbered sub-clauses (e.g. 10.1, 10.2) under section headings. If body text already states the provision (e.g. "governed by the laws of Ethiopia"), mark PRESENT or PARTIAL — not MISSING.
- Prefer block_id of the substantive paragraph block (e.g. the 10.1 body text) over the section heading alone when analyzing a section.
- If a prior baseline is provided above, preserve each item's status unless the bound block text materially changed.

IMPORTANT: You MUST populate the "clauses" array. List each substantive clause or paragraph from the document: for each one give clause_id (e.g. clause_1, clause_2), text (excerpt of the clause), risk_level (LOW|MEDIUM|HIGH|CRITICAL), implications (legal implications in 1–2 sentences), block_id from the Document blocks list above (or null), and citations: []. For any clause with risk_level MEDIUM, HIGH, or CRITICAL, you MUST also populate ethiopian_law_implications (list of specific Ethiopian law implications for that clause) and recommendations (list of actionable steps to address the risk). Leave both as [] for LOW risk clauses. Do NOT return an empty "clauses" array when the document has content—include at least one clause per substantive paragraph or section.

For clauses with risk_level MEDIUM, HIGH, or CRITICAL, you MUST populate "editor_fix" (a structured edit spec for the document editor). For LOW clauses only, set "editor_fix": null. The editor_fix object is separate from recommendations: recommendations are human-facing advice; editor_fix powers automated rewrites. Keep every editor_fix field concise (short phrases, not paragraphs). editor_fix.rewrite_directive and editor_fix.suggested_text must be imperative rewrite instructions with concrete replacement text — NOT restatements of law like "Ethiopian law requires...". Use [BRACKETED_PLACEHOLDERS] in suggested_text when specific values are unknown. Set editor_fix.block_id to the clause block_id when known. Set editor_fix.severity to medium_risk for MEDIUM, high_risk for HIGH, or critical_risk for CRITICAL. Example: BAD rewrite_directive: "Ethiopian law requires clear compensation terms." GOOD rewrite_directive: "Rewrite to state base salary, payment frequency, and benefits explicitly." GOOD suggested_text: "The Employee shall receive a monthly base salary of [AMOUNT] ETB, payable [FREQUENCY]..."

Output a single JSON object with this structure (use empty arrays only for issues/citations/missing_clauses if none; clauses must be non-empty when the document has text; rubric_checks MUST include every rubric id listed above):
{{
  "document_type": "string (detected or given)",
  "summary": "string (executive summary)",
  "rubric_checks": [{{ "id": "string (rubric id)", "status": "PRESENT|PARTIAL|MISSING|NON_COMPLIANT|NOT_APPLICABLE", "block_id": "string or null", "rationale": "string" }}],
  "clauses": [{{ "clause_id": "string", "text": "string", "risk_level": "LOW|MEDIUM|HIGH|CRITICAL", "implications": "string", "block_id": "string or null", "citations": [], "ethiopian_law_implications": ["string"], "recommendations": ["string"], "editor_fix": null or {{ "action": "replace", "block_id": "string or null", "clause_reference": "string", "current_text": "string", "problem_summary": "string", "offending_phrases": ["string"], "legal_requirement": "string", "rewrite_directive": "string", "remove_phrases": ["string"], "add_elements": ["string"], "suggested_text": "string", "placeholder_policy": "use_bracketed_placeholders_when_values_unknown", "legal_basis": [{{ "source": "string", "article": "string", "rationale": "string" }}], "document_language": "string", "severity": "medium_risk | high_risk | critical_risk", "confidence": 0.0 to 1.0 }} }}],
  "issues": [{{ "issue_id": "string", "description": "string", "severity": "LOW|MEDIUM|HIGH|CRITICAL", "block_id": "string or null", "citations": [] }}],
  "ethiopian_law_compliance": {{ "summary": "string", "applicable_laws": ["string"], "concerns": ["string"] }},
  "recommendations": ["string"],
  "should_sign": true or false or null,
  "critical_issues": [same as issues],
  "missing_clauses": ["string"]
}}

Output only the JSON object, no markdown or explanation."""


def _normalize_llm_json_text(raw: str) -> str:
    """Strip markdown fences, preamble, and common LLM JSON formatting issues."""
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    text = match.group(1).strip() if match else re.sub(r"^```(?:json)?\s*", "", text).strip()
    text = re.sub(r"\s*```\s*$", "", text).strip()
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
    # Remove trailing commas before closing braces/brackets.
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _repair_truncated_json(text: str, error_pos: int) -> str | None:
    """Attempt to repair JSON truncated mid-string by closing the string and any open braces/brackets."""
    if error_pos <= 0 or error_pos > len(text):
        return None
    s = text[:error_pos].rstrip()
    # Track open brackets/braces and whether we're inside a string (to close truncation)
    in_string = False
    escape = False
    stack: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == "\\" and in_string:
            escape = True
            i += 1
            continue
        if c == '"':
            in_string = not in_string
            i += 1
            continue
        if not in_string:
            if c == "{":
                stack.append("}")
            elif c == "[":
                stack.append("]")
            elif (c == "}" or c == "]") and stack and stack[-1] == c:
                stack.pop()
        i += 1
    if in_string:
        s += '"'
    repaired = s + "".join(reversed(stack))
    return repaired


def _try_parse_json_text(text: str) -> dict[str, Any]:
    """Parse JSON text with repair attempts for truncated LLM output."""
    try:
        out = json.loads(text)
        if not isinstance(out, dict):
            raise json.JSONDecodeError("Expected JSON object", text, 0)
        return out
    except json.JSONDecodeError as e:
        repair_positions = [len(text)]
        if e.pos:
            repair_positions.append(e.pos)
        for pos in repair_positions:
            repaired = _repair_truncated_json(text, pos)
            if not repaired:
                continue
            try:
                out = json.loads(repaired)
                if isinstance(out, dict):
                    return out
            except json.JSONDecodeError:
                continue
        raise


def _salvage_truncated_json(raw: str) -> dict[str, Any] | None:
    """Best-effort recovery when the model hit max_tokens and cut off mid-object."""
    text = _normalize_llm_json_text(raw)
    if "{" not in text:
        return None

    # Walk backward in coarse steps, closing any open structures.
    for end in range(len(text), max(0, len(text) - 30_000), -100):
        chunk = text[:end].rstrip().rstrip(",")
        repaired = _repair_truncated_json(chunk, len(chunk))
        if not repaired:
            continue
        try:
            out = json.loads(repaired)
        except json.JSONDecodeError:
            continue
        if isinstance(out, dict) and (out.get("clauses") or out.get("document_type")):
            return out
    return None


def _parse_analysis_response(raw: str) -> dict[str, Any]:
    """Parse LLM JSON response; strip markdown/preamble and repair truncated output when possible."""
    text = _normalize_llm_json_text(raw)
    try:
        return _try_parse_json_text(text)
    except json.JSONDecodeError:
        salvaged = _salvage_truncated_json(raw)
        if salvaged is not None:
            log.warning(
                "compliance_parse_salvaged",
                extra={"event": "compliance_parse_salvaged", "raw_len": len(raw)},
            )
            return salvaged
        raise


def _validate_and_dedupe(data: dict) -> dict:
    """Ensure unique clause_id/issue_id, normalize risk levels."""
    seen_clause: set[str] = set()
    seen_issue: set[str] = set()
    clauses = []
    for c in data.get("clauses", []) or []:
        cid = c.get("clause_id") or f"c{len(clauses)}"
        if cid in seen_clause:
            continue
        seen_clause.add(cid)
        c["clause_id"] = cid
        clauses.append(c)
    data["clauses"] = clauses

    issues = []
    for i in data.get("issues", []) or []:
        iid = i.get("issue_id") or f"i{len(issues)}"
        if iid in seen_issue:
            continue
        seen_issue.add(iid)
        i["issue_id"] = iid
        issues.append(i)
    data["issues"] = issues
    return data


def _clauses_fallback_from_blocks(blocks: list[dict]) -> list[dict]:
    """When LLM returns no clauses, derive one clause per block with text so response isn't empty."""
    out: list[dict] = []
    for i, b in enumerate(blocks):
        text = (b.get("text") or "").strip()
        if not text:
            continue
        out.append(
            {
                "clause_id": f"clause_{i + 1}",
                "text": text[:2000],
                "risk_level": "LOW",
                "implications": "",
                "block_id": b.get("block_id"),
                "citations": [],
            }
        )
    return out


class ComplianceAnalysisAgent:
    """Single-run compliance analysis: targeted retrieval, LLM analysis, per-clause citations."""

    def analyze_document(
        self,
        document_text: str | None = None,
        tiptap_json: dict[str, Any] | None = None,
        document_blocks: list[dict[str, Any]] | None = None,
        language: str = "en",
        document_type: str | None = None,
        check_level: str = "quick",
        doc_id: str | None = None,
        prior_rubric_result: dict | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        token_callback: Callable[[str], None] | None = None,
    ) -> ComplianceAnalysisResponse:
        """Run full pipeline and return ComplianceAnalysisResponse. If document_blocks (e.g. from Qdrant) is provided, use it for full_text and block context (block_id, type). check_level (quick/standard/deep) controls context and citation depth.

        If progress_callback is set, it is invoked with ``{"phase": str, "percent": int, "message": str}`` at coarse pipeline boundaries (and during per-clause citation when applicable). Percent is approximate (0–100).

        When doc_id is provided, diff-aware anchoring reuses prior rubric statuses for unchanged blocks.
        """
        if document_blocks:
            blocks = document_blocks
            full_text = "\n\n".join(b.get("text", "") for b in blocks if b.get("text"))
        else:
            full_text = _extract_full_document_text(tiptap_json, document_text)
            blocks = extract_blocks_from_tiptap(tiptap_json) if tiptap_json else []
        if not full_text:
            raise ValueError("No document text to analyze")
        language = normalize_language_code(language)
        _emit_progress(
            progress_callback,
            phase="prepare",
            percent=5,
            message="Document loaded; starting compliance check",
        )
        limits = get_compliance_limits(check_level)
        doc_type = document_type or _detect_document_type(full_text)
        rubric_items = get_rubric_items_for_document_type(doc_type)
        content_hash = compute_document_content_hash(blocks)
        per_block_hashes = compute_per_block_hashes(blocks)

        prior_result = prior_rubric_result
        if prior_result is None and doc_id:
            prior_result = get_last_rubric_result(doc_id)
        prior_checks: list[dict] = []
        prior_block_hashes: dict[str, str] = {}
        if prior_result and prior_result.get("rubric_version") == RUBRIC_VERSION:
            prior_checks = [
                _normalize_rubric_check(c) for c in (prior_result.get("checks") or [])
            ]
            prior_block_hashes = prior_result.get("per_block_hashes") or {}

        summary_for_queries = full_text[:500]

        # 1) Targeted query generation
        queries = generate_targeted_queries(doc_type, summary_for_queries)
        if not queries:
            queries = [f"Ethiopian law {doc_type}"]
        _emit_progress(
            progress_callback,
            phase="queries",
            percent=15,
            message="Generated search queries for legal knowledge",
        )

        # 2) Optional source filter (use all sources for now; can add LLM pick later)
        source_filter: list[str] | None = None
        # source_filter = get_available_source_files()  # then filter by doc_type if needed

        # 3) Retrieve and rerank (level-specific limits)
        retrieved = search_legal_knowledge(
            queries,
            source_filter=source_filter,
            top_k_per_query=limits["top_k_per_query"],
            rrf_top_k=limits["initial_limit"],
        )
        _emit_progress(
            progress_callback,
            phase="retrieve",
            percent=32,
            message="Retrieved relevant legal sources",
        )
        reranked = rerank_with_llm(queries[0], retrieved, limits["rerank_top"])
        legal_context = _format_legal_context(reranked)
        global_citations = _docs_to_citations(reranked)
        _emit_progress(
            progress_callback,
            phase="rerank",
            percent=40,
            message="Ranked sources for analysis context",
        )

        # 4) Main LLM analysis (level-specific prompt truncation)
        llm = _build_analysis_llm()
        prompt = _build_analysis_prompt(
            full_text,
            blocks,
            legal_context,
            doc_type,
            language,
            rubric_items=rubric_items,
            prior_checks=prior_checks if prior_checks else None,
            doc_char_limit=limits["doc_char_limit"],
            blocks_limit=limits["blocks_limit"],
            block_char_limit=limits["block_char_limit"],
            legal_context_limit=limits["legal_context_limit"],
        )
        _emit_progress(
            progress_callback,
            phase="analyze",
            percent=42,
            message="Analyzing document against Ethiopian law (this step may take a while)",
        )
        raw_parts: list[str] = []
        for chunk in llm.stream(prompt):
            content = getattr(chunk, "content", "")
            if not content:
                continue
            if isinstance(content, list):
                text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            else:
                text = str(content)
            if not text:
                continue
            raw_parts.append(text)
            if token_callback is not None:
                token_callback(text)
        raw = "".join(raw_parts).strip()
        try:
            data = _parse_analysis_response(raw)
        except json.JSONDecodeError as e:
            log.error(
                "compliance_parse_error",
                extra={
                    "event": "compliance_parse_error",
                    "error": str(e),
                    "raw_len": len(raw),
                    "raw_preview": raw[:500],
                },
            )
            raise ValueError(
                "Compliance analysis could not be parsed. The document may be too long for a single pass — "
                "try again with a shorter document or a quicker check level."
            ) from e

        _emit_progress(
            progress_callback,
            phase="parse",
            percent=68,
            message="Parsed analysis results",
        )
        data = _validate_and_dedupe(data)
        # Fallback: if LLM returned no clauses but we have blocks with text, derive minimal clauses so response isn't empty
        if not (data.get("clauses") or []) and blocks:
            data["clauses"] = _clauses_fallback_from_blocks(blocks)
            log.info(
                "compliance_clauses_fallback",
                extra={"event": "clauses_from_blocks", "count": len(data["clauses"])},
            )
        clauses = data.get("clauses", []) or []
        issues = data.get("issues", []) or []
        _map_clauses_to_blocks(clauses, blocks)
        _map_issues_to_blocks(issues, blocks)
        _sanitize_clause_and_issue_block_ids(clauses, issues, blocks)
        _filter_false_missing_provision_findings(clauses, issues, blocks, full_text)
        corpus = _document_corpus(full_text, blocks)
        data["missing_clauses"] = _filter_missing_clauses_list(
            data.get("missing_clauses") or [], corpus
        )
        data["clauses"] = clauses
        data["issues"] = issues

        # Rubric checks: normalize, diff-anchor carry-over, ensure completeness
        raw_rubric = [
            _normalize_rubric_check(c) for c in (data.get("rubric_checks") or [])
        ]
        rubric_checks = _ensure_rubric_checks_complete(raw_rubric, rubric_items)
        rubric_checks = _apply_carryover_guard(
            rubric_checks,
            prior_checks,
            prior_block_hashes,
            per_block_hashes,
        )
        valid_ids = _valid_block_ids(blocks)
        for check in rubric_checks:
            bid = check.get("block_id")
            if bid and str(bid) not in valid_ids:
                check["block_id"] = None
        data["rubric_checks"] = rubric_checks

        # 5) Per-clause citations for non-LOW risk clauses with implications (level-specific)
        max_clauses_citations = limits["max_clauses_citations"]
        clauses_with_citations = 0
        citation_budget = min(
            max_clauses_citations,
            sum(
                1
                for c in (data.get("clauses", []) or [])
                if (c.get("risk_level") or "").upper() != "LOW"
                and (c.get("implications") or "").strip()
            ),
        )
        for c in data.get("clauses", []) or []:
            if clauses_with_citations >= max_clauses_citations:
                break
            if (c.get("risk_level") or "").upper() == "LOW":
                continue
            if not (c.get("implications") or "").strip():
                continue
            clause_text = c.get("text", "") or ""
            implications = c.get("implications", "") or ""
            # Concept-focused query + dynamic synonyms from LLM (no raw clause dump)
            query = build_clause_legal_query(clause_text, implications)
            if not query.strip():
                query = f"{clause_text[:150]} {implications[:150]}".strip()
            if not query:
                continue
            log.info(
                "per_clause_legal_query",
                extra={
                    "event": "compliance_per_clause_query",
                    "clause_id": c.get("clause_id"),
                    "legal_query": query,
                },
            )
            per_docs = search_legal_knowledge(
                [query],
                source_filter=source_filter,
                top_k_per_query=limits["per_clause_top_k"],
                rrf_top_k=limits["per_clause_rrf_k"],
            )
            per_reranked = rerank_with_llm(query, per_docs, limits["per_clause_rerank_k"])
            c["citations"] = [cit.model_dump() for cit in _docs_to_citations(per_reranked)]
            clauses_with_citations += 1
            if citation_budget > 0 and progress_callback is not None:
                sub = 70 + int(24 * clauses_with_citations / citation_budget)
                _emit_progress(
                    progress_callback,
                    phase="clause_citations",
                    percent=sub,
                    message=f"Resolved citations for clause {clauses_with_citations} of {citation_budget}",
                )

        # 6) Build issues_by_block_id and critical_issues (only valid document block_ids)
        issues = data.get("issues", []) or []
        valid_ids = _valid_block_ids(blocks)
        issues_by_block_id: dict[str, list[dict]] = {}
        critical_issues: list[dict] = []
        for i in issues:
            bid = i.get("block_id")
            if bid and str(bid) in valid_ids:
                issues_by_block_id.setdefault(str(bid), []).append(i)
            if (i.get("severity") or "").upper() in ("HIGH", "CRITICAL"):
                critical_issues.append(i)

        # Deterministic scoring: authoritative score from rubric statuses
        max_penalty = getattr(settings, "COMPLIANCE_SCORE_MAX_PENALTY", DEFAULT_MAX_PENALTY)
        eth_data_raw = data.get("ethiopian_law_compliance") or {}
        eth_concerns_raw = eth_data_raw.get("concerns") or []
        rubric_breakdown = compute_rubric_score(
            rubric_checks,
            version=RUBRIC_VERSION,
            max_penalty=max_penalty,
        )
        legacy_breakdown = compute_risk_score(
            clauses=data.get("clauses", []) or [],
            issues=data.get("issues", []) or [],
            missing_clauses=data.get("missing_clauses", []) or [],
            should_sign=data.get("should_sign"),
            concern_count=len(eth_concerns_raw),
            max_penalty=max_penalty,
        )
        score_breakdown = {
            **legacy_breakdown,
            "rubric": rubric_breakdown,
            "content_hash": content_hash,
        }
        risk_score: float = rubric_breakdown["risk_score"]
        compliance_score: float = rubric_breakdown["compliance_score"]
        overall_risk_level: str = rubric_breakdown["overall_risk_level"]

        if doc_id:
            store_last_rubric_result(
                doc_id,
                content_hash=content_hash,
                per_block_hashes=per_block_hashes,
                checks=rubric_checks,
            )

        eth = eth_data_raw
        eth_compliance = EthiopianLawCompliance(
            summary=eth.get("summary", ""),
            applicable_laws=eth.get("applicable_laws", []) or [],
            concerns=eth.get("concerns", []) or [],
        )

        def _to_citation(cit: dict | Any) -> LegalCitation | None:
            d = cit if isinstance(cit, dict) else getattr(cit, "model_dump", lambda: cit)()
            if not isinstance(d, dict):
                return None
            doc_id = str(d.get("document_id", "") or "").strip()
            item_id = str(d.get("item_id", "") or "").strip()
            title = str(d.get("title", "") or "").strip()[:500]
            excerpt = str(d.get("excerpt", "") or "").strip()[:500]
            if not doc_id and not item_id and not title and not excerpt:
                return None
            return LegalCitation(document_id=doc_id, item_id=item_id, title=title, excerpt=excerpt)

        def _filter_citations(cits: list) -> list[LegalCitation]:
            out = []
            for cit in cits or []:
                c = _to_citation(cit)
                if c is not None:
                    out.append(c)
            return out

        def to_issue(i: dict, *, block_id_override: str | None = None) -> LegalIssue:
            raw_severity = (i.get("severity") or "LOW").upper()
            description = str(i.get("description") or "").strip()
            block_id = str(i.get("block_id") or block_id_override or "").strip()
            clause_text = str(i.get("clause_text") or "").strip() or None
            return LegalIssue(
                block_id=block_id,
                severity=_risk_level_to_issue_severity(raw_severity),
                issue_type=str(i.get("issue_type") or _determine_issue_type(raw_severity, description)),
                description=description,
                risk_factors=[str(x) for x in (i.get("risk_factors") or []) if x],
                ethiopian_law_implications=[
                    str(x) for x in (i.get("ethiopian_law_implications") or []) if x
                ],
                recommendations=[str(x) for x in (i.get("recommendations") or []) if x],
                citations=_citations_to_str_list(i.get("citations", [])),
                clause_text=clause_text,
                clause_excerpt=(clause_text[:120] if clause_text else None),
                issue_id=i.get("issue_id"),
            )

        _emit_progress(
            progress_callback,
            phase="finalize",
            percent=96,
            message="Computing risk scores and assembling response",
        )

        def to_clause(c: dict) -> ClauseAnalysis:
            return ClauseAnalysis(
                clause_id=c.get("clause_id", ""),
                text=c.get("text", ""),
                risk_level=c.get("risk_level", "LOW"),
                implications=c.get("implications", ""),
                block_id=c.get("block_id"),
                citations=_filter_citations(c.get("citations", [])),
                ethiopian_law_implications=c.get("ethiopian_law_implications", []) or [],
                recommendations=c.get("recommendations", []) or [],
                editor_fix=_normalize_editor_fix(c.get("editor_fix"), c, blocks, language),
            )

        return ComplianceAnalysisResponse(
            document_type=data.get("document_type", doc_type),
            overall_risk_level=overall_risk_level,
            risk_score=risk_score,
            compliance_score=compliance_score,
            summary=data.get("summary", ""),
            clauses=[to_clause(c) for c in data.get("clauses", [])],
            issues_by_block_id={
                k: [to_issue(i, block_id_override=k) for i in v] for k, v in issues_by_block_id.items()
            },
            ethiopian_law_compliance=eth_compliance,
            recommendations=data.get("recommendations", []) or [],
            should_sign=data.get("should_sign"),
            critical_issues=[to_issue(i) for i in critical_issues],
            missing_clauses=data.get("missing_clauses", []) or [],
            citations=global_citations,
            score_breakdown=score_breakdown,
        )
