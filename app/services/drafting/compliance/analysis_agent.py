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
from app.services.drafting.compliance.scoring import (
    DEFAULT_MAX_PENALTY,
    compute_risk_score,
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


# Checklist for the analysis prompt (reduce missed issues)
COMPLIANCE_CHECKLIST = """
Consider at least: jurisdiction and governing law, contract type, parties' obligations, liability and limitation of liability, termination and notice, confidentiality, IP and assignment, dispute resolution and venue, mandatory law (Ethiopian), missing standard clauses for this document type, unfair terms.""".strip()


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
    """Extract blocks (paragraphs/sections) with optional block_id and text from TipTap JSON."""
    blocks: list[dict[str, Any]] = []
    content = tiptap_json.get("content") or []
    if not content:
        return blocks

    for i, node in enumerate(content):
        if not isinstance(node, dict):
            continue
        node_type = node.get("type", "")
        text = _extract_text_from_tiptap_node(node).strip()
        attrs = node.get("attrs") or {}
        block_id = attrs.get("id") or attrs.get("blockId") or (f"b{i}" if text else None)
        if text or node_type in ("heading", "paragraph", "blockquote"):
            blocks.append({"block_id": block_id or f"b{i}", "type": node_type, "text": text})
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
        lines.append(f"[block_id: {block_id} | type: {block_type}] {excerpt}")
    return "\n".join(lines) if lines else "(no structured blocks available)"


def _valid_block_ids(blocks: list[dict]) -> set[str]:
    return {str(b["block_id"]) for b in blocks if b.get("block_id")}


def _normalize_match_text(text: str) -> str:
    """Lowercase and collapse whitespace for fuzzy block text matching."""
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _match_block_by_text(text: str, blocks: list[dict]) -> str | None:
    """Return block_id whose text best matches the given excerpt."""
    excerpt = _normalize_match_text(text[:300])
    if len(excerpt) < 8:
        return None
    excerpt_core = excerpt.rstrip(".,;:!?")
    for b in blocks:
        block_text = _normalize_match_text(b.get("text") or "")
        if not block_text:
            continue
        block_core = block_text.rstrip(".,;:!?")
        if (
            block_text in excerpt
            or excerpt in block_text
            or (excerpt_core and excerpt_core in block_text)
            or (block_core and block_core in excerpt)
        ):
            bid = b.get("block_id")
            if bid:
                return str(bid)
    return None


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


def _build_analysis_prompt(
    full_text: str,
    blocks: list[dict],
    legal_context: str,
    document_type: str,
    language: str,
    *,
    doc_char_limit: int = 12_000,
    blocks_limit: int = 50,
    block_char_limit: int = 200,
    legal_context_limit: int = 15_000,
) -> str:
    """Build the main analysis prompt with checklist and output schema."""
    blocks_context = _format_blocks_for_prompt(
        blocks,
        blocks_limit=blocks_limit,
        block_char_limit=block_char_limit,
    )
    return f"""Analyze the following document for compliance with Ethiopian law. Output valid JSON only.

Document type: {document_type}
Language for response: {language}

Checklist of areas to consider: {COMPLIANCE_CHECKLIST}

Document blocks (use ONLY these block_id values when tying clauses or issues to the document; do not invent block_ids):
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

When tying clauses or issues to the document, set block_id to one of the block_id values listed in Document blocks above. If no block matches, use null — never guess or invent a block_id.

IMPORTANT: You MUST populate the "clauses" array. List each substantive clause or paragraph from the document: for each one give clause_id (e.g. clause_1, clause_2), text (excerpt of the clause), risk_level (LOW|MEDIUM|HIGH|CRITICAL), implications (legal implications in 1–2 sentences), block_id from the Document blocks list above (or null), and citations: []. For any clause with risk_level MEDIUM, HIGH, or CRITICAL, you MUST also populate ethiopian_law_implications (list of specific Ethiopian law implications for that clause) and recommendations (list of actionable steps to address the risk). Leave both as [] for LOW risk clauses. Do NOT return an empty "clauses" array when the document has content—include at least one clause per substantive paragraph or section.

For clauses with risk_level MEDIUM, HIGH, or CRITICAL, you MUST populate "editor_fix" (a structured edit spec for the document editor). For LOW clauses only, set "editor_fix": null. The editor_fix object is separate from recommendations: recommendations are human-facing advice; editor_fix powers automated rewrites. Keep every editor_fix field concise (short phrases, not paragraphs). editor_fix.rewrite_directive and editor_fix.suggested_text must be imperative rewrite instructions with concrete replacement text — NOT restatements of law like "Ethiopian law requires...". Use [BRACKETED_PLACEHOLDERS] in suggested_text when specific values are unknown. Set editor_fix.block_id to the clause block_id when known. Set editor_fix.severity to medium_risk for MEDIUM, high_risk for HIGH, or critical_risk for CRITICAL. Example: BAD rewrite_directive: "Ethiopian law requires clear compensation terms." GOOD rewrite_directive: "Rewrite to state base salary, payment frequency, and benefits explicitly." GOOD suggested_text: "The Employee shall receive a monthly base salary of [AMOUNT] ETB, payable [FREQUENCY]..."

Output a single JSON object with this structure (use empty arrays only for issues/citations/missing_clauses if none; clauses must be non-empty when the document has text):
{{
  "document_type": "string (detected or given)",
  "summary": "string (executive summary)",
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
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        token_callback: Callable[[str], None] | None = None,
    ) -> ComplianceAnalysisResponse:
        """Run full pipeline and return ComplianceAnalysisResponse. If document_blocks (e.g. from Qdrant) is provided, use it for full_text and block context (block_id, type). check_level (quick/standard/deep) controls context and citation depth.

        If progress_callback is set, it is invoked with ``{"phase": str, "percent": int, "message": str}`` at coarse pipeline boundaries (and during per-clause citation when applicable). Percent is approximate (0–100).
        """
        if document_blocks:
            blocks = document_blocks
            full_text = "\n\n".join(b.get("text", "") for b in blocks if b.get("text"))
        else:
            full_text = _extract_full_document_text(tiptap_json, document_text)
            blocks = extract_blocks_from_tiptap(tiptap_json) if tiptap_json else []
        if not full_text:
            raise ValueError("No document text to analyze")
        _emit_progress(
            progress_callback,
            phase="prepare",
            percent=5,
            message="Document loaded; starting compliance check",
        )
        limits = get_compliance_limits(check_level)
        doc_type = document_type or _detect_document_type(full_text)
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
        model = settings.COMPLIANCE_ANALYSIS_MODEL or settings.GEMINI_MODEL
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
            model=model,
            temperature=getattr(settings, "COMPLIANCE_ANALYSIS_TEMPERATURE", 0.1),
            max_tokens=getattr(settings, "COMPLIANCE_ANALYSIS_MAX_TOKENS", 32768),
            streaming=True,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        prompt = _build_analysis_prompt(
            full_text,
            blocks,
            legal_context,
            doc_type,
            language,
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
        data["clauses"] = clauses
        data["issues"] = issues

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

        # 6) Build issues_by_block_id and critical_issues
        issues = data.get("issues", []) or []
        issues_by_block_id: dict[str, list[dict]] = {}
        critical_issues: list[dict] = []
        for i in issues:
            bid = i.get("block_id")
            if bid:
                issues_by_block_id.setdefault(bid, []).append(i)
            if (i.get("severity") or "").upper() in ("HIGH", "CRITICAL"):
                critical_issues.append(i)

        # Deterministic scoring: derive risk_score, compliance_score, and overall_risk_level
        # from the LLM's clause/issue severity classifications. No LLM math involved.
        max_penalty = getattr(settings, "COMPLIANCE_SCORE_MAX_PENALTY", DEFAULT_MAX_PENALTY)
        eth_data_raw = data.get("ethiopian_law_compliance") or {}
        eth_concerns_raw = eth_data_raw.get("concerns") or []
        score_breakdown = compute_risk_score(
            clauses=data.get("clauses", []) or [],
            issues=data.get("issues", []) or [],
            missing_clauses=data.get("missing_clauses", []) or [],
            should_sign=data.get("should_sign"),
            concern_count=len(eth_concerns_raw),
            max_penalty=max_penalty,
        )
        risk_score: float = score_breakdown["risk_score"]
        compliance_score: float = score_breakdown["compliance_score"]
        overall_risk_level: str = score_breakdown["overall_risk_level"]

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
