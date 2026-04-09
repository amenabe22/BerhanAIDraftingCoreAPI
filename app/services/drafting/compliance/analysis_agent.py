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
    EthiopianLawCompliance,
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
    return f"""Analyze the following document for compliance with Ethiopian law. Output valid JSON only.

Document type: {document_type}
Language for response: {language}

Checklist of areas to consider: {COMPLIANCE_CHECKLIST}

Document text (excerpt):
---
{full_text[:doc_char_limit]}
---

Relevant Ethiopian law (use for citations):
---
{legal_context[:legal_context_limit]}
---

If blocks were provided, reference block_id when tying issues to specific clauses.

IMPORTANT: You MUST populate the "clauses" array. List each substantive clause or paragraph from the document: for each one give clause_id (e.g. clause_1, clause_2), text (excerpt of the clause), risk_level (LOW|MEDIUM|HIGH|CRITICAL), implications (legal implications in 1–2 sentences), block_id if you can match to a block above, and citations: []. For any clause with risk_level MEDIUM, HIGH, or CRITICAL, you MUST also populate ethiopian_law_implications (list of specific Ethiopian law implications for that clause) and recommendations (list of actionable steps to address the risk). Leave both as [] for LOW risk clauses. Do NOT return an empty "clauses" array when the document has content—include at least one clause per substantive paragraph or section.

Output a single JSON object with this structure (use empty arrays only for issues/citations/missing_clauses if none; clauses must be non-empty when the document has text):
{{
  "document_type": "string (detected or given)",
  "summary": "string (executive summary)",
  "clauses": [{{ "clause_id": "string", "text": "string", "risk_level": "LOW|MEDIUM|HIGH|CRITICAL", "implications": "string", "block_id": "string or null", "citations": [], "ethiopian_law_implications": ["string"], "recommendations": ["string"] }}],
  "issues": [{{ "issue_id": "string", "description": "string", "severity": "LOW|MEDIUM|HIGH|CRITICAL", "block_id": "string or null", "citations": [] }}],
  "ethiopian_law_compliance": {{ "summary": "string", "applicable_laws": ["string"], "concerns": ["string"] }},
  "recommendations": ["string"],
  "should_sign": true or false or null,
  "critical_issues": [same as issues],
  "missing_clauses": ["string"]
}}

Output only the JSON object, no markdown or explanation."""


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


def _parse_analysis_response(raw: str) -> dict[str, Any]:
    """Parse LLM JSON response; strip markdown code block if present (closed or unclosed). Repair if truncated."""
    text = raw.strip()
    # Extract content inside ``` ... ``` if both fences present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    text = match.group(1).strip() if match else re.sub(r"^```(?:json)?\s*", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if "Unterminated string" in e.msg or ("Expecting value" in e.msg and e.pos):
            # Try repair at end of text first (LLM output cut off at end), then at parser position
            for pos in (len(text), e.pos):
                repaired = _repair_truncated_json(text, pos)
                if repaired:
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        continue
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


def _map_clauses_to_blocks(clauses: list[dict], blocks: list[dict]) -> None:
    """Assign block_id to clauses by matching text/order where possible."""
    if not blocks:
        return
    for c in clauses:
        if c.get("block_id"):
            continue
        text = (c.get("text") or "")[:300]
        for b in blocks:
            if (
                b.get("text")
                and text
                and b.get("text", "").strip() in text
                or text in b.get("text", "")
            ):
                c["block_id"] = b.get("block_id")
                break


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
            max_tokens=getattr(settings, "COMPLIANCE_ANALYSIS_MAX_TOKENS", 16384),
            streaming=False,
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
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        try:
            data = _parse_analysis_response(raw)
        except json.JSONDecodeError as e:
            log.error("compliance_parse_error", extra={"error": str(e), "raw_preview": raw[:500]})
            raise ValueError("LLM did not return valid JSON") from e

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
        _map_clauses_to_blocks(data.get("clauses", []), blocks)

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

        def to_issue(i: dict) -> LegalIssue:
            return LegalIssue(
                issue_id=i.get("issue_id", ""),
                description=i.get("description", ""),
                severity=i.get("severity", "MEDIUM"),
                block_id=i.get("block_id"),
                citations=_filter_citations(i.get("citations", [])),
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
            )

        return ComplianceAnalysisResponse(
            document_type=data.get("document_type", doc_type),
            overall_risk_level=overall_risk_level,
            risk_score=risk_score,
            compliance_score=compliance_score,
            summary=data.get("summary", ""),
            clauses=[to_clause(c) for c in data.get("clauses", [])],
            issues_by_block_id={k: [to_issue(i) for i in v] for k, v in issues_by_block_id.items()},
            ethiopian_law_compliance=eth_compliance,
            recommendations=data.get("recommendations", []) or [],
            should_sign=data.get("should_sign"),
            critical_issues=[to_issue(i) for i in critical_issues],
            missing_clauses=data.get("missing_clauses", []) or [],
            citations=global_citations,
            score_breakdown=score_breakdown,
        )
