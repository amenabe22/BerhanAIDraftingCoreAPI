"""Compliance analysis agent: extract -> targeted retrieval -> LLM -> per-clause citations -> response."""

import json
import re
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
from app.services.drafting.knowledge_retrieval import (
    generate_targeted_queries,
    get_available_source_files,
    rerank_with_llm,
    search_legal_knowledge,
)

log = get_logger("compliance_agent")

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


def _extract_full_document_text(tiptap_json: dict[str, Any] | None, document_text: str | None) -> str:
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
    blocks_preview = "\n".join(
        f"- {b.get('block_id', '')}: {b.get('text', '')[:block_char_limit]}"
        for b in blocks[:blocks_limit]
        if b.get("text")
    )
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

If blocks were provided, reference block_id when tying issues to specific clauses. Output a single JSON object with this structure (use empty arrays/objects where needed):
{{
  "document_type": "string (detected or given)",
  "overall_risk_level": "LOW | MEDIUM | HIGH | CRITICAL",
  "risk_score": number 0-100,
  "summary": "string (executive summary)",
  "clauses": [{{ "clause_id": "string", "text": "string", "risk_level": "LOW|MEDIUM|HIGH|CRITICAL", "implications": "string", "block_id": "string or null", "citations": [] }}],
  "issues": [{{ "issue_id": "string", "description": "string", "severity": "LOW|MEDIUM|HIGH|CRITICAL", "block_id": "string or null", "citations": [] }}],
  "ethiopian_law_compliance": {{ "summary": "string", "applicable_laws": ["string"], "concerns": ["string"] }},
  "recommendations": ["string"],
  "should_sign": true or false or null,
  "critical_issues": [same as issues],
  "missing_clauses": ["string"]
}}

Output only the JSON object, no markdown or explanation."""


def _parse_analysis_response(raw: str) -> dict[str, Any]:
    """Parse LLM JSON response; strip markdown code block if present."""
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


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


def _map_clauses_to_blocks(clauses: list[dict], blocks: list[dict]) -> None:
    """Assign block_id to clauses by matching text/order where possible."""
    if not blocks:
        return
    for c in clauses:
        if c.get("block_id"):
            continue
        text = (c.get("text") or "")[:300]
        for b in blocks:
            if b.get("text") and text and b.get("text", "").strip() in text or text in b.get("text", ""):
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
    ) -> ComplianceAnalysisResponse:
        """Run full pipeline and return ComplianceAnalysisResponse. If document_blocks (e.g. from Qdrant) is provided, use it for full_text and block context (block_id, type). check_level (quick/standard/deep) controls context and citation depth."""
        if document_blocks:
            blocks = document_blocks
            full_text = "\n\n".join(b.get("text", "") for b in blocks if b.get("text"))
        else:
            full_text = _extract_full_document_text(tiptap_json, document_text)
            blocks = extract_blocks_from_tiptap(tiptap_json) if tiptap_json else []
        if not full_text:
            raise ValueError("No document text to analyze")
        limits = get_compliance_limits(check_level)
        doc_type = document_type or _detect_document_type(full_text)
        summary_for_queries = full_text[:500]

        # 1) Targeted query generation
        queries = generate_targeted_queries(doc_type, summary_for_queries)
        if not queries:
            queries = [f"Ethiopian law {doc_type}"]

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
        reranked = rerank_with_llm(queries[0], retrieved, limits["rerank_top"])
        legal_context = _format_legal_context(reranked)
        global_citations = _docs_to_citations(reranked)

        # 4) Main LLM analysis (level-specific prompt truncation)
        model = settings.COMPLIANCE_ANALYSIS_MODEL or settings.GEMINI_MODEL
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY,
            model=model,
            temperature=getattr(settings, "COMPLIANCE_ANALYSIS_TEMPERATURE", 0.1),
            max_tokens=getattr(settings, "COMPLIANCE_ANALYSIS_MAX_TOKENS", 8192),
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
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        try:
            data = _parse_analysis_response(raw)
        except json.JSONDecodeError as e:
            log.error("compliance_parse_error", extra={"error": str(e), "raw_preview": raw[:500]})
            raise ValueError("LLM did not return valid JSON") from e

        data = _validate_and_dedupe(data)
        _map_clauses_to_blocks(data.get("clauses", []), blocks)

        # 5) Per-clause citations for non-LOW risk clauses with implications (level-specific)
        max_clauses_citations = limits["max_clauses_citations"]
        clauses_with_citations = 0
        for c in data.get("clauses", []) or []:
            if clauses_with_citations >= max_clauses_citations:
                break
            if (c.get("risk_level") or "").upper() == "LOW":
                continue
            if not (c.get("implications") or "").strip():
                continue
            query = f"{c.get('text', '')[:150]} {c.get('implications', '')[:150]}".strip()
            if not query:
                continue
            per_docs = search_legal_knowledge(
                [query],
                source_filter=source_filter,
                top_k_per_query=limits["per_clause_top_k"],
                rrf_top_k=limits["per_clause_rrf_k"],
            )
            per_reranked = rerank_with_llm(query, per_docs, limits["per_clause_rerank_k"])
            c["citations"] = [cit.model_dump() for cit in _docs_to_citations(per_reranked)]
            clauses_with_citations += 1

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

        # Round risk_score
        risk_score = data.get("risk_score", 0)
        if isinstance(risk_score, (int, float)):
            rounding = getattr(settings, "COMPLIANCE_SCORE_ROUNDING", 2)
            risk_score = round(float(risk_score), rounding)
        else:
            risk_score = 0.0

        eth = data.get("ethiopian_law_compliance") or {}
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

        def to_clause(c: dict) -> ClauseAnalysis:
            return ClauseAnalysis(
                clause_id=c.get("clause_id", ""),
                text=c.get("text", ""),
                risk_level=c.get("risk_level", "LOW"),
                implications=c.get("implications", ""),
                block_id=c.get("block_id"),
                citations=_filter_citations(c.get("citations", [])),
            )

        return ComplianceAnalysisResponse(
            document_type=data.get("document_type", doc_type),
            overall_risk_level=data.get("overall_risk_level", "MEDIUM"),
            risk_score=risk_score,
            summary=data.get("summary", ""),
            clauses=[to_clause(c) for c in data.get("clauses", [])],
            issues_by_block_id={k: [to_issue(i) for i in v] for k, v in issues_by_block_id.items()},
            ethiopian_law_compliance=eth_compliance,
            recommendations=data.get("recommendations", []) or [],
            should_sign=data.get("should_sign"),
            critical_issues=[to_issue(i) for i in critical_issues],
            missing_clauses=data.get("missing_clauses", []) or [],
            citations=global_citations,
        )
