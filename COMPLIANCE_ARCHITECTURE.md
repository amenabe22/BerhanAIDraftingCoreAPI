# Compliance Analysis — Architecture & Structure

This document describes the **current** compliance analysis architecture: where it lives, how it uses Qdrant and knowledge bases, and how the pieces fit together.

---

## 1. Overview

Compliance analysis checks a user’s document (contract, NDA, MOU, etc.) against the **Ethiopian legal framework**. It:

- Takes a **TipTap JSON document** (from Postgres blocks).
- Retrieves **legal context** from the **Legal KB** (Qdrant).
- Calls an **LLM** to produce structured analysis (clauses, issues, scores, citations).
- Optionally runs **per-clause citation retrieval** from the Legal KB.
- Returns a **ComplianceAnalysisResponse** and optionally persists it in the **compliance_analyses** table.

The **Doc KB** (per-document block vectors in Qdrant) is **not** used by compliance; it is used elsewhere (e.g. editor semantic search).

**Implementation in this codebase:** Document input is provided in the **request body** as either `document_text` (plain text) or `tiptap_json` (TipTap JSON). There is no Postgres/Block layer here; the sync endpoint `POST /drafting/compliance/analyze` accepts the document directly. Retrieval is **targeted**: one LLM call generates 2–4 document-specific search queries; vector search runs per query and results are merged with **Reciprocal Rank Fusion (RRF)**; an **LLM-based reranker** (not a separate rerank API) then narrows to the top chunks. Per-clause citation retrieval runs only for clauses with implications and non-LOW risk. Citations use the fixed format `[Source: {document_id} | Article {item_id} | {title}]` plus excerpt. Background/async analysis and persistence are out of scope in this API.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Core API (FastAPI)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  POST /drafting/compliance/analyze          (sync)                           │
│  POST /drafting/compliance/analyze-background (async → Celery)                │
│  GET  /documents/{doc_id}/compliance/analyses | /analyses/{id} | /latest      │
└─────────────────────────────────────────────────────────────────────────────┘
         │                              │
         │ load blocks from Postgres    │ enqueue task
         ▼                              ▼
┌──────────────────────┐      ┌────────────────────────────────────────────────┐
│  ComplianceAnalysis  │      │  Celery: analyze_compliance_task                │
│  Agent               │◄─────│  - Load doc from DB, build TipTap               │
│  (analysis_agent.py) │      │  - Run agent.analyze_document()                │
│                      │      │  - Save result via compliance_crud              │
│  - Legal KB (Qdrant) │      │  - WebSocket + task status updates             │
│  - LLM (OpenRouter)  │      └────────────────────────────────────────────────┘
│  - Embeddings        │
└──────────────────────┘
         │
         │ search_legal_knowledge(), get_available_source_files()
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Qdrant                                                                      │
│  - Legal KB collection (QDRANT_KNOWLEDGE_COLLECTION, e.g. BerhanAILegal…)    │
│  - Doc KB collection (QDRANT_DEFAULT_COLLECTION, e.g. doc_blocks) — NOT used │
│    by compliance                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **Document content**: always from **Postgres** (Block model), converted to TipTap by the API or the Celery task.
- **Legal context & citations**: from **Qdrant Legal KB** only, via `QdrantRAGService` in `knowledge_retrieval.py`.

---

## 3. Component Structure

### 3.1 API Layer

| File | Responsibility |
|------|----------------|
| `app/api/v1/endpoints/drafting/compliance.py` | **POST /analyze** (sync only in this codebase): accepts `document_text` or `tiptap_json` in the request body, calls `ComplianceAnalysisAgent().analyze_document()`, returns `ComplianceAnalysisResponse`. No background task or CRUD here. |

### 3.2 Compliance Agent

| File | Responsibility |
|------|----------------|
| `app/services/drafting/compliance/analysis_agent.py` | **ComplianceAnalysisAgent**: full pipeline (extract blocks, detect doc type, retrieve legal knowledge, build prompt, call LLM, parse, validate duplicates, per-clause citations, map to block IDs, build response). |
| `app/services/drafting/compliance/__init__.py` | Package init. |

Agent dependencies (this codebase):

- **knowledge_retrieval.py** — `generate_targeted_queries()`, `search_legal_knowledge()` (multi-query + optional source filter + RRF), `rerank_with_llm()`, `get_available_source_files()`. Uses `app.retrieval.get_legal_kb_vector_store()` and Cohere embeddings; reranking is done with a regular LLM (OpenRouter), not a dedicated rerank API.
- **Config** — `app.config.settings` (COMPLIANCE_ANALYSIS_MODEL, COMPLIANCE_RERANKER_MODEL, KNOWLEDGE_RERANK_*, etc.).

### 3.3 Knowledge & Vector Stores

| Component | Role in compliance |
|-----------|----------------------|
| **Legal KB** | Single Qdrant collection (`settings.qdrant_knowledge_collection`). Contains Ethiopian law chunks (e.g. by `source_file`). Used for: (1) global legal context in the analysis prompt, (2) per-clause implication citations. Embeddings: Cohere `embed-multilingual-v3.0`; reranker configurable (e.g. `compliance_reranker_provider`). |
| **Doc KB** | Single shared Qdrant collection (`settings.qdrant_default_collection`, e.g. `doc_blocks`). Stores per-document block vectors for semantic search in the editor. **Not used by the compliance agent.** |

### 3.4 Persistence & CRUD

| File | Responsibility |
|------|----------------|
| `app/models/drafting/compliance.py` | Pydantic models (ComplianceAnalysisRequest, ComplianceAnalysisResponse, ClauseAnalysis, LegalIssue, EthiopianLawCompliance, etc.) and SQLAlchemy model **ComplianceAnalysis** (table `compliance_analyses`). |
| `app/crud/compliance.py` | **ComplianceCRUD**: create_compliance_analysis, get_compliance_analyses_by_document, get_compliance_analysis_by_id, get_latest_compliance_analysis, and failed-analysis detection (so bad runs are not stored). |

### 3.5 Background Task

| File | Responsibility |
|------|----------------|
| `app/tasks/ai_tasks.py` | **analyze_compliance_task** (Celery): load document and blocks from DB, build TipTap, call `ComplianceAnalysisAgent().analyze_document()`, then save via `compliance_crud.create_compliance_analysis()`, update task status, and send WebSocket updates (Redis). |

---

## 4. End-to-End Data Flow

### 4.1 Sync flow (POST /drafting/compliance/analyze)

1. Resolve document by `doc_id` (drafting UUID or DB document ID), company-aware.
2. Resolve billing user; check and consume **DOC_COMPLIANCE** credits.
3. Load **Block** rows for the document from Postgres, ordered by index.
4. Convert blocks to TipTap JSON (`FormatConverter.html_blocks_to_tiptap`).
5. Get optional `document_type` from the Document record.
6. Call **ComplianceAnalysisAgent.analyze_document(** doc_json, language, document_type **)**.
7. Return **ComplianceAnalysisResponse** (and optionally save via documents API).

### 4.2 Background flow (POST /drafting/compliance/analyze-background)

1. Same document and credit checks as above.
2. Enqueue **analyze_compliance_task** with doc_uuid, user_id, language, document_type, check_level (or temperature/seed, depending on branch).
3. Create a **Task** record (task_crud) for status polling.
4. Return 202 with task_id.

**Celery worker:**

1. Load document and blocks from DB; build TipTap.
2. Call **ComplianceAnalysisAgent.analyze_document()** (same as sync).
3. Save result with **compliance_crud.create_compliance_analysis()** (document_id, user_id, analysis_response, language, document_type).
4. Update Task status and push WebSocket update via Redis.

### 4.3 Agent pipeline (analyze_document)

1. **Extract** blocks and full text from TipTap (`extract_blocks_from_tiptap`, `_extract_full_document_text`).
2. **Document type**: use request value or detect from content (`_detect_document_type`).
3. **Source keywords**: LLM chooses which law “books” to cite; list of books comes from **Legal KB** (`get_available_source_files()`).
4. **Retrieve legal knowledge**: `_retrieve_legal_knowledge(full_text, document_type, source_file_keywords)` → `QdrantRAGService.search_legal_knowledge()` (optional HYDE, rerank). One search per keyword, then merge. Result: formatted legal context string + raw chunks for deterministic citations.
5. **Build prompt** with document text, blocks, legal context, scoring rules, output schema (`_build_analysis_prompt`).
6. **Call LLM** (`_call_llm_for_analysis`); parse JSON (`_parse_analysis_response`).
7. **Validate** duplicate flags (`_validate_and_filter_duplicates`).
8. **Per-clause citations**: for clauses with implications and non-LOW risk, batch-embed queries and call `search_legal_knowledge(..., precomputed_embedding=...)` again; attach citation labels/dicts to clauses (`_retrieve_implication_citations`).
9. **Map** clauses to block IDs (`_map_clauses_to_blocks`).
10. **Build** final **ComplianceAnalysisResponse** (`_build_response`).

All Qdrant access in this pipeline is to the **Legal KB** collection only.

---

## 5. Qdrant Usage Summary

| Collection (config) | Used by compliance? | Purpose |
|--------------------|----------------------|--------|
| **qdrant_knowledge_collection** (`QDRANT_KNOWLEDGE_COLLECTION`) | **Yes** | Legal KB: Ethiopian law chunks. Searched for prompt context and per-clause citations. Same Cohere embeddings and optional HYDE/reranker as in `knowledge_retrieval.py`. |
| **qdrant_default_collection** (`QDRANT_DEFAULT_COLLECTION`) | **No** | Doc KB: document block vectors. Used by editor/ingestion/semantic features, not by compliance. |

Compliance does **not** read the document from Qdrant; it always receives TipTap built from Postgres blocks.

---

## 6. Request / Response Models

- **ComplianceAnalysisRequest** (this codebase): `document_text` or `tiptap_json` (one required), `language` (default `"en"`), optional `document_type`, `check_level`.
- **ComplianceAnalysisResponse**: `document_type`, `overall_risk_level`, `risk_score`, `summary`, `clauses` (ClauseAnalysis), `issues_by_block_id`, `ethiopian_law_compliance`, `recommendations`, `should_sign`, `critical_issues`, `missing_clauses`, `citations` (LegalCitation with document_id, item_id, title, excerpt), etc.
- **ComplianceAnalysis** (DB): `document_id`, `user_id`, `analysis_data` (JSON), `language`, `document_type`, `overall_risk_level`, `risk_score`, `summary`, `created_at`.

---

## 7. Configuration (Compliance-Relevant)

From `app/config.py` and env:

- **Qdrant**: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_KNOWLEDGE_COLLECTION`, `QDRANT_DEFAULT_COLLECTION`.
- **Compliance LLM**: `COMPLIANCE_ANALYSIS_MODEL` (else `gemini_model`), `COMPLIANCE_ANALYSIS_TIMEOUT`, `COMPLIANCE_ANALYSIS_TEMPERATURE`, `COMPLIANCE_ANALYSIS_SEED`, `COMPLIANCE_ANALYSIS_MAX_TOKENS`, `COMPLIANCE_SCORE_ROUNDING`.
- **RAG**: `COMPLIANCE_USE_HYDE`, `COMPLIANCE_RERANKER_PROVIDER`, `COMPLIANCE_IMPLICATION_INITIAL_LIMIT`, `COMPLIANCE_MAX_CLAUSES_FOR_CITATIONS`.
- **Knowledge**: `knowledge_rerank_initial_limit`, `knowledge_rerank_top`, `rag_relevance_threshold`, and embedding/reranker settings used by `QdrantRAGService`.

---

## 8. File Layout (Compliance-Related)

```
app/
├── api/v1/endpoints/
│   ├── drafting/compliance.py    # POST /analyze, /analyze-background
│   └── documents.py              # Compliance CRUD under /documents/{id}/compliance/...
├── config.py                     # Settings (qdrant_*, compliance_*, knowledge_*)
├── crud/compliance.py            # ComplianceCRUD
├── models/drafting/compliance.py # Request/response Pydantic + ComplianceAnalysis (DB)
├── services/drafting/
│   ├── knowledge_retrieval.py    # QdrantRAGService (Legal KB)
│   ├── compliance/
│   │   ├── __init__.py
│   │   └── analysis_agent.py     # ComplianceAnalysisAgent
│   └── advisor/citation_service.py
├── tasks/ai_tasks.py             # analyze_compliance_task (Celery)
└── vector_store/
    ├── qdrant_client.py          # Doc blocks collection / async client (not used by compliance agent)
    └── embeddings.py             # Cohere batch embeddings (used for implication citations)
```

---

## 9. Summary

- **Compliance** = Postgres (document blocks) + **Legal KB (Qdrant)** + LLM + optional per-clause Legal KB retrieval.
- **Doc KB** in Qdrant is for editor/semantic features; compliance does not use it.
- One place implements the analysis logic: **ComplianceAnalysisAgent** in `app/services/drafting/compliance/analysis_agent.py`. The API and Celery task only prepare input (TipTap + options) and persist or stream the result.
