# Compliance Analyze API

Run compliance analysis against Ethiopian law using a document from the doc collection. The document is loaded by `doc_id` so that block structure (`block_id`, `type`) is available for context and citations.

**Endpoint:** `POST /drafting/compliance/analyze`

**Content-Type:** `application/json`

---

## Request (input shape)

| Field         | Type     | Required | Default   | Description |
|---------------|----------|----------|-----------|-------------|
| `doc_id`      | string   | Yes      | —         | Document ID in the doc collection (e.g. Qdrant `doc_blocks`). The API loads all blocks for this document and uses their `block_id`, `text`, and `type` for analysis. |
| `language`    | string   | No       | `"en"`    | Response language. Use `"en"` for English, `"am"` for Amharic, `"om"` for Oromo (or other codes your deployment supports). |
| `check_level` | string   | No       | `"quick"` | Analysis depth. One of: `"quick"` (fastest, less context), `"standard"`, `"deep"` (most thorough, more document and legal context, more per-clause citations). |

**Example request body:**

```json
{
  "doc_id": "8749c6dc-4bb3-4f5c-b593-ae54d0da5437",
  "language": "en",
  "check_level": "standard"
}
```

---

## Response (output shape)

On success the response is a JSON object with the following structure.

### Top-level fields

| Field                     | Type     | Description |
|---------------------------|----------|-------------|
| `document_type`           | string   | Detected or inferred document type (e.g. `"Contract"`, `"NDA"`, `"Employment Agreement"`, `"Lease"`). |
| `overall_risk_level`      | string   | One of: `"LOW"`, `"MEDIUM"`, `"HIGH"`, `"CRITICAL"`. |
| `risk_score`              | number   | Numeric risk score, typically 0–100. |
| `summary`                 | string   | Executive summary of the compliance analysis. |
| `clauses`                 | array    | List of clause analyses (see [Clause object](#clause-object) below). |
| `issues_by_block_id`      | object   | Map of `block_id` → list of [Issue](#issue-object) objects. Issues tied to a specific block are listed under that block’s id. |
| `ethiopian_law_compliance`| object   | Summary of compliance with Ethiopian law (see [Ethiopian law compliance](#ethiopian-law-compliance-object) below). |
| `recommendations`         | array    | List of recommendation strings. |
| `should_sign`             | boolean \| null | Recommendation to sign (`true`), not sign (`false`), or `null` if not determined. |
| `critical_issues`         | array    | List of [Issue](#issue-object) objects with severity `HIGH` or `CRITICAL`. |
| `missing_clauses`         | array    | List of strings describing clauses that may be missing for this document type. |
| `citations`               | array    | Global legal citations used in the analysis (see [Citation object](#citation-object) below). |

### Clause object

Each element of `clauses` has:

| Field          | Type     | Description |
|----------------|----------|-------------|
| `clause_id`    | string   | Unique id for the clause (e.g. `"1"`, `"2.1"`, `"4.1"`). |
| `text`         | string   | Clause text or summary. |
| `risk_level`   | string   | One of: `"LOW"`, `"MEDIUM"`, `"HIGH"`, `"CRITICAL"`. |
| `implications` | string   | Legal implications of the clause. |
| `block_id`     | string \| null | Block id from the document, if the clause was mapped to a block; otherwise `null`. |
| `citations`    | array    | Per-clause legal citations (list of [Citation](#citation-object) objects). May be empty. |
| `ethiopian_law_implications` | array | Specific Ethiopian law implications (populated for MEDIUM+ risk). |
| `recommendations` | array | Human-facing actionable recommendations (populated for MEDIUM+ risk). |
| `editor_fix`   | object \| null | Structured edit spec for the semantic editor. Populated for **MEDIUM+** clauses (warnings and above); `null` for **LOW** only. See [Editor fix object](#editor-fix-object). |

### Editor fix object

Present on `clauses[].editor_fix` when `risk_level` is `MEDIUM`, `HIGH`, or `CRITICAL`. Omitted (`null`) for `LOW` clauses. Separate from human-readable `recommendations` — intended for automated document edits.

| Field | Type | Description |
|-------|------|-------------|
| `action` | string | Edit action. Currently `"replace"`. |
| `block_id` | string \| null | Target document block to edit. |
| `clause_reference` | string | Human-readable clause label (e.g. `"4. Compensation"`). |
| `current_text` | string | Current clause or block text (enriched from document blocks when `block_id` is set). |
| `problem_summary` | string | Brief summary of the compliance problem. |
| `offending_phrases` | array | Phrases in the current text that cause the issue. |
| `legal_requirement` | string | What Ethiopian law requires in this area. |
| `rewrite_directive` | string | Imperative instruction for rewriting (not a restatement of law). |
| `remove_phrases` | array | Phrases to remove from the clause. |
| `add_elements` | array | Elements that must appear in the rewritten clause. |
| `suggested_text` | string | Concrete draft replacement text; uses `[BRACKETED_PLACEHOLDERS]` when values are unknown. |
| `placeholder_policy` | string | Always `"use_bracketed_placeholders_when_values_unknown"`. |
| `legal_basis` | array | List of [Legal basis](#legal-basis-object) objects. |
| `document_language` | string | Language code for the fix (matches request `language`). |
| `severity` | string | `"medium_risk"` (MEDIUM), `"high_risk"` (HIGH), or `"critical_risk"` (CRITICAL). |
| `confidence` | number | Confidence score from 0.0 to 1.0. |

### Legal basis object

Used in `editor_fix.legal_basis`:

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Legal source name (e.g. `"ethiopian-labor-proclamation"`, `"Civil Code"`). |
| `article` | string | Article or section reference. |
| `rationale` | string | Why this law requires the fix. |

### Issue object

Used in `issues_by_block_id` and `critical_issues`:

| Field         | Type     | Description |
|---------------|----------|-------------|
| `issue_id`    | string   | Unique id for the issue. |
| `description` | string   | Description of the issue. |
| `severity`    | string   | One of: `"LOW"`, `"MEDIUM"`, `"HIGH"`, `"CRITICAL"`. |
| `block_id`    | string \| null | Block id if the issue is tied to a specific clause; otherwise `null`. |
| `citations`   | array    | List of [Citation](#citation-object) objects. |

### Citation object

Used in `citations`, and in each clause’s and issue’s `citations`:

| Field          | Type   | Description |
|----------------|--------|-------------|
| `document_id`  | string | Source name / legal document (e.g. `"amharic-family-code"`, `"Civil Code"`). |
| `item_id`      | string | Article or provision id (e.g. `"85"`, `"197"`). |
| `title`        | string | Section or article title. |
| `excerpt`      | string | Relevant text excerpt from the cited provision. |

### Ethiopian law compliance object

| Field              | Type   | Description |
|--------------------|--------|-------------|
| `summary`          | string | Brief compliance summary. |
| `applicable_laws`  | array  | List of applicable law names or references. |
| `concerns`         | array  | List of concern strings. |

---

## Example response (minimal)

```json
{
  "document_type": "Contract",
  "overall_risk_level": "LOW",
  "risk_score": 25.5,
  "summary": "The document presents low compliance risk. Key terms are consistent with common practice.",
  "clauses": [
    {
      "clause_id": "1",
      "text": "This agreement is made between Party A and Party B.",
      "risk_level": "LOW",
      "implications": "Standard party identification.",
      "block_id": "abc-123",
      "citations": [],
      "ethiopian_law_implications": [],
      "recommendations": [],
      "editor_fix": null
    },
    {
      "clause_id": "4",
      "text": "The Employee's compensation may include benefits as defined internally by the Company.",
      "risk_level": "HIGH",
      "implications": "Compensation terms are vague and may be unenforceable.",
      "block_id": "def-456",
      "citations": [],
      "ethiopian_law_implications": ["Labor law requires clear wage terms"],
      "recommendations": ["Specify salary amount and payment schedule"],
      "editor_fix": {
        "action": "replace",
        "block_id": "def-456",
        "clause_reference": "4. Compensation",
        "current_text": "The Employee's compensation may include benefits as defined internally by the Company.",
        "problem_summary": "Compensation terms are vague and may be unenforceable under Ethiopian employment law.",
        "offending_phrases": ["as defined internally by the Company"],
        "legal_requirement": "Employment terms must clearly specify wages, benefits, and material compensation components.",
        "rewrite_directive": "Rewrite the clause to state salary amount, payment schedule, and benefits in explicit contractual language.",
        "remove_phrases": ["as defined internally by the Company"],
        "add_elements": ["base salary or salary band", "payment frequency", "other benefits in writing"],
        "suggested_text": "The Employee shall receive a monthly base salary of [AMOUNT] ETB, payable [FREQUENCY]. Other benefits shall be specified in an annex to this Contract.",
        "placeholder_policy": "use_bracketed_placeholders_when_values_unknown",
        "legal_basis": [
          {
            "source": "ethiopian-labor-proclamation",
            "article": "Section on wages and working conditions",
            "rationale": "Wages and benefits must be stated clearly in the contract."
          }
        ],
        "document_language": "en",
        "severity": "high_risk",
        "confidence": 0.85
      }
    }
  ],
  "issues_by_block_id": {},
  "ethiopian_law_compliance": {
    "summary": "",
    "applicable_laws": [],
    "concerns": []
  },
  "recommendations": [],
  "should_sign": true,
  "critical_issues": [],
  "missing_clauses": [],
  "citations": []
}
```

---

## Error responses

| Status | Meaning |
|--------|--------|
| **400** | Bad request (e.g. invalid parameters). Body may include a `detail` message. |
| **404** | Document not found or has no content for the given `doc_id`. |
| **422** | Validation error (e.g. missing `doc_id` or invalid `check_level`). Response body describes the validation failure. |
| **503** | Compliance analysis failed (e.g. upstream service error). |

Error responses typically include a JSON body with a `detail` field (string or array of validation errors).
