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
      "citations": []
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
