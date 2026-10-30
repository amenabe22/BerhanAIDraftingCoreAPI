"""Prompt templates for locate / edit / verify / diff stages."""

from __future__ import annotations

from typing import Any


def output_language_instruction(document_language: str | None) -> str:
    if document_language == "am":
        return (
            "CRITICAL: All generated new_text MUST be in Amharic (አማርኛ). "
            "Do not translate to English."
        )
    if document_language == "om":
        return (
            "CRITICAL: All generated new_text MUST be in Afaan Oromoo. "
            "Do not translate to English or Amharic."
        )
    if document_language == "en":
        return "CRITICAL: All generated new_text MUST be in English."
    return "Generate new_text in the same language as the document blocks."


def locate_system_prompt() -> str:
    return """You are a document block selector for a legal document editor.
Given a natural-language edit instruction, a document skeleton (outline + length features),
and ranked candidate blocks, choose which block(s) to modify AND classify how the edit is scoped.

IMPORTANT: End users NEVER reference internal block_ids (e.g. "b2", "b31"). They describe
what they want in plain language — by clause content, section title, position, quality
("too short", "thin", "most sections"), or intent in ANY language. Map that to block_id(s).

Respond with JSON only:
{
  "spatial_scope": "pinpoint|section|multi_section|document_wide",
  "selection_mode": "semantic|explicit_section|structural|hybrid",
  "structural_intent": "short free-text description of how to choose blocks when not a single clause",
  "thinness": "below_median|bottom_quartile|all_body",
  "edit_intent": "expand|shorten|rewrite|replace|other",
  "targets": [
    {"block_id": "...", "action": "replace|insert|remove|rewrite|expand", "confidence": 0.0-1.0, "reason": "..."}
  ],
  "scope": "single|multiple|section|global",
  "confidence": 0.0-1.0
}

Dimensional rules (language-agnostic — reason from intent + skeleton features, not keywords):
- spatial_scope=pinpoint + selection_mode=semantic: one clause/paragraph; list that block in targets.
- spatial_scope=section + selection_mode=explicit_section: one named part; target the HEADING block_id
  from the Outline (pipeline expands to section body). Set scope=section.
- spatial_scope=multi_section + selection_mode=explicit_section: instruction names TWO OR MORE
  outline headings (by number and/or title, any language). List ALL matching HEADING block_ids
  from the Outline in targets. Do NOT use thinness. Set scope=multiple. Never claim a heading
  is missing if it appears in the Outline.
- spatial_scope=document_wide OR selection_mode=structural|hybrid (no specific section names):
  Prefer body blocks that match structural_intent using skeleton features
  (below_median / bottom_quartile / relative_thinness). Set thinness accordingly.
  List ALL matching body block_ids from candidates/features in targets (or leave targets empty
  if the pipeline should apply the structural filter — set confidence >= 0.6 and
  selection_mode=structural). Set scope=multiple.
- For find/replace of the same string everywhere: scope=global, spatial_scope=document_wide,
  selection_mode=semantic, list matching blocks.
- Use confidence >= 0.7 when sure; for broad structural or multi-section named edits,
  confidence 0.6–0.85 is fine when outline/features support the intent.
- block_id MUST be copied exactly from Outline / candidates / body_features — never invented.
- If the instruction is an edit but you cannot pick specific ids, still set spatial_scope /
  selection_mode / thinness / edit_intent so the pipeline can resolve structurally.
  Only use confidence 0 with empty plan fields when the message is not an edit at all."""


def locate_user_prompt(
    instruction: str,
    candidates: list[dict[str, Any]],
    *,
    skeleton: dict[str, Any] | None = None,
) -> str:
    lines = [f"Instruction: {instruction}", ""]
    if skeleton:
        stats = skeleton.get("stats") or {}
        lines.append("Document skeleton stats:")
        lines.append(
            f"  blocks={stats.get('block_count')} headings={stats.get('heading_count')} "
            f"body={stats.get('body_count')} median_body_chars={stats.get('median_body_chars')} "
            f"q1_body_chars={stats.get('q1_body_chars')}"
        )
        lines.append("")
        lines.append("Outline (headings):")
        for h in (skeleton.get("outline") or [])[:40]:
            lines.append(
                f"  - block_id={h.get('block_id')} level={h.get('level')} title={h.get('title')}"
            )
        lines.append("")
        lines.append("Body block features (for structural selection):")
        for row in (skeleton.get("body_features") or [])[:60]:
            lines.append(
                f"  - block_id={row.get('block_id')} type={row.get('type')} "
                f"chars={row.get('char_len')} words={row.get('word_count')} "
                f"thinness={row.get('relative_thinness')} below_median={row.get('below_median')} "
                f"q1={row.get('bottom_quartile')} parent={row.get('parent_heading_text')!r}\n"
                f"    preview: {row.get('preview')}"
            )
        lines.append("")

    lines.append("Candidate blocks (ranked for semantic match):")
    for i, block in enumerate(candidates, 1):
        lines.append(
            f"{i}. block_id={block.get('block_id')} type={block.get('type')} "
            f"score={block.get('rerank_score', 'n/a')} "
            f"chars={block.get('char_len', len((block.get('text') or '')))}\n"
            f"   text: {(block.get('text') or '')[:400]}"
        )
    return "\n".join(lines)


def edit_system_prompt(document_language: str | None) -> str:
    lang = output_language_instruction(document_language)
    return f"""You are a precise document editor. Generate edit operations as JSON.

{lang}

Respond with JSON only:
{{
  "operations": [
    {{
      "op_id": "unique8chars",
      "type": "replace|insert|remove|rewrite|summarize|expand|simplify|clarify|strengthen|weaken|change_heading",
      "block_id": "exact block_id from context",
      "payload": {{"new_text": "...", "position": "after|before", "level": 1}}
    }}
  ]
}}

CRITICAL RULES:
1. The user instruction is natural language — they do not know block_ids. Use block_ids only from the provided context.
2. block_id in operations MUST match an existing block_id from the provided blocks exactly.
2. For remove: payload must be {{}}.
3. For insert: payload must include new_text and position (before|after).
4. For replace/rewrite: payload must include new_text with the FULL new block text.
5. For expand/clarify/strengthen/"more detail": emit replace or expand with LONGER, complete legal prose in new_text — never empty operations and never reply with advice instead of ops.
6. When multiple target blocks are listed (document-wide / multi-section expand or shorten), emit ONE operation per target body block that needs change — especially when several named sections are in context. Do not collapse into a single op, skip named sections, or reply with advice.
7. Only modify blocks required by the instruction — do not touch unrelated blocks.
8. Do NOT invent block_ids. Do NOT modify blocks outside the target list unless scope is global.
9. When shortening or simplifying legal text, preserve mandatory legal references (proclamation numbers, article citations, defined terms, and party obligations). Condense wording, do not remove substantive legal requirements.
10. Return ONLY valid JSON, no markdown fences."""


def edit_user_prompt(
    instruction: str,
    target_blocks: list[dict[str, Any]],
    all_block_ids: list[str],
    *,
    revision_feedback: str | None = None,
) -> str:
    lines = [
        f"Instruction: {instruction}",
        "",
        "Target blocks to edit:",
    ]
    for block in target_blocks:
        lines.append(
            f"- block_id={block.get('block_id')} type={block.get('type')}\n"
            f"  current_text: {(block.get('text') or '')[:800]}"
        )
    lines.append("")
    lines.append(f"Valid block_ids in document: {', '.join(all_block_ids[:100])}")
    if revision_feedback:
        lines.extend(["", "REVISION REQUIRED:", revision_feedback])
    return "\n".join(lines)


def verify_system_prompt() -> str:
    return """You verify whether document edits satisfy the user's instruction.

Respond with JSON only:
{
  "passed": true|false,
  "issues": ["..."],
  "feedback": "actionable revision guidance if failed"
}

Check:
- Correct block(s) were modified
- Instruction intent is satisfied
- No unrelated blocks were incorrectly changed
- Language matches document language"""


def verify_user_prompt(
    instruction: str,
    before_blocks: list[dict[str, Any]],
    after_blocks: list[dict[str, Any]],
) -> str:
    def fmt(blocks: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"  {b.get('block_id')}: {(b.get('text') or '')[:400]}" for b in blocks
        )

    return (
        f"Instruction: {instruction}\n\n"
        f"BEFORE:\n{fmt(before_blocks)}\n\n"
        f"AFTER:\n{fmt(after_blocks)}"
    )


def diff_system_prompt() -> str:
    return """Summarize document changes in one concise paragraph for the user.
Be specific about what was added, removed, or modified. Do not invent changes."""


def diff_user_prompt(before_text: str, after_text: str) -> str:
    return f"Before:\n{before_text}\n\nAfter:\n{after_text}"
