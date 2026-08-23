"""Prompt templates for locate / edit / verify / diff stages."""

from __future__ import annotations

from typing import Any


def output_language_instruction(document_language: str | None) -> str:
    if document_language == "am":
        return (
            "CRITICAL: All generated new_text MUST be in Amharic (አማርኛ). "
            "Do not translate to English."
        )
    if document_language == "en":
        return "CRITICAL: All generated new_text MUST be in English."
    return "Generate new_text in the same language as the document blocks."


def locate_system_prompt() -> str:
    return """You are a document block selector for a legal document editor.
Given a natural-language edit instruction and ranked candidate blocks, choose which block(s) to modify.

IMPORTANT: End users NEVER reference internal block_ids (e.g. "b2", "b31"). They describe
what they want in plain language — by clause content, section title, position, or intent.
Your job is to map that natural-language request to the correct block_id(s) from the candidates.

Respond with JSON only:
{
  "targets": [
    {"block_id": "...", "action": "replace|insert|remove|rewrite", "confidence": 0.0-1.0, "reason": "..."}
  ],
  "scope": "single|multiple|section|global",
  "confidence": 0.0-1.0
}

Rules:
- block_id MUST be copied exactly from the candidate list (internal IDs — never from the user instruction).
- Match blocks by their text content, type (heading vs paragraph), and position relative to the request.
- Use confidence >= 0.7 only when you are sure of the target.
- For section-wide edits (shorten/simplify/rewrite an entire part like Definitions, Recitals, Termination):
  set scope to "section" and target the section HEADING block_id (the pipeline expands to the full section body).
- For global replace (change X to Y everywhere), set scope to "global" and list all matching blocks.
- If no block matches the natural-language intent, return empty targets and confidence 0."""


def locate_user_prompt(instruction: str, candidates: list[dict[str, Any]]) -> str:
    lines = [f"Instruction: {instruction}", "", "Candidate blocks (ranked):"]
    for i, block in enumerate(candidates, 1):
        lines.append(
            f"{i}. block_id={block.get('block_id')} type={block.get('type')} "
            f"score={block.get('rerank_score', 'n/a')}\n   text: {(block.get('text') or '')[:400]}"
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
5. Only modify blocks required by the instruction — do not touch unrelated blocks.
6. Do NOT invent block_ids. Do NOT modify blocks outside the target list unless scope is global.
7. When shortening or simplifying legal text, preserve mandatory legal references (proclamation numbers, article citations, defined terms, and party obligations). Condense wording, do not remove substantive legal requirements.
8. Return ONLY valid JSON, no markdown fences."""


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
