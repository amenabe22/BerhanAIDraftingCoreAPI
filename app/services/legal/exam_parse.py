"""Split law-exam text into questions and strip printed answer keys."""

from __future__ import annotations

import re
from typing import Literal

Cause = Literal[
    "batch_retrieval",
    "wrong_instrument",
    "neighbor_article",
    "option_mismatch",
    "kb_gap_real",
    "prompt_conflict",
    "key_in_prompt",
    "alignment_noise",
]

ANSWER_LINE_RE = re.compile(
    r"(?im)^\s*(?:[-–—]\s*)?answer(?:\s+is)?\s*"
    r"(?:[:\-]\s*\(?\s*[A-Fa-f](?:\s*/\s*[A-Fa-f])?\s*\)?|\(\s*[A-Fa-f]\s*\)|\s+[A-Fa-f])\s*[.]?\s*$"
)
OPTION_KEY_RE = re.compile(r"(?im)^\s*option\s+[A-E](?:\s*/\s*[A-E])?\s*$")
QID_TAIL_RE = re.compile(
    r"(?i)\s*question\s*id\s*:\s*\d+\s+and\s+answer\s+is\s+[A-Ea-e]\s*$"
)
NUMBERED_STEM_RE = re.compile(r"(?m)^(?P<n>\d{1,3})[.)]\s+(?P<rest>\S.*)$")
EXCEPT_STEM_RE = re.compile(
    r"(?i)\b(except|wrong|not true|not correct|incorrect|not necessary|"
    r"false statement|odd one|which one is not)\b"
)
THEORY_RE = re.compile(
    r"(?i)\b(hart|austin|kelsen|rawls|dworkin|positivi|natural law|"
    r"social contract|jurisprudence|command theory|pure theory|"
    r"primary and secondary rules|original position)\b"
)
COMPANY_RE = re.compile(
    r"(?i)\b(compan(?:y|ies)|shareholder|director|share capital|plc|"
    r"share company|partnership|trader)\b"
)
CIVIL_CITE_RE = re.compile(r"(?i)\bcivil code\b")
COMMERCIAL_CITE_RE = re.compile(r"(?i)\bcommercial code\b")

PRINTED_KEY_EXAMS = {
    "ddu-practice",
    "debre-berhan-2023",
    "hu-col-mock-2026",
    "uog-sol",
}

QUESTION_PROMPT = """You are answering ONE multiple-choice (or short) law exam question.

Instructions:
1. Search the legal knowledge base for THIS question only. Do not answer other questions.
2. Read the stem carefully. If it says EXCEPT, WRONG, NOT TRUE, or NOT CORRECT, pick the option that does NOT match the law.
3. For multiple choice: if the tool returned articles that bear on the options, give the letter (A/B/C/D/E) first, then a short justification with pinpoint cites from those results. Do not refuse a letter merely because the articles are incomplete.
4. Use this sentence only when the tool results are empty or clearly about a different topic: 'The knowledge base does not contain a specific provision on this point.' In that case do not invent a letter and do not invent article numbers.
5. Output the answer only — no restating the full question.

QUESTION:
"""


def strip_printed_keys(text: str) -> str:
    """Remove standalone Answer:/Option keys and trailing Question-ID keys."""
    lines = []
    for line in (text or "").splitlines():
        if ANSWER_LINE_RE.match(line.strip()) or OPTION_KEY_RE.match(line.strip()):
            continue
        cleaned = QID_TAIL_RE.sub("", line).rstrip()
        lines.append(cleaned)
    return "\n".join(lines).strip()


ALWAYS_STEM_RE = re.compile(
    r"(?i)^(which|among|identify|suppose that|suppose |of the following|"
    r"all are |all statements|one is |one of the|under the ethiopian|"
    r"under ethiopian|regarding |from the following|what is|what are|"
    r"a person who|according to |currently |the following|"
    r"business organization|talking about|pinpoint )"
)


CHOICE_LINE_RE = re.compile(r"^[A-E][.)]\s+\S")


def split_unnumbered_mcq(text: str) -> list[dict]:
    """Split stem+choice exams that are not numbered (e.g. Assosa)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return []
    blocks: list[list[str]] = []
    cur: list[str] = []
    for line in lines:
        has_choices = any(CHOICE_LINE_RE.match(x) for x in cur)
        new_stem = False
        if cur and ALWAYS_STEM_RE.match(line) and len(cur) >= 4:
            new_stem = True
        elif (
            cur
            and has_choices
            and not CHOICE_LINE_RE.match(line)
            and len(line) > 40
            and not line.lower().startswith("justification")
        ):
            new_stem = True
        if new_stem:
            blocks.append(cur)
            cur = [line]
            continue
        cur.append(line)
    if cur:
        blocks.append(cur)
    out = []
    for i, block in enumerate(blocks, start=1):
        out.append({"n": i, "stem": block[0], "text": "\n".join(block)})
    return out


def split_exam_questions(text: str) -> list[dict]:
    """Split exam plaintext into numbered questions.

    Returns a list of ``{n, stem, text}`` in order. ``text`` is the full
    question block (stem + options) with printed keys already expected stripped.
    """
    text = (text or "").strip()
    if not text:
        return []

    matches = list(NUMBERED_STEM_RE.finditer(text))
    if len(matches) >= 2:
        questions = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[start:end].strip()
            stem = m.group("rest").split("\n", 1)[0].strip()
            questions.append({"n": int(m.group("n")), "stem": stem, "text": block})
        return questions

    return split_unnumbered_mcq(text)


def classify_mismatch(
    *,
    exam_id: str,
    q: int,
    gold: str,
    berhan: str,
    stem: str,
    kb_miss: bool,
    status: str,
    berhan_body: str = "",
) -> Cause:
    """Assign one primary failure cause for a scored mismatch or KB-miss wrong."""
    if exam_id in PRINTED_KEY_EXAMS:
        return "key_in_prompt"
    if exam_id == "samara" and q > 7:
        return "alignment_noise"
    body = f"{berhan}\n{berhan_body}\n{stem}"
    if THEORY_RE.search(body) or THEORY_RE.search(stem):
        return "kb_gap_real"
    if COMPANY_RE.search(stem) and CIVIL_CITE_RE.search(berhan_body) and not COMMERCIAL_CITE_RE.search(
        berhan_body
    ):
        return "wrong_instrument"
    if kb_miss and re.search(r"(?m)^##\s*Q\d+", berhan_body or ""):
        return "batch_retrieval"
    if kb_miss and re.match(r"^\s*[A-Ea-e]\b", berhan or ""):
        return "prompt_conflict"
    if kb_miss:
        return "kb_gap_real"
    if EXCEPT_STEM_RE.search(stem):
        return "option_mismatch"
    if COMPANY_RE.search(stem):
        return "wrong_instrument"
    return "option_mismatch"
