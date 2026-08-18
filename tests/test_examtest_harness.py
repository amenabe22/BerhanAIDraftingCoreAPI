"""Golden smoke: exam harness must not dump the whole file or printed keys."""

from examtest import build_question_messages
from app.services.legal.exam_parse import QUESTION_PROMPT

MINI_EXAM = """
1. Under Ethiopian law of contracts, validity demands:
A. Object, form, consent and capacity
B. Form, capacity and consent
C. Consent, capacity and object
Answer: C
2. Which of the following is WRONG about variation of contracts?
A. Judges must always vary
B. Parties may exclude a period of grace
C. Variation is possible whenever performance is impossible
Answer: C
3. Identify the INCORRECT statement about damages.
A. Fault is always required
B. Force majeure may exempt the debtor
C. Damages equal damage suffered
Answer: A
"""

EXCEPT_ITEMS = """
1. All are true about shares EXCEPT
A. Dividend shares do not represent capital
B. Transfer may be limited
C. Currently companies may issue bearer shares
Answer: C
2. One is NOT TRUE about indivisible obligations
A. Same as joint and several
B. Partial performance discharges
C. May arise from nature, contract or law
Answer: A
"""


def test_printed_keys_never_reach_the_model():
    jobs = build_question_messages(MINI_EXAM)
    blob = "\n".join(j["message"] for j in jobs)
    assert "Answer: C" not in blob
    assert "Answer: A" not in blob
    assert "Answer:" not in blob


def test_one_message_per_question_not_full_dump():
    jobs = build_question_messages(MINI_EXAM)
    assert len(jobs) == 3
    for job in jobs:
        assert job["message"].startswith(QUESTION_PROMPT.strip()[:40]) or QUESTION_PROMPT[:20] in job["message"]
        # A single question message must not contain both Q1 and Q2 stems
        assert not (
            "validity demands" in job["message"] and "WRONG about variation" in job["message"]
        )


def test_except_stems_preserved_for_overlay():
    jobs = build_question_messages(EXCEPT_ITEMS)
    assert len(jobs) == 2
    assert any("EXCEPT" in j["message"] for j in jobs)
    assert any("NOT TRUE" in j["message"] for j in jobs)
    assert "do not invent a letter" in QUESTION_PROMPT.lower()


def test_full_dump_is_not_the_default_path():
    """A future dump-the-whole-file default would put both stems in one message."""
    jobs = build_question_messages(MINI_EXAM)
    assert all(j["n"] == i for i, j in enumerate(jobs, start=1))
    assert all("Answer EVERY question" not in j["message"] for j in jobs)


def test_assosa_early_questions_split_without_full_dump():
    """Assosa Q1–Q13 style: unnumbered stems must become separate messages."""
    from pathlib import Path
    from examtest import extract_docx_text

    path = Path("exms/Assosa University- Law Model Exit Exam - 2015 E.C.docx")
    if not path.exists():
        return
    jobs = build_question_messages(extract_docx_text(path))
    assert len(jobs) >= 70
    first = "\n".join(j["message"] for j in jobs[:13])
    assert "Answer:" not in first
    # Must not be a single mega-prompt
    assert jobs[0]["n"] == 1
    assert jobs[1]["n"] == 2
