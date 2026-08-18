"""Tests for exam question splitting and printed-key stripping."""

from app.services.legal.exam_parse import (
    QUESTION_PROMPT,
    classify_mismatch,
    split_exam_questions,
    strip_printed_keys,
)


class TestStripPrintedKeys:
    def test_strips_answer_colon_line(self):
        text = "Which statement is true?\nA. One\nB. Two\nAnswer: B\n"
        out = strip_printed_keys(text)
        assert "Answer: B" not in out
        assert "Which statement is true?" in out
        assert "B. Two" in out

    def test_strips_option_line(self):
        text = "Q text\nA. foo\nB. bar\nOption C\n"
        out = strip_printed_keys(text)
        assert "Option C" not in out
        assert "A. foo" in out

    def test_strips_question_id_answer_suffix(self):
        text = "2.What is custom? A. X B. Y Question ID:2227 and answer is B"
        out = strip_printed_keys(text)
        assert "answer is B" not in out.lower()
        assert "What is custom?" in out

    def test_keeps_choice_text_all_of_the_above(self):
        text = "Which is correct?\nA. One\nB. All of the above\n"
        out = strip_printed_keys(text)
        assert "All of the above" in out


class TestSplitExamQuestions:
    def test_splits_numbered_mcq(self):
        text = (
            "1. What is a contract?\n"
            "A. An agreement\n"
            "B. A gift\n"
            "2. Which is WRONG?\n"
            "A. Courts may vary\n"
            "B. Parties may agree\n"
        )
        qs = split_exam_questions(text)
        assert len(qs) == 2
        assert qs[0]["n"] == 1
        assert "What is a contract?" in qs[0]["stem"]
        assert "A. An agreement" in qs[0]["text"]
        assert "Which is WRONG?" in qs[1]["stem"]

    def test_split_does_not_include_stripped_keys(self):
        text = "1. Stem?\nA. One\nB. Two\nAnswer: A\n2. Next?\nA. X\nAnswer: B\n"
        qs = split_exam_questions(strip_printed_keys(text))
        assert len(qs) == 2
        joined = "\n".join(q["text"] for q in qs)
        assert "Answer:" not in joined

    def test_splits_unnumbered_stems(self):
        paras = [
            "Under Ethiopian law of contracts, the validity of any contract demands:",
            "A. Object, form, consent and capacity",
            "B. Form, capacity and consent",
            "C. Consent, capacity and object",
            "Which of the following is WRONG about variation of contracts?",
            "A. Judges must vary",
            "B. Parties may exclude grace",
            "C. Variation is possible whenever performance is impossible",
        ]
        qs = split_exam_questions("\n".join(paras))
        assert len(qs) >= 2
        assert "validity of any contract" in qs[0]["text"]
        assert any("WRONG about variation" in q["text"] for q in qs)


class TestQuestionPrompt:
    def test_forbids_inventing_letter_on_kb_miss(self):
        assert "do not invent a letter" in QUESTION_PROMPT.lower()
        assert "EXCEPT" in QUESTION_PROMPT or "NOT TRUE" in QUESTION_PROMPT

    def test_search_this_question_only(self):
        assert "this question" in QUESTION_PROMPT.lower()


class TestClassifyMismatch:
    def test_key_in_prompt_exam(self):
        cause = classify_mismatch(
            exam_id="ddu-practice",
            q=1,
            gold="D",
            berhan="D. something",
            stem="Which is true?",
            kb_miss=False,
            status="wrong",
        )
        assert cause == "key_in_prompt"

    def test_assosa_except_stem_is_option_mismatch(self):
        cause = classify_mismatch(
            exam_id="assosa-2015",
            q=2,
            gold="C",
            berhan="In principle, courts are not allowed",
            stem="Which of the following is WRONG about variation of contracts?",
            kb_miss=False,
            status="wrong",
        )
        assert cause == "option_mismatch"

    def test_kb_miss_late_assosa_dump_is_batch_retrieval(self):
        cause = classify_mismatch(
            exam_id="assosa-2015",
            q=40,
            gold="A",
            berhan="All",
            stem="Identify the correct statement.",
            kb_miss=True,
            status="wrong",
            berhan_body="## Q1\nA\n\n## Q40\nAll\nThe knowledge base does not contain a specific provision on this point.",
        )
        assert cause == "batch_retrieval"

    def test_per_question_kb_miss_is_not_batch_retrieval(self):
        cause = classify_mismatch(
            exam_id="assosa-2015",
            q=40,
            gold="A",
            berhan="The knowledge base does not contain a specific provision on this point.",
            stem="Identify the correct statement.",
            kb_miss=True,
            status="wrong",
        )
        assert cause == "kb_gap_real"

    def test_samara_after_q7_is_alignment_noise(self):
        cause = classify_mismatch(
            exam_id="samara",
            q=20,
            gold="A",
            berhan="C. It is a bundle of right.",
            stem="Property is:",
            kb_miss=False,
            status="wrong",
        )
        assert cause == "alignment_noise"

    def test_company_civil_code_is_wrong_instrument(self):
        cause = classify_mismatch(
            exam_id="assosa-2015",
            q=11,
            gold="E",
            berhan="Agency relationship between partners.",
            stem="One is necessarily a characteristic of companies generally EXCEPT.",
            kb_miss=False,
            status="wrong",
            berhan_body="Under Article 10 of the Civil Code, a company has personality.",
        )
        assert cause == "wrong_instrument"

    def test_theory_is_kb_gap_real(self):
        cause = classify_mismatch(
            exam_id="llb-debre-markos",
            q=62,
            gold="B",
            berhan="A. Social contract is a wholesale of our rights",
            stem="Hart's concept of primary and secondary rules refers to",
            kb_miss=False,
            status="wrong",
        )
        assert cause == "kb_gap_real"
