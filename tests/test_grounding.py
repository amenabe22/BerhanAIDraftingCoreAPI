"""Unit tests for app/services/legal/grounding.py."""

import pytest
from langchain_core.messages import ToolMessage

from app.services.legal.grounding import (
    GroundingResult,
    _is_amharic_heavy,
    extract_retrieved_item_ids,
    is_legal_conclusion,
    repair_message,
    safe_fallback_answer,
    validate_answer,
)


def _tm(content: str, tool_call_id: str = "tc1") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tool_call_id)


CITATION_BLOCK = (
    "[Source: english-civil-code-1960 | Article 1675 | Contract defined]\n"
    "A contract is an agreement whereby two or more persons create obligations."
)


# ---------------------------------------------------------------------------
# extract_retrieved_item_ids
# ---------------------------------------------------------------------------


class TestExtractRetrievedItemIds:
    def test_single_source_block(self):
        ids = extract_retrieved_item_ids([_tm(CITATION_BLOCK)])
        assert "1675" in ids

    def test_multiple_blocks(self):
        content = (
            "[Source: doc-a | Article 1675 | Title A]\nContent A.\n\n"
            "[Source: doc-b | Article 1243 | Title B]\nContent B."
        )
        ids = extract_retrieved_item_ids([_tm(content)])
        assert "1675" in ids
        assert "1243" in ids

    def test_empty_tool_messages(self):
        assert extract_retrieved_item_ids([]) == set()

    def test_no_source_blocks(self):
        ids = extract_retrieved_item_ids([_tm("Some text without headers.")])
        assert ids == set()


# ---------------------------------------------------------------------------
# is_legal_conclusion
# ---------------------------------------------------------------------------


class TestIsLegalConclusion:
    def test_under_ethiopian_law(self):
        assert is_legal_conclusion("Under Ethiopian law, employers must provide notice.")

    def test_according_to_the_law(self):
        assert is_legal_conclusion("According to the law, this is prohibited.")

    def test_amharic_assertion(self):
        assert is_legal_conclusion("በኢትዮጵያ ህግ መሰረት፣ ውሉ ፀናለች።")

    def test_amharic_assertion2(self):
        assert is_legal_conclusion("በሕጉ መሠረት ይህ ድርጊት ሕጋዊ ነው።")

    def test_conversational_not_legal(self):
        assert not is_legal_conclusion("Hello, how can I help you today?")

    def test_empty_string(self):
        assert not is_legal_conclusion("")

    def test_none_returns_false(self):
        assert not is_legal_conclusion(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_answer
# ---------------------------------------------------------------------------


class TestValidateAnswer:
    def test_kb_gap_acknowledgement_passes(self):
        answer = "The knowledge base does not contain a specific provision on this point."
        result = validate_answer(answer, [])
        assert result.ok is True

    def test_amharic_kb_gap_passes(self):
        answer = "ይህን አንቀጽ ማግኘት አልቻልኩም።"
        result = validate_answer(answer, [])
        assert result.ok is True

    def test_conversational_answer_passes(self):
        answer = "Sure, let me help you with that request."
        result = validate_answer(answer, [])
        assert result.ok is True

    def test_legal_claim_with_pinpoint_passes(self):
        answer = "Under Ethiopian law, Article 1675 defines a contract."
        result = validate_answer(answer, [])
        assert result.ok is True

    def test_legal_claim_without_pinpoint_fails(self):
        answer = "Under Ethiopian law, employers must provide 30 days notice."
        result = validate_answer(answer, [])
        assert result.ok is False
        assert result.reason == "legal_claim_without_pinpoint"

    def test_amharic_legal_claim_without_pinpoint_fails(self):
        answer = "በኢትዮጵያ ህግ መሰረት፣ አሠሪው 30 ቀን ማሳወቅ አለበት።"
        result = validate_answer(answer, [])
        assert result.ok is False

    def test_amharic_legal_claim_with_pinpoint_passes(self):
        answer = "አዋጅ ቁጥር 1156/2019 አንቀጽ 43 መሰረት አሠሪው ማሳወቅ አለበት።"
        result = validate_answer(answer, [])
        assert result.ok is True

    def test_expect_kb_gap_empty_retrieval_passes(self):
        answer = "Under Ethiopian law, the court can..." 
        # With expect_kb_gap=True and empty tool messages, should pass
        result = validate_answer(answer, [], expect_kb_gap=True)
        assert result.ok is True

    def test_expect_kb_gap_with_retrieval_still_requires_pinpoint(self):
        # If there IS retrieval, even with expect_kb_gap, we need a cite
        answer = "Under Ethiopian law, the court can..."
        tool_msgs = [_tm(CITATION_BLOCK)]
        result = validate_answer(answer, tool_msgs, expect_kb_gap=True)
        assert result.ok is False

    def test_result_is_grounding_result(self):
        result = validate_answer("hello", [])
        assert isinstance(result, GroundingResult)

    def test_proclamation_cite_passes(self):
        answer = "Under Proclamation No. 1243/2021, the director may bind the company."
        result = validate_answer(answer, [])
        assert result.ok is True


# ---------------------------------------------------------------------------
# repair_message & safe_fallback_answer
# ---------------------------------------------------------------------------


class TestRepairAndFallback:
    def test_repair_message_english_for_english_answer(self):
        answer = "Under Ethiopian law, employers must..."
        msg = repair_message(answer)
        assert "search_legal_knowledge" in msg
        assert "Article" in msg or "pinpoint" in msg

    def test_repair_message_amharic_for_amharic_answer(self):
        amharic_answer = (
            "በኢትዮጵያ ህግ መሰረት፣ አሠሪው ማሳወቅ አለበት። "
            "ሠራተኛው ደረጃ ሳይጠነቀቅ ሊሰናበት ይችላል። "
            "ህጉ ይህን ይፈቅዳል። ተጨማሪ ምርምር ያስፈልጋል።"
        )
        msg = repair_message(amharic_answer)
        # Should return Amharic repair message
        assert "search_legal_knowledge" in msg
        assert "አዋጅ" in msg or "አንቀጽ" in msg

    def test_fallback_english_for_english_answer(self):
        answer = "Under Ethiopian law..."
        fb = safe_fallback_answer(answer)
        assert "qualified legal" in fb.lower() or "legal professional" in fb.lower()

    def test_fallback_amharic_for_amharic_answer(self):
        amharic_answer = "በኢትዮጵያ ህግ " * 5
        fb = safe_fallback_answer(amharic_answer)
        assert "ጠበቃ" in fb or "ባለሙያ" in fb


# ---------------------------------------------------------------------------
# _is_amharic_heavy
# ---------------------------------------------------------------------------


class TestIsAmharicHeavy:
    def test_amharic_dominant(self):
        text = "በኢትዮጵያ ህግ መሰረት ሠራተኛው ደረጃ ሳይጠነቀቅ ሊሰናበት ይችላል። " * 3
        assert _is_amharic_heavy(text) is True

    def test_english_dominant(self):
        text = "Under Ethiopian law, the employer must provide notice before termination." * 3
        assert _is_amharic_heavy(text) is False

    def test_empty_string(self):
        assert _is_amharic_heavy("") is False
