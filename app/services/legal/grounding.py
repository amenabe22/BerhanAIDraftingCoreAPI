"""Grounding verifier for legal agent answers.

Validates that a final assistant answer:
1. Contains numbered article / proclamation pinpoints when making legal claims.
2. Handles KB-gap answers as always-valid (no pinpoint required).
3. Triggers a one-shot repair when validation fails.
4. Falls back to a safe template when repair still fails.

Design principles:
- Permissive on ambiguous cases: if we're not sure it's a legal claim, pass.
- Strict on explicit legal-assertion language without any article number.
- Never hard-block non-legal or conversational turns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.messages import ToolMessage

# ---------------------------------------------------------------------------
# Pinpoint citation detection
# Matches: Article N, Art. N, አንቀጽ N, Proclamation No. N, አዋጅ N, Section N, Reg. N
# ---------------------------------------------------------------------------

_PINPOINT_PAT = re.compile(
    r"""
    (?:
        \bArticle\s*\d+
        | \bArt\.?\s*\d+
        | አንቀጽ\s*\d+
        | \bProclamation\s+(?:No\.?\s*)?\d+
        | አዋጅ\s*(?:ቁጥር\s*)?\d+
        | \bSection\s*\d+
        | \bRegulation\s+(?:No\.?\s*)?\d+
        | ደንብ\s*(?:ቁጥር\s*)?\d+
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Legal-assertion detection
# These phrases indicate the model is making a legal claim that needs backing.
# ---------------------------------------------------------------------------

_LEGAL_ASSERT_PAT = re.compile(
    r"""
    (?:
        \bunder\s+ethiopian\s+law\b
        | \baccording\s+to\s+(?:the\s+)?(?:ethiopian\s+)?law\b
        | \bpursuant\s+to\b
        | \bthe\s+law\s+(?:requires?|provides?|states?|allows?|prohibits?)\b
        | \bethiopian\s+law\s+(?:requires?|provides?|states?|allows?|prohibits?)\b
        | \bas\s+per\s+(?:the\s+)?(?:ethiopian\s+)?law\b
        | \bunder\s+(?:the\s+)?(?:civil|commercial|labour|labor|penal|family|criminal)\s+code\b
        | \bunder\s+proclamation\b
        | \bthe\s+(?:civil|commercial|labour|labor|penal|family)\s+code\s+(?:provides?|states?|requires?)\b
        | \bproclamation\s+no\.?\s*\d+\b
        | \bአዋጅ\s*(?:ቁጥር\s*)?\d+\b
        | በኢትዮጵያ\s+ህግ|በኢትዮጵያ\s+ሕግ
        | በሕጉ\s+(?:መሠረት|መሰረት)|በህጉ\s+(?:መሠረት|መሰረት)
        | ሕጉ\s+እንደሚ(?:ያስቀምጠው|ያስቀምጥ|ይደነግገው)?
        | ህጉ\s+እንደሚ(?:ያስቀምጠው|ያስቀምጥ|ይደነግገው)?
        | አዋጁ\s+እንደሚ(?:ያስቀምጠው|ያስቀምጥ|ይደነግገው)?
        | ህጋዊ\s+ግዴታ|ህጋዊ\s+መብት|ሕጋዊ\s+ግዴታ|ሕጋዊ\s+መብት
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# KB-gap language detection
# When the answer already acknowledges the knowledge base has no provision,
# grounding passes unconditionally (no citation needed for a gap acknowledgement).
# ---------------------------------------------------------------------------

_KB_GAP_PAT = re.compile(
    r"""
    (?:
        knowledge\s+base\s+(?:does\s+not|did\s+not|doesn'?t|didn'?t)\s+(?:contain|have|find|cover)
        | not\s+(?:found|available)\s+in\s+(?:the\s+)?(?:knowledge|database|kb)
        | no\s+(?:specific\s+)?(?:provision|article|section)\s+(?:found|available|covering|on\s+this)
        | cannot\s+(?:confirm|find|cite|identify)\s+(?:a\s+)?(?:specific|relevant)\s+(?:provision|article|section)
        | (?:the\s+)?(?:knowledge\s+base|kb|legal\s+sources?)\s+(?:returns?|returned|contains?|has)\s+no
        | (?:no|not)\s+in\s+(?:my|the)\s+(?:knowledge|database|sources|kb)
        | this\s+(?:area|topic|question)\s+is\s+not\s+covered
        | not\s+covered\s+in\s+the\s+(?:knowledge|legal)\s+(?:base|database)
        | no\s+matching\s+provision
        | the\s+kb\s+has\s+no
        | no\s+relevant\s+(?:law|provision|article)\s+(?:found|in|available)
        | (?:the\s+)?(?:legal\s+)?(?:knowledge\s+base|database)\s+does\s+not\s+(?:contain|have|include)
        | (?:qualified|qualified\s+legal)\s+counsel\s+(?:may|should|must|would)\s+be\s+(?:needed|required|consulted)
        | consult\s+(?:a\s+)?(?:qualified\s+)?(?:legal\s+)?(?:professional|lawyer|attorney|counsel)
        | (?:i|we)\s+(?:am|are)\s+unable\s+to\s+(?:find|locate|confirm|cite)
        | cannot\s+(?:verify|confirm)\s+(?:this|the\s+(?:specific|exact))\s+(?:provision|article|figure|duration)
        | (?:no|not)\s+found\s+in\s+(?:the\s+)?(?:knowledge|legal)\s+(?:base|database|sources)
        | (?:the\s+kb\s+(?:does\s+)?not\s+(?:contain|have))
        | (?:ማግኘት\s+አልቻልኩም)
        | (?:አልተገኘም|አናገኘም)
        | (?:ተገቢ\s+መረጃ\s+አልተገኘም)
        | (?:ምንጭ\s+አልተገኘም)
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Fake back-reference (problematic pattern — noted but not hard-failed in v1)
_FAKE_BACKREF_PAT = re.compile(
    r"ከላይ\s+የተጠቀሱት\s+አንቀጾች|above[- ]?mentioned\s+articles?|as\s+cited\s+above",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Source-header parser (aligned with _parse_citations in app/main.py)
# ---------------------------------------------------------------------------

_SOURCE_HEADER_PAT = re.compile(
    r"\[Source:\s*(?P<doc>[^\|]+?)\s*\|\s*Article\s+(?P<item>[^\|\]]+?)\s*(?:\|[^\]]+)?\]"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class GroundingResult:
    ok: bool
    reason: str | None  # None when ok
    repair_attempted: bool = False
    expect_kb_gap: bool = False


def extract_retrieved_item_ids(tool_messages: list[ToolMessage]) -> set[str]:
    """Return the set of item_ids (article numbers) from all retrieved [Source: …] blocks."""
    ids: set[str] = set()
    for tm in tool_messages:
        content = tm.content or ""
        if not isinstance(content, str):
            continue
        for m in _SOURCE_HEADER_PAT.finditer(content):
            item = m.group("item").strip()
            if item:
                ids.add(item)
    return ids


def is_legal_conclusion(answer: str) -> bool:
    """Return True if the answer contains explicit legal-assertion language."""
    if not answer:
        return False
    return bool(_LEGAL_ASSERT_PAT.search(answer))


def validate_answer(
    answer: str,
    tool_messages: list[ToolMessage],
    *,
    expect_kb_gap: bool = False,
) -> GroundingResult:
    """Validate that the answer is adequately cited.

    Validation rules (evaluated in order — first matching rule sets result):

    1. Answer contains KB-gap acknowledgement → ``ok=True`` (no citation needed).
    2. Answer does not contain legal-assertion language → ``ok=True`` (not a legal claim).
    3. ``expect_kb_gap=True`` and retrieval is empty → ``ok=True`` (expected gap).
    4. Answer has no pinpoint article/section number → ``ok=False``.
    5. Otherwise → ``ok=True``.
    """
    if _KB_GAP_PAT.search(answer):
        return GroundingResult(ok=True, reason=None, expect_kb_gap=expect_kb_gap)

    if not is_legal_conclusion(answer):
        return GroundingResult(ok=True, reason=None)

    if expect_kb_gap:
        retrieved = extract_retrieved_item_ids(tool_messages)
        if not retrieved:
            return GroundingResult(ok=True, reason=None, expect_kb_gap=True)

    pinpoints = _PINPOINT_PAT.findall(answer)
    if not pinpoints:
        return GroundingResult(
            ok=False,
            reason="legal_claim_without_pinpoint",
            expect_kb_gap=expect_kb_gap,
        )

    return GroundingResult(ok=True, reason=None, expect_kb_gap=expect_kb_gap)


def _is_amharic_heavy(text: str) -> bool:
    """Return True if more than 20% of the text is Amharic script."""
    if not text:
        return False
    am = len(re.findall(r"[\u1200-\u137F]", text))
    return am > 20 and am / max(len(text), 1) > 0.20


# ---------------------------------------------------------------------------
# Repair / fallback messages
# ---------------------------------------------------------------------------

_REPAIR_MSG_EN = (
    "Your previous answer made legal claims without specific article numbers. "
    "Please call search_legal_knowledge again now and rewrite the answer. "
    "Every legal statement must include a pinpoint citation "
    "(e.g. 'Under Article N of Proclamation No. …' or 'አንቀጽ N…'). "
    "Only cite instruments that appear in the tool results you receive. "
    "If the question concerns a company, director, board, or shareholders, "
    "cite the Commercial Code (Proclamation No. 1243/2021) as the primary source, "
    "not the Civil Code associations chapter. "
    "If the knowledge base has no relevant provision, state that explicitly — "
    "do not invent timelines, freeze durations, or procedural steps."
)

_REPAIR_MSG_AM = (
    "የቀደመው መልስ ህጋዊ ድምዳሜ አካቷል ነገር ግን ልዩ የአንቀጽ ቁጥሮች አልነበሩም። "
    "እባክዎ አሁን search_legal_knowledge ን ደውለው መልሱን እንደገና ይፃፉ። "
    "ሁሉም ህጋዊ ሃሳቦች የሚጠቀሰውን አንቀጽ ቁጥር ማካተት አለባቸው "
    "(ለምሳሌ 'አዋጅ ቁጥር … አንቀጽ N …' ወይም 'Under Article N of Proclamation No. …'). "
    "ጥያቄው ኩባንያ፣ ዳይሬክተር ወይም አክሲዮን ካካተተ፣ "
    "የንግድ ሕግ (አዋጅ ቁጥር 1243/2021) ዋና ምንጭ ያድርጉ — የፍትሐብሔር ሕጉን ሳይሆን። "
    "የህጉ ዳታቤዝ ተገቢ ድንጋጌ ካላቀረበ ያሉትን ምንጮች ጠቅሰው ያሳውቁ — "
    "ምናባዊ የጊዜ ገደቦችን ወይም ሂደቶችን አይፍጠሩ።"
)

_FALLBACK_EN = (
    "I was unable to retrieve a specific legal provision to support a confident answer on this point. "
    "The legal knowledge base does not contain a matching provision for this query. "
    "Please consult a qualified legal professional for authoritative advice on this matter."
)

_FALLBACK_AM = (
    "ለጥያቄዎ ትክክለኛ ምላሽ ለመስጠት ከህጉ ዳታቤዝ ተገቢ ድንጋጌዎችን ማምጣት አልቻልኩም። "
    "ይህ ነጥብ በህጉ ዳታቤዝ ውስጥ ተሸፍኖ አይታይም። "
    "ለዚህ ጉዳይ ትክክለኛ ምክር ለማግኘት የህግ ባለሙያ ወይም ጠበቃ ማማከር ያስፈልጋል።"
)


def repair_message(answer: str) -> str:
    """Return the repair prompt to inject when grounding fails on the first attempt."""
    return _REPAIR_MSG_AM if _is_amharic_heavy(answer) else _REPAIR_MSG_EN


def safe_fallback_answer(answer: str) -> str:
    """Return a safe fallback answer after repair still fails."""
    return _FALLBACK_AM if _is_amharic_heavy(answer) else _FALLBACK_EN
