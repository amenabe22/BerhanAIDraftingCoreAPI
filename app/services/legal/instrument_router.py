"""Instrument router: maps a user query to the most relevant Ethiopian legal instruments.

The router uses a deterministic keyword/regex rule table (English + Amharic) to
choose the best instrument for a query. A ``RouteDecision`` is returned and used to:

1. Enrich the search query with instrument-specific terms (``query_suffix``).
2. Signal to the grounding verifier that KB gaps are expected (``expect_kb_gap``).
3. Hint which document families should NOT be the primary citation
   (``forbidden_primary``) — e.g. Civil Code for company/director questions.

v1 note: ``preferred_document_ids`` is intentionally empty; filtering is done via
query enrichment only (safer, no hard Qdrant filter that could empty the result set
when metadata IDs differ from expectation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RouteDecision:
    """Routing output for a single user query."""

    preferred_document_ids: list[str]  # v1: always empty; reserved for future filter
    query_suffix: str  # appended to the tool query to anchor retrieval
    expect_kb_gap: bool  # True when the KB is unlikely to have specific content
    forbidden_primary: list[str]  # document_id substrings that must NOT be primary source


# ---------------------------------------------------------------------------
# Rule table — each entry: (trigger_patterns, query_suffix, expect_gap, forbidden_primary)
# Rules are evaluated top-to-bottom; first match wins.
# ---------------------------------------------------------------------------

_RULES: list[tuple[list[str], str, bool, list[str]]] = [
    # Commercial / company law
    (
        [
            r"(?i)\bcompan(?:y|ies)\b",
            r"(?i)\bdirector\b",
            r"(?i)\bboard[\s\-]?(?:resolution|meeting|of\s+directors)\b",
            r"(?i)\bshareholder",
            r"(?i)\bshare\s+capital\b",
            r"(?i)\bcommercial\s+part(?:ner|nership)",
            r"(?i)\bprivate\s+limited\b",
            r"(?i)\bplc\b",
            r"(?i)\bshare\s+company\b",
            r"(?i)\bincorporat",
            r"ኩባንያ|ዳይሬክተር|ሕዝባዊ\s*ኩባንያ|አክሲዮን|ቦርድ",
        ],
        "Commercial Code Proclamation 1243/2021 company director board shareholder",
        False,
        ["civil_code", "civil-code"],
    ),
    # Labour / employment
    (
        [
            r"(?i)\bemploy(?:ment|ee|er|ees|ers)\b",
            r"(?i)\btermina(?:tion|te)d?\b",
            r"(?i)\bwages?\b",
            r"(?i)\blabou?r\b",
            r"(?i)\bdismissal\b",
            r"(?i)\bseverance\b",
            r"(?i)\bnotice\s+period\b",
            r"(?i)\bwork(?:place|er)\b",
            r"(?i)\bprobation\b",
            r"(?i)\bapprentic",
            r"ሠራተኛ|ሰራተኛ|ደመወዝ|ስንብት|አሠሪ|አሰሪ|ቅጥር|ፕሮቤሽን",
        ],
        "Labour Proclamation 1156/2019 employment termination wages worker",
        False,
        [],
    ),
    # Tax / customs
    (
        [
            r"(?i)\bincome\s*tax\b",
            r"(?i)\bvalue\s*added\s*tax\b",
            r"(?i)\bVAT\b",
            r"(?i)\bcustoms?\b",
            r"(?i)\bimport\s+(?:duty|tax|tariff)\b",
            r"(?i)\bexcise\s+tax\b",
            r"(?i)\btax\s+(?:exempt|exemption|free|evasion|avoidance)\b",
            r"(?i)\bexempt(?:ion)?\s+from\s+(?:income\s+)?tax\b",
            r"(?i)\btax\s+(?:return|declaration|filing)\b",
            r"(?i)\bwithholding\s+tax\b",
            r"(?i)\btax\b",
            r"ግብር|ጉምሩክ|ቀረጥ|ታክስ|የተጨማሪ\s*እሴት",
        ],
        "Income Tax Proclamation VAT customs duty Ethiopia tax exempt",
        False,
        [],
    ),
    # Inheritance / limitation
    (
        [
            r"(?i)\binheritance\b",
            r"(?i)\bestate\s+(?:of|distribution|division)\b",
            r"(?i)\blegatee\b",
            r"(?i)\bbequest\b",
            r"(?i)\bstatute\s+of\s+limitations?\b",
            r"(?i)\blimitation\s+(?:period|clause)\b",
            r"(?i)\byirga\b",
            r"(?i)\bprescription\s+(?:period|of)\b",
            r"(?i)\bsuccessors?\b",
            r"(?i)\bwill\s+and\s+testament\b",
            r"(?i)\btestament(?:ary)?\b",
            r"ውርስ|ወራሽ|ኑዛዜ|የይርጋ|ይርጋ|ዝርዝር\s*ወራሽ",
        ],
        "Civil Code inheritance succession limitation Article 1677 1000 995",
        False,
        [],
    ),
    # General contract / civil obligations — catch-all before land/property rules
    (
        [
            r"(?i)\bcontract\b",
            r"(?i)\bobligations?\b",
            r"(?i)\bbreach\s+of\s+contract\b",
            r"(?i)\brescind(?:ing|ed)?\b",
            r"(?i)\bcancell?ation\s+of\s+(?:the\s+)?(?:contract|agreement)\b",
            r"(?i)\bterminat(?:e|ing|ion)\s+(?:the\s+)?(?:contract|agreement)\b",
            r"ውል|ኮንትራት|ዋስትና|ውሉን\s*ለማፍረስ|ውሉን\s*ለማቋረጥ",
        ],
        "Civil Code contract obligation termination Article Ethiopian",
        False,
        [],
    ),
    # Cassation / case law — always expect KB gap
    (
        [
            r"(?i)\bcassation\b",
            r"(?i)\bsebre\b",
            r"(?i)\bsebr\b",
            r"(?i)\bfederal\s+supreme\s+court\s+(?:ruling|decision)\b",
            r"ሰበር|ሰ/መ/ቁ|ሰ\.?መ\.?ቁ|ሰ\.?መ\b",
        ],
        "cassation Federal Supreme Court ruling decision volume",
        True,  # expect KB gap
        [],
    ),
    # Constitution / human rights
    (
        [
            r"(?i)\bconstitution(?:al)?\b",
            r"(?i)\bhuman\s+rights?\b",
            r"(?i)\bfundamental\s+freedom\b",
            r"(?i)\bfundamental\s+rights?\b",
            r"(?i)\bFDRE\b",
            r"ሕገ[- ]?መንግሥት|ህገ[- ]?መንግስት|ሰብዓዊ[- ]?መብት|ሕገ[- ]?ወጥ",
        ],
        "FDRE Constitution fundamental rights Article freedom",
        False,
        [],
    ),
    # Family law
    (
        [
            r"(?i)\bfamily\s+(?:law|code|court)\b",
            r"(?i)\bmarriage\b",
            r"(?i)\bdivorce\b",
            r"(?i)\bcustody\b",
            r"(?i)\bspouse\b",
            r"(?i)\bconjugal\b",
            r"ጋብቻ|ፍቺ|ቤተሰብ|ባልና\s*ሚስት|ስጋ",
        ],
        "Family Code marriage divorce custody Ethiopian",
        False,
        [],
    ),
    # Land / property
    (
        [
            r"(?i)\bland\s+(?:use|law|registration|policy|right|title|tenure)\b",
            r"(?i)\breal\s+estate\b",
            r"(?i)\btenancy\b",
            r"(?i)\blease\s+(?:agreement|hold|land)\b",
            r"(?i)\bproperty\s+(?:right|law|title|registration)\b",
            r"(?i)\bexpropriat",
            r"መሬት|ሊዝ|ርስት|ቦታ\s*(?:ይዞታ|ምዝገባ)",
        ],
        "Ethiopian land property ownership lease expropriation",
        False,
        [],
    ),
    # Criminal / penal
    (
        [
            r"(?i)\bcriminal\b",
            r"(?i)\bpenal\s+(?:code|law)\b",
            r"(?i)\barrest\b",
            r"(?i)\bprison\b",
            r"(?i)\boffence\b",
            r"(?i)\bfelony\b",
            r"(?i)\bmisdemean",
            r"(?i)\bprosecution\b",
            r"(?i)\bsentence\b",
            r"ወንጀል|ቅጣት|እስር|ክስ|ፍርደኛ",
        ],
        "Penal Code criminal offence punishment sentence Ethiopia",
        False,
        [],
    ),
    # Financial Intelligence / AML / banking regulation — partial KB gap expected
    (
        [
            r"(?i)\bFIS\b",
            r"(?i)\bfinancial\s+intelligence\b",
            r"(?i)\bmoney\s+laundering\b",
            r"(?i)\bAML\b",
            r"(?i)\bterrorist\s+financing\b",
            r"(?i)\bfreeze\s+(?:account|order)\b",
            r"(?i)\baccount\s+(?:freeze|block|suspension)\b",
            r"ፋይናንሻል\s*ኢንተለጀንስ|ህገወጥ\s*ገንዘብ\s*ዝውውር|አካውንት\s*እግድ",
        ],
        "Financial Intelligence Service anti-money laundering account freeze Proclamation",
        True,  # regulatory procedures not always in statute KB
        [],
    ),
]


def route(query: str) -> RouteDecision:
    """Return a ``RouteDecision`` for the given user query string.

    Only the first matching rule fires (most-specific first ordering).
    When no rule matches, a neutral ``RouteDecision`` is returned.
    """
    for triggers, suffix, gap, forbidden in _RULES:
        for pat in triggers:
            if re.search(pat, query):
                return RouteDecision(
                    preferred_document_ids=[],
                    query_suffix=suffix,
                    expect_kb_gap=gap,
                    forbidden_primary=forbidden,
                )

    # No specific instrument matched
    return RouteDecision(
        preferred_document_ids=[],
        query_suffix="",
        expect_kb_gap=False,
        forbidden_primary=[],
    )
