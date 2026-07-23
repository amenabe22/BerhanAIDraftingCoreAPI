"""Citation regression harness for legal agent chat failure cases.

Offline mode (default):
    Tests the instrument_router and grounding.validate_answer functions
    against mocked tool payloads from the failure-case fixtures. No LLM, no
    network. Runs as part of the normal pytest suite.

Manual live mode:
    Set CHAT_CITATION_LIVE=1 and ensure the API is running.
    Sends real queries to /legal-agent/stream and asserts cite quality.
    Guarded by @pytest.mark.manual.

Fixture schema (tests/fixtures/chat_cases/*.json):
    id                        – unique case identifier
    description               – human-readable description
    query                     – the user's question
    expect_kb_gap             – True if KB gap is expected
    require_pinpoint          – True if answer must contain Article/አንቀጽ/Proc
    preferred_instruments     – list of substrings expected in route query_suffix
    forbidden_primary         – document_id substrings that must NOT be sole cite
    tool_payload              – mocked [Source: …] block(s) as returned by tool
    expected_answer_contains_any – list of strings; at least one must be in answer
    expected_answer_forbidden_any – strings that must not appear in answer
    kb_gap_scenario           – True for cases where empty retrieval is intended
"""

import json
import os
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage

from app.services.legal.grounding import (
    extract_retrieved_item_ids,
    validate_answer,
)
from app.services.legal.instrument_router import route

CASES_DIR = Path(__file__).parent / "fixtures" / "chat_cases"


def _load_cases() -> list[dict]:
    cases = []
    for path in sorted(CASES_DIR.glob("*.json")):
        cases.append(json.loads(path.read_text(encoding="utf-8")))
    return cases


def _tool_messages_from_payload(payload: str) -> list[ToolMessage]:
    """Wrap fixture tool_payload string into a list of ToolMessage."""
    if not payload:
        return []
    return [ToolMessage(content=payload, tool_call_id="fixture_tc")]


# ---------------------------------------------------------------------------
# Offline: instrument router tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["id"])
def test_instrument_router_offline(case: dict) -> None:
    """Route decision should match expected instrument hints from fixture."""
    query = case["query"]
    preferred = case.get("preferred_instruments", [])
    if not preferred:
        pytest.skip("No preferred_instruments to assert in this fixture.")

    decision = route(query)
    combined = (decision.query_suffix + " ".join(decision.preferred_document_ids)).lower()
    matches = [p.lower() in combined for p in preferred]
    assert any(matches), (
        f"[{case['id']}] Expected at least one of {preferred!r} in route suffix "
        f"'{decision.query_suffix}' for query: {query!r}"
    )


# ---------------------------------------------------------------------------
# Offline: grounding validator tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [c for c in _load_cases() if c.get("kb_gap_scenario")],
    ids=lambda c: c["id"],
)
def test_grounding_kb_gap_scenario(case: dict) -> None:
    """Empty retrieval + expect_kb_gap → validate_answer must return ok."""
    # Simulate an answer that properly acknowledges the gap
    gap_answer_en = (
        "The knowledge base does not contain a specific provision on this point. "
        "Please consult a qualified legal professional."
    )
    gap_answer_am = "ይህን ድንጋጌ ማግኘት አልቻልኩም። ጠበቃ ያማክሩ።"

    tool_msgs = _tool_messages_from_payload(case.get("tool_payload", ""))
    for answer in (gap_answer_en, gap_answer_am):
        result = validate_answer(answer, tool_msgs, expect_kb_gap=True)
        assert result.ok is True, (
            f"[{case['id']}] KB-gap answer should pass grounding. Reason: {result.reason}"
        )


@pytest.mark.parametrize(
    "case",
    [c for c in _load_cases() if c.get("require_pinpoint") and not c.get("kb_gap_scenario")],
    ids=lambda c: c["id"],
)
def test_grounding_requires_pinpoint_when_tool_payload_provided(case: dict) -> None:
    """When tool_payload has citation blocks, an uncited legal answer must fail."""
    if not case.get("tool_payload"):
        pytest.skip("No tool_payload — grounding cannot check citation overlap.")

    tool_msgs = _tool_messages_from_payload(case["tool_payload"])
    retrieved_ids = extract_retrieved_item_ids(tool_msgs)

    if not retrieved_ids:
        pytest.skip("Tool payload has no [Source: …] headers — skipping.")

    # An answer that asserts law but cites no article number must fail
    uncited_answer = "Under Ethiopian law, the relevant rules apply in this situation."
    result = validate_answer(uncited_answer, tool_msgs, expect_kb_gap=False)
    assert result.ok is False, (
        f"[{case['id']}] Uncited legal answer should fail grounding. "
        f"Retrieved ids: {retrieved_ids}"
    )


@pytest.mark.parametrize(
    "case",
    [c for c in _load_cases() if c.get("require_pinpoint") and not c.get("kb_gap_scenario")],
    ids=lambda c: c["id"],
)
def test_grounding_passes_with_pinpoint(case: dict) -> None:
    """A legal answer that includes an article number from the tool payload must pass."""
    if not case.get("tool_payload"):
        pytest.skip("No tool_payload for this case.")

    tool_msgs = _tool_messages_from_payload(case["tool_payload"])
    retrieved_ids = extract_retrieved_item_ids(tool_msgs)

    if not retrieved_ids:
        pytest.skip("No [Source: …] headers in tool_payload — skipping.")

    # Construct an answer with a valid pinpoint from the retrieved ids
    some_id = next(iter(retrieved_ids))
    cited_answer = f"Under Ethiopian law, Article {some_id} provides that the relevant rules apply."
    result = validate_answer(cited_answer, tool_msgs, expect_kb_gap=False)
    assert result.ok is True, (
        f"[{case['id']}] Answer with Article {some_id} should pass grounding. "
        f"Reason: {result.reason}"
    )


# ---------------------------------------------------------------------------
# Offline: specific instrument rules from fixture metadata
# ---------------------------------------------------------------------------


def test_commercial_director_forbidden_primary() -> None:
    """Commercial director query should have civil_code in forbidden_primary."""
    case_path = CASES_DIR / "commercial_director_194.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    decision = route(case["query"])
    forbidden = [f.lower() for f in decision.forbidden_primary]
    assert any("civil" in f for f in forbidden), (
        f"Expected civil_code in forbidden_primary for query {case['query']!r}, "
        f"got: {decision.forbidden_primary}"
    )


def test_cassation_expect_kb_gap() -> None:
    """Cassation query must set expect_kb_gap=True in route decision."""
    case_path = CASES_DIR / "cassation_33945_178.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    decision = route(case["query"])
    assert decision.expect_kb_gap is True, (
        f"Expected expect_kb_gap=True for cassation query {case['query']!r}"
    )


def test_fis_expect_kb_gap() -> None:
    """FIS / AML query must set expect_kb_gap=True in route decision."""
    case_path = CASES_DIR / "fis_basis_197.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    decision = route(case["query"])
    assert decision.expect_kb_gap is True, (
        f"Expected expect_kb_gap=True for FIS query {case['query']!r}"
    )


# ---------------------------------------------------------------------------
# Manual live harness — @pytest.mark.manual + CHAT_CITATION_LIVE=1
# ---------------------------------------------------------------------------


@pytest.mark.manual
@pytest.mark.asyncio
async def test_live_citation_harness() -> None:
    """Live harness: sends real queries to /legal-agent/stream and checks citations.

    Requirements:
        - API running at BERHAN_API_URL (default http://localhost:8000)
        - CHAT_CITATION_LIVE=1 set in environment

    Scoring:
        - Each case: PASS if answer contains ≥1 expected string AND no forbidden strings
        - Suite target: ≥5/6 cases pass
    """
    if os.getenv("CHAT_CITATION_LIVE") != "1":
        pytest.skip("Set CHAT_CITATION_LIVE=1 to run live citation harness.")

    import httpx

    base_url = os.getenv("BERHAN_API_URL", "http://localhost:8000")
    cases = _load_cases()
    passed = 0
    total = len(cases)

    for case in cases:
        answer_tokens: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                async with client.stream(
                    "POST",
                    f"{base_url}/legal-agent/stream",
                    json={"message": case["query"], "language": None},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[len("data: "):]
                        try:
                            evt = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if evt.get("type") == "token":
                            answer_tokens.append(evt.get("content", ""))
        except Exception as e:
            print(f"[{case['id']}] SKIP (request error): {e}")
            continue

        answer = "".join(answer_tokens)
        contains_any = case.get("expected_answer_contains_any", [])
        forbidden_any = case.get("expected_answer_forbidden_any", [])

        ok_contains = not contains_any or any(s.lower() in answer.lower() for s in contains_any)
        ok_forbidden = not any(s.lower() in answer.lower() for s in forbidden_any)
        case_pass = ok_contains and ok_forbidden

        if case_pass:
            passed += 1
            print(f"[{case['id']}] PASS")
        else:
            if not ok_contains:
                print(f"[{case['id']}] FAIL — expected one of {contains_any!r}")
            if not ok_forbidden:
                fails = [s for s in forbidden_any if s.lower() in answer.lower()]
                print(f"[{case['id']}] FAIL — forbidden strings found: {fails!r}")

    min_pass = max(1, total - 1)  # ≥ 5/6
    assert passed >= min_pass, (
        f"Live citation harness: {passed}/{total} passed. Expected ≥{min_pass}."
    )
