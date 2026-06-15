"""Opt-in A/B harness for edit-case fixtures (offline deterministic cases).

Fixture fields:
- instruction: natural-language only (users never reference block_ids like b1/b2)
- expected_target_blocks: internal test assertions — not part of user input
"""

import json
from pathlib import Path

import pytest

from app.services.drafting.editing.agent import SemanticEditAgent
from app.services.drafting.editing.tiptap import extract_blocks_from_tiptap
from tests.fixtures.tiptap_docs import make_amharic_doc, make_en_doc

CASES_DIR = Path(__file__).parent / "fixtures" / "edit_cases"


def _load_cases():
    cases = []
    for path in sorted(CASES_DIR.glob("*.json")):
        cases.append(json.loads(path.read_text()))
    return cases


def _assert_no_block_id_in_instruction(instruction: str) -> None:
    """Guardrail: fixtures must use natural language, not internal block ids."""
    lowered = instruction.lower()
    for token in (" block b", "block_id", " b1", " b2", " b3"):
        assert token not in lowered, f"Fixture instruction must be natural language, found {token!r}"


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["id"])
@pytest.mark.asyncio
async def test_offline_edit_case(case):
    """Runs deterministic fast-path cases without LLM (global replace)."""
    _assert_no_block_id_in_instruction(case["instruction"])

    doc_fixture = case.get("doc_fixture", "en")
    doc = make_amharic_doc() if doc_fixture == "amharic" else make_en_doc()
    lang = case.get("document_language", "en")

    # Only global-replace fast path runs offline without mocked LLM
    if "everywhere" not in case["instruction"].lower() and "change" not in case["instruction"].lower():
        pytest.skip("Non-deterministic case — run with EDIT_AB_LIVE=1")

    agent = SemanticEditAgent()
    result = await agent.edit_document(
        doc, instruction=case["instruction"], document_language=lang
    )

    if case.get("expected_removed"):
        remaining = {b["block_id"] for b in extract_blocks_from_tiptap(result["document"])}
        for bid in case["expected_removed"]:
            assert bid not in remaining

    if case.get("expected_contains"):
        all_text = " ".join(b["text"] for b in extract_blocks_from_tiptap(result["document"]))
        assert case["expected_contains"].lower() in all_text.lower()


@pytest.mark.manual
@pytest.mark.asyncio
async def test_ab_harness_report():
    """Opt-in live harness: set EDIT_AB_LIVE=1 to run all cases against real LLM."""
    import os

    if os.getenv("EDIT_AB_LIVE") != "1":
        pytest.skip("Set EDIT_AB_LIVE=1 to run live A/B harness")

    agent = SemanticEditAgent()
    passed = 0
    total = 0
    for case in _load_cases():
        _assert_no_block_id_in_instruction(case["instruction"])
        total += 1
        doc = make_amharic_doc() if case.get("doc_fixture") == "amharic" else make_en_doc()
        lang = case.get("document_language", "en")
        result = await agent.edit_document(
            doc, instruction=case["instruction"], document_language=lang
        )
        ok = not result.get("no_effective_change")
        passed += int(ok)
        print(f"  {case['id']}: {'PASS' if ok else 'FAIL'} — {case['instruction'][:60]}...")
    print(f"A/B harness: {passed}/{total} cases produced changes")
