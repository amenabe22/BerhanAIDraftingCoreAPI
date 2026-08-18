#!/usr/bin/env python3
"""Join gold keys, Berhan answers, and exam stems; classify mismatch causes."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.legal.exam_parse import (  # noqa: E402
    classify_mismatch,
    split_exam_questions,
    strip_printed_keys,
)

EXMS = ROOT / "exms"
KB_MISS_RE = re.compile(r"(?i)knowledge base does not contain")
QHEAD_RE = re.compile(r"(?m)^##\s*Q(\d+)\s*$")

EXAM_IDS = {
    "Assosa University- Law Model Exit Exam - 2015 E.C.": "assosa-2015",
    "DDU LAW MODEL EXAM for Practice": "ddu-practice",
    "Debre Berhan University Model Exit Exam (2023)": "debre-berhan-2023",
    "HU- COL-MOCK Exam-2026": "hu-col-mock-2026",
    "LL.B Model Exam Debre Markos University": "llb-debre-markos",
    "Samara University Law model exam": "samara",
    "UoG, SOL Model  Exam": "uog-sol",
    "Wolaita Sodo MODEL Exam 2018": "wolaita-sodo-2018",
}

ASSOSA_WHY = {
    1: "Stem asks essential validity elements; sheet C is consent+capacity+object. Berhan picked A (includes form). Form is only required when the law prescribes it — classic EXCEPT/best-of-four miss.",
    2: "Stem asks which statement is WRONG about variation. Berhan restated the true principle (courts may not vary). The marked choice is the false statement about variation whenever performance is impossible.",
    5: "Stem asks the incorrect statement about damages. Berhan picked a true proposition (fault not always required) instead of the marked force-majeure exemption option.",
    6: "Counter-guarantor vs secondary guarantor. Berhan picked a distractor about who disfavors the creditor; the marked choice is a different comparison.",
    7: "Creditor-protection / simulation. Berhan picked the burden-of-proof distractor rather than the marked statement about provisions favoring the creditor.",
    11: "Company characteristics EXCEPT. Berhan answered from Civil Code association ideas (agency) instead of Commercial Code company features.",
}


def parse_berhan(md_path: Path) -> dict[int, dict]:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"(?m)^##\s*Q(\d+)\s*$", text)
    out: dict[int, dict] = {}
    seen: dict[int, int] = {}
    for i in range(1, len(parts), 2):
        n = int(parts[i])
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            continue
        body = parts[i + 1].strip()
        first = body.split("\n", 1)[0].strip() if body else ""
        out[n] = {"first": first, "body": body, "kb_miss": bool(KB_MISS_RE.search(body))}
    return out


def exam_id_for_answers(md: Path) -> str:
    stem = md.name[: -len("_answers.md")].rstrip(".")
    for prefix, eid in EXAM_IDS.items():
        if stem.startswith(prefix[:20]) or prefix in stem or stem in prefix:
            return eid
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:40]


def load_compare() -> dict:
    p = Path("/tmp/exam_compare.json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"exams": []}


def why_line(eid: str, q: int, cause: str, gold: str, berhan: str, stem: str, kb_miss: bool) -> str:
    if eid == "assosa-2015" and q in ASSOSA_WHY:
        return ASSOSA_WHY[q]
    if cause == "batch_retrieval":
        return "Full-exam dump: later questions reused early retrieval and fell back to the KB-miss sentence."
    if cause == "prompt_conflict":
        return "Exam overlay demanded a letter; system prompt forbade guessing — Berhan emitted a choice then the KB-miss line."
    if cause == "key_in_prompt":
        return "Printed Answer:/Option line was in the dumped exam text, so this mismatch is not a clean legal-skill signal."
    if cause == "alignment_noise":
        return "Sheet has more Answer lines than Berhan questions; gold Qn likely does not pair with Berhan Qn."
    if cause == "kb_gap_real":
        return "Topic is doctrine/theory/cassation-style material that the statute KB typically does not contain."
    if cause == "wrong_instrument":
        return "Company/commercial stem answered from Civil Code (or the wrong code family)."
    if cause == "option_mismatch":
        extra = " EXCEPT/WRONG stem." if re.search(r"(?i)except|wrong|not true|incorrect", stem) else ""
        return f"Retrieved or guessed a legally nearby option, not the marked key.{extra}"
    return f"{cause}: gold={gold!r} berhan={berhan[:80]!r}"


def main() -> None:
    compare = load_compare()
    by_file = {e["file"]: e for e in compare.get("exams", [])}
    rows = []
    cause_counts: dict[str, int] = {}

    for md in sorted(EXMS.glob("*_answers.md")):
        eid = exam_id_for_answers(md)
        berhan = parse_berhan(md)
        src = None
        stem_name = md.name[: -len("_answers.md")]
        for ext in (".docx", ".doc"):
            cand = EXMS / f"{stem_name}{ext}"
            if cand.exists():
                src = cand
                break
            cand = EXMS / f"{stem_name.rstrip('.')}{ext}"
            if cand.exists():
                src = cand
                break
        questions = []
        if src and src.suffix.lower() == ".docx":
            from examtest import extract_docx_text

            raw = extract_docx_text(src)
            questions = split_exam_questions(strip_printed_keys(raw))
        qmap = {q["n"]: q for q in questions}

        cmp = None
        for e in compare.get("exams", []):
            if Path(e["file"]).stem.rstrip(".") in md.name or e["file"].split(".")[0][:30] in md.name:
                cmp = e
                break
        details = {d["q"]: d for d in (cmp or {}).get("details", [])}

        for qn, rec in berhan.items():
            det = details.get(qn)
            status = det["status"] if det else ("kb-miss" if rec["kb_miss"] else "unknown")
            if status not in {"wrong", "kb-miss"} and not (rec["kb_miss"] and status != "correct"):
                if status != "wrong":
                    continue
            gold = (det or {}).get("gold") or ""
            stem = (qmap.get(qn) or {}).get("stem") or ""
            qtext = (qmap.get(qn) or {}).get("text") or stem
            kb = rec["kb_miss"] and status != "correct"
            if status == "correct":
                continue
            if not kb and status != "wrong":
                continue
            cause = classify_mismatch(
                exam_id=eid,
                q=qn,
                gold=str(gold),
                berhan=rec["first"],
                stem=qtext or stem,
                kb_miss=kb,
                status="wrong",
                berhan_body=rec["body"],
            )
            why = why_line(eid, qn, cause, str(gold), rec["first"], qtext or stem, kb)
            rows.append(
                {
                    "exam_id": eid,
                    "q": qn,
                    "gold": gold,
                    "berhan": rec["first"][:120],
                    "stem": (stem or qtext)[:160],
                    "kb_miss": kb,
                    "cause": cause,
                    "why": why,
                }
            )
            cause_counts[cause] = cause_counts.get(cause, 0) + 1

    report = {"cause_counts": cause_counts, "mismatches": rows}
    out = EXMS / "mismatch_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("causes", json.dumps(cause_counts, indent=2))
    print("n", len(rows), "wrote", out)


if __name__ == "__main__":
    main()
