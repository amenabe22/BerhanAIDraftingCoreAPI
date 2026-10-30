"""High-accuracy semantic edit agent: locate -> edit -> verify -> apply."""

from __future__ import annotations

import logging
import re
import time
import uuid
from copy import deepcopy
from typing import Any

from app.config import settings
from app.services.drafting.editing import llm as edit_llm
from app.services.drafting.editing.ops import apply_operations, compute_simple_diff
from app.services.drafting.editing.prompts import (
    diff_system_prompt,
    diff_user_prompt,
    edit_system_prompt,
    edit_user_prompt,
    locate_system_prompt,
    locate_user_prompt,
    verify_system_prompt,
    verify_user_prompt,
)
from app.services.drafting.editing.retrieval import (
    build_document_skeleton,
    expand_section_blocks,
    match_named_section_headings,
    merge_candidate_blocks,
    neighbor_blocks,
    rank_blocks_for_instruction,
    structural_candidate_blocks,
)
from app.services.drafting.editing.schemas import (
    ensure_op_ids,
    generate_fallback_operation,
    validate_operations,
)
from app.services.drafting.editing.tiptap import (
    detect_language,
    extract_blocks_from_tiptap,
    validate_tiptap_structure,
)

logger = logging.getLogger(__name__)

_GLOBAL_REPLACE_RE = re.compile(
    r"(?:change|replace|swap)\s+(.+?)\s+(?:to|with)\s+(.+?)(?:\s+(?:everywhere|throughout|in all|globally))?$",
    re.IGNORECASE,
)


class SemanticEditAgent:
    """Drop-in replacement for ai_editor_mvp EditingAgent."""

    async def edit_document(
        self,
        doc_json: dict[str, Any],
        instruction: str,
        doc_id: str | None = None,
        document_language: str | None = None,
        compliance_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        operation_id = uuid.uuid4().hex[:8]
        started = time.perf_counter()
        metrics: dict[str, Any] = {
            "operation_id": operation_id,
            "doc_id": doc_id,
            "stages": {},
        }

        output_language = (document_language or "").lower()
        if output_language not in ("en", "am", "om"):
            output_language = detect_language(instruction)

        if not validate_tiptap_structure(doc_json):
            return self._error_result(
                doc_json,
                operation_id,
                metrics,
                started,
                "Invalid TipTap document structure",
            )

        all_blocks = extract_blocks_from_tiptap(doc_json)
        if not all_blocks:
            return self._no_change_result(
                doc_json,
                operation_id,
                metrics,
                started,
                "Document has no editable blocks",
            )

        try:
            # Deterministic global replace fast-path
            fast_ops = self._try_global_replace(instruction, all_blocks)
            if fast_ops:
                metrics["stages"]["fast_path"] = "global_replace"
                return await self._finalize(
                    doc_json,
                    fast_ops,
                    instruction,
                    operation_id,
                    metrics,
                    started,
                    output_language,
                )

            # LOCATE: skeleton + dimensional plan + semantic/structural candidates
            t0 = time.perf_counter()
            skeleton = build_document_skeleton(all_blocks)
            ranked = rank_blocks_for_instruction(instruction, all_blocks)
            metrics["stages"]["rerank_ms"] = int((time.perf_counter() - t0) * 1000)
            metrics["stages"]["skeleton_stats"] = skeleton.get("stats")

            locate_result, targets, scope, locate_confidence = await self._locate_targets(
                instruction,
                all_blocks,
                ranked,
                skeleton,
                metrics,
            )

            if not targets:
                return self._no_change_result(
                    doc_json,
                    operation_id,
                    metrics,
                    started,
                    "Could not confidently locate target block(s) for the instruction. "
                    "Try naming a section or quoting a short phrase from the clause to edit.",
                )

            target_ids = {t["block_id"] for t in targets if t.get("block_id")}

            # Section expand: heading seeds → full section body (fixes multi-block
            # requests like "shorten DEFINITIONS" that previously got tiny context).
            # Skip expand when structural multi-body targets were already resolved
            # (thin body ids), but always expand named heading seeds.
            selection_mode = (locate_result or {}).get("selection_mode") or "semantic"
            spatial_scope = (locate_result or {}).get("spatial_scope") or "pinpoint"
            seed_blocks = [b for b in all_blocks if b["block_id"] in target_ids]
            has_heading_seed = any(b.get("type") == "heading" for b in seed_blocks)
            named_multi = selection_mode == "explicit_section" and (
                spatial_scope == "multi_section" or len(target_ids) > 1
            )
            structural_body_multi = selection_mode in ("structural", "hybrid") or (
                spatial_scope == "document_wide" and selection_mode != "explicit_section"
            )
            # multi_section + explicit_section = named headings → expand; structural thinness → don't
            if has_heading_seed and (
                named_multi
                or selection_mode == "explicit_section"
                or not structural_body_multi
            ):
                expanded_ids = expand_section_blocks(all_blocks, target_ids)
                if expanded_ids != target_ids:
                    metrics["stages"]["section_expand"] = {
                        "before": len(target_ids),
                        "after": len(expanded_ids),
                    }
                    target_ids = expanded_ids
                    if scope == "single":
                        scope = "section" if len(target_ids) <= 3 else "multiple"
                    elif named_multi:
                        scope = "multiple"

            structural_multi = structural_body_multi or (
                spatial_scope == "multi_section" and selection_mode != "explicit_section"
            )

            if scope == "global" and not structural_multi and not named_multi:
                edit_context = [
                    b
                    for b in all_blocks
                    if any(
                        tok in (b.get("text") or "").lower()
                        for tok in instruction.lower().split()
                        if len(tok) > 3
                    )
                ] or ranked[:10]
            elif (
                scope in ("section", "multiple")
                or len(target_ids) > 3
                or structural_multi
                or named_multi
            ):
                # Full section / multi-target context — no 3-block neighbor cap.
                id_order = {b["block_id"]: i for i, b in enumerate(all_blocks)}
                edit_context = sorted(
                    [b for b in all_blocks if b["block_id"] in target_ids],
                    key=lambda b: id_order.get(b["block_id"], 0),
                )
                max_section = settings.EDIT_SECTION_MAX_BLOCKS
                if max_section and len(edit_context) > max_section:
                    if named_multi and has_heading_seed:
                        # Fair per-section budget so later named sections aren't dropped
                        heading_seeds = [
                            b["block_id"]
                            for b in seed_blocks
                            if b.get("type") == "heading"
                        ]
                        per = max(3, max_section // max(1, len(heading_seeds)))
                        kept: list[dict[str, Any]] = []
                        for hid in heading_seeds:
                            section_ids = expand_section_blocks(all_blocks, {hid})
                            section_blocks = [
                                b
                                for b in edit_context
                                if b["block_id"] in section_ids
                            ]
                            kept.extend(section_blocks[:per])
                            if len(kept) >= max_section:
                                break
                        edit_context = kept[:max_section]
                    else:
                        edit_context = edit_context[:max_section]
                    target_ids = {b["block_id"] for b in edit_context}
                if (structural_multi or named_multi) and scope == "single":
                    scope = "multiple"
            else:
                edit_context = neighbor_blocks(all_blocks, target_ids, window=1)

            all_block_ids = [b["block_id"] for b in all_blocks]
            validated_operations: list[dict[str, Any]] = []
            patched_doc = doc_json
            revision_feedback: str | None = None
            attempts = 0
            max_attempts = 1 + settings.EDIT_MAX_REVISIONS

            while attempts < max_attempts:
                attempts += 1
                ops_json = await edit_llm.complete_json(
                    edit_system_prompt(output_language),
                    edit_user_prompt(
                        instruction,
                        edit_context,
                        all_block_ids,
                        revision_feedback=revision_feedback,
                    ),
                )
                if not ops_json:
                    break

                is_valid, error = validate_operations(ops_json)
                if not is_valid:
                    revision_feedback = f"Invalid operations JSON: {error}"
                    metrics.setdefault("validation_errors", []).append(error)
                    continue

                ops_json["operations"] = ensure_op_ids(ops_json["operations"])
                validated_operations = self._filter_valid_ops(
                    ops_json["operations"], doc_json, scope, target_ids
                )
                if not validated_operations:
                    revision_feedback = "All operations referenced invalid block_ids. Use only valid block_ids."
                    continue

                candidate_doc = apply_operations(doc_json, validated_operations)

                before_ctx = [b for b in all_blocks if b["block_id"] in target_ids]
                after_ctx = [
                    b
                    for b in extract_blocks_from_tiptap(candidate_doc)
                    if b["block_id"] in target_ids
                ]

                verify_result = await edit_llm.complete_json(
                    verify_system_prompt(),
                    verify_user_prompt(instruction, before_ctx, after_ctx),
                )
                metrics.setdefault("verify_attempts", []).append(verify_result)

                if verify_result and verify_result.get("passed"):
                    patched_doc = candidate_doc
                    break

                revision_feedback = (verify_result or {}).get("feedback") or (
                    "; ".join((verify_result or {}).get("issues") or [])
                    or "Verification failed; revise the edit."
                )
                if attempts >= max_attempts:
                    patched_doc = candidate_doc

            if not validated_operations:
                fallback = generate_fallback_operation(doc_json, ranked, "No valid operations")
                return self._no_change_result(
                    doc_json,
                    operation_id,
                    metrics,
                    started,
                    "Edit could not produce valid operations for the instruction.",
                    operations=fallback.get("operations") or [],
                )

            return await self._finalize(
                doc_json,
                validated_operations,
                instruction,
                operation_id,
                metrics,
                started,
                output_language,
                patched_doc=patched_doc,
            )

        except Exception as exc:
            logger.exception("Semantic edit failed op=%s err=%s", operation_id, exc)
            return self._error_result(doc_json, operation_id, metrics, started, str(exc))

    async def _locate_targets(
        self,
        instruction: str,
        all_blocks: list[dict[str, Any]],
        ranked: list[dict[str, Any]],
        skeleton: dict[str, Any],
        metrics: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str, float]:
        """Dimensional locate with structural/semantic candidates and soft recovery."""

        def _normalize_locate(raw: dict[str, Any] | None) -> dict[str, Any]:
            data = dict(raw or {})
            data.setdefault("spatial_scope", "pinpoint")
            data.setdefault("selection_mode", "semantic")
            data.setdefault("thinness", "below_median")
            data.setdefault("edit_intent", "other")
            data.setdefault("scope", "single")
            data.setdefault("targets", [])
            data.setdefault("confidence", 0.0)
            return data

        def _needs_structural(plan: dict[str, Any]) -> bool:
            # Named outline headings use explicit_section even with multi_section scope —
            # never replace those heading seeds with thin-body structural filters.
            if plan.get("selection_mode") == "explicit_section":
                return False
            return (plan.get("selection_mode") in ("structural", "hybrid")) or (
                plan.get("spatial_scope") in ("document_wide", "multi_section")
            )

        def _infer_edit_action(plan: dict[str, Any] | None = None) -> str:
            intent = ((plan or {}).get("edit_intent") or "").lower()
            if intent in ("expand", "shorten", "rewrite", "replace"):
                return "expand" if intent == "expand" else intent
            lower = instruction.lower()
            expand_hints = (
                "expand",
                "enrich",
                "more detail",
                "አስፋ",
                "ማስፋት",
                "አብራራ",
                "በዝርዝር",
            )
            if any(h in lower or h in instruction for h in expand_hints):
                return "expand"
            return "rewrite"

        def _resolve_from_plan(
            plan: dict[str, Any],
            *,
            allow_empty_targets: bool,
        ) -> tuple[list[dict[str, Any]], str, float]:
            targets = list(plan.get("targets") or [])
            scope = plan.get("scope") or "single"
            confidence = float(plan.get("confidence") or 0.0)
            thinness = plan.get("thinness") or "below_median"

            # Preserve heading targets for named sections (do not filter vs body structural ids)
            if plan.get("selection_mode") == "explicit_section" and targets:
                if len(targets) > 1:
                    scope = "multiple"
                    confidence = max(confidence, 0.75)
                return targets, scope, confidence

            if _needs_structural(plan):
                structural = structural_candidate_blocks(
                    all_blocks,
                    thinness=str(thinness),
                    max_targets=settings.EDIT_STRUCTURAL_MAX_TARGETS,
                )
                if plan.get("selection_mode") == "hybrid":
                    structural = merge_candidate_blocks(
                        structural,
                        ranked[: settings.EDIT_SELECTOR_TOP_K],
                    )
                structural_ids = {b["block_id"] for b in structural}
                action = (
                    "expand" if (plan.get("edit_intent") == "expand") else "rewrite"
                )

                def _structural_target(bid: str) -> dict[str, Any]:
                    return {
                        "block_id": bid,
                        "action": action,
                        "confidence": max(confidence, 0.65),
                        "reason": "structural_feature_match",
                    }

                if targets:
                    # Prefer LLM picks that fall inside the structural filter
                    kept = [
                        t
                        for t in targets
                        if t.get("block_id") and t["block_id"] in structural_ids
                    ]
                    if kept:
                        targets = kept
                    elif allow_empty_targets or confidence >= settings.EDIT_MIN_CONFIDENCE * 0.9:
                        targets = [_structural_target(b["block_id"]) for b in structural]
                    if plan.get("selection_mode") == "hybrid" and structural:
                        # Hybrid: union semantic LLM picks (existing) with structural filter
                        id_set = {t.get("block_id") for t in targets if t.get("block_id")}
                        for b in structural:
                            if b["block_id"] not in id_set:
                                targets.append(_structural_target(b["block_id"]))
                                id_set.add(b["block_id"])
                elif allow_empty_targets or confidence >= settings.EDIT_MIN_CONFIDENCE * 0.9:
                    targets = [_structural_target(b["block_id"]) for b in structural]
                if targets:
                    scope = "multiple" if len(targets) > 1 else scope
                    confidence = max(confidence, 0.65)
                metrics["stages"]["structural_candidates"] = len(structural)

            return targets, scope, confidence

        # Deterministic named-section resolve (before LLM) for multi-heading instructions
        named_headings = match_named_section_headings(
            instruction, skeleton.get("outline") or []
        )
        if len(named_headings) >= 2:
            action = _infer_edit_action()
            plan = _normalize_locate(
                {
                    "spatial_scope": "multi_section",
                    "selection_mode": "explicit_section",
                    "structural_intent": "named outline headings listed in the instruction",
                    "edit_intent": action if action in ("expand", "shorten", "rewrite") else "expand",
                    "targets": [
                        {
                            "block_id": h["block_id"],
                            "action": action,
                            "confidence": 0.9,
                            "reason": f"named_section:{h.get('title') or h['block_id']}",
                        }
                        for h in named_headings
                    ],
                    "scope": "multiple",
                    "confidence": 0.9,
                }
            )
            metrics["stages"]["named_sections"] = [
                {"block_id": h["block_id"], "title": h.get("title")} for h in named_headings
            ]
            metrics["stages"]["locate"] = plan
            return plan, list(plan["targets"]), "multiple", 0.9

        selector_k = settings.EDIT_SELECTOR_TOP_K
        locate_result = await edit_llm.complete_json(
            locate_system_prompt(),
            locate_user_prompt(
                instruction,
                ranked[:selector_k],
                skeleton=skeleton,
            ),
        )
        plan = _normalize_locate(locate_result)
        # Single named heading: seed explicit_section when LLM returned no targets
        if len(named_headings) == 1 and not (plan.get("targets") or []):
            action = _infer_edit_action(plan)
            plan["spatial_scope"] = "section"
            plan["selection_mode"] = "explicit_section"
            plan["scope"] = "section"
            h = named_headings[0]
            plan["targets"] = [
                {
                    "block_id": h["block_id"],
                    "action": action,
                    "confidence": 0.85,
                    "reason": f"named_section:{h.get('title') or h['block_id']}",
                }
            ]
            plan["confidence"] = max(float(plan.get("confidence") or 0), 0.8)
            metrics["stages"]["named_sections"] = [
                {"block_id": h["block_id"], "title": h.get("title")}
            ]
        metrics["stages"]["locate"] = plan

        # Enrich candidates for hybrid/structural before resolve
        if _needs_structural(plan):
            structural = structural_candidate_blocks(
                all_blocks,
                thinness=str(plan.get("thinness") or "below_median"),
                max_targets=settings.EDIT_STRUCTURAL_MAX_TARGETS,
            )
            ranked_for_pick = merge_candidate_blocks(structural, ranked[:selector_k])
            # Second pass only if first targets empty — avoid double LLM when targets present
            if not (plan.get("targets") or []):
                locate_retry = await edit_llm.complete_json(
                    locate_system_prompt(),
                    locate_user_prompt(
                        instruction,
                        ranked_for_pick[: max(selector_k, len(structural))],
                        skeleton=skeleton,
                    ),
                )
                if locate_retry:
                    plan = _normalize_locate({**plan, **locate_retry})
                    metrics["stages"]["locate_structural_pass"] = locate_retry

        targets, scope, locate_confidence = _resolve_from_plan(plan, allow_empty_targets=True)

        if targets and locate_confidence >= settings.EDIT_MIN_CONFIDENCE:
            return plan, targets, scope, locate_confidence

        # Soft recovery: structural plan with low LLM confidence still usable
        if _needs_structural(plan):
            targets, scope, locate_confidence = _resolve_from_plan(
                {**plan, "confidence": max(locate_confidence, 0.65)},
                allow_empty_targets=True,
            )
            if targets:
                metrics["stages"]["locate_recovery"] = "structural"
                plan["confidence"] = locate_confidence
                return plan, targets, scope, locate_confidence

        # Soft recovery: broaden semantic pool and retry once
        if locate_confidence < settings.EDIT_MIN_CONFIDENCE or not targets:
            retry_k = settings.EDIT_SELECTOR_RETRY_TOP_K
            broader = rank_blocks_for_instruction(
                instruction, all_blocks, top_k=retry_k
            )
            locate_retry = await edit_llm.complete_json(
                locate_system_prompt(),
                locate_user_prompt(
                    instruction,
                    broader[:retry_k],
                    skeleton=skeleton,
                ),
            )
            metrics["stages"]["locate_retry"] = locate_retry
            plan = _normalize_locate({**plan, **(locate_retry or {})})
            targets, scope, locate_confidence = _resolve_from_plan(
                plan, allow_empty_targets=_needs_structural(plan)
            )
            if targets and locate_confidence >= settings.EDIT_MIN_CONFIDENCE:
                metrics["stages"]["locate_recovery"] = "retry_broaden"
                return plan, targets, scope, locate_confidence
            if targets and _needs_structural(plan):
                metrics["stages"]["locate_recovery"] = "retry_structural"
                return plan, targets, scope, max(locate_confidence, 0.65)

        if locate_confidence < settings.EDIT_MIN_CONFIDENCE:
            return plan, [], scope, locate_confidence
        return plan, targets, scope, locate_confidence

    def _try_global_replace(
        self, instruction: str, blocks: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        match = _GLOBAL_REPLACE_RE.search(instruction)
        if not match:
            return None
        old_text, new_text = match.group(1).strip(), match.group(2).strip()
        if not old_text:
            return None
        ops: list[dict[str, Any]] = []
        for block in blocks:
            text = block.get("text") or ""
            if old_text in text:
                ops.append(
                    {
                        "op_id": uuid.uuid4().hex[:8],
                        "type": "replace",
                        "block_id": block["block_id"],
                        "payload": {"new_text": text.replace(old_text, new_text)},
                    }
                )
        return ops or None

    def _filter_valid_ops(
        self,
        operations: list[dict[str, Any]],
        doc_json: dict[str, Any],
        scope: str,
        target_ids: set[str],
    ) -> list[dict[str, Any]]:
        existing = {b["block_id"] for b in extract_blocks_from_tiptap(doc_json)}
        filtered: list[dict[str, Any]] = []
        for op in operations:
            bid = op.get("block_id")
            if not bid:
                continue
            if bid not in existing and not bid.startswith("auto_"):
                continue
            if scope != "global" and bid not in target_ids and op.get("type") != "insert":
                continue
            filtered.append(op)
        return filtered

    async def _finalize(
        self,
        doc_json: dict[str, Any],
        operations: list[dict[str, Any]],
        instruction: str,
        operation_id: str,
        metrics: dict[str, Any],
        started: float,
        output_language: str,
        patched_doc: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        patched = patched_doc if patched_doc is not None else apply_operations(doc_json, operations)

        before_blocks = extract_blocks_from_tiptap(doc_json)
        after_blocks = extract_blocks_from_tiptap(patched)
        changed_ids = {
            b["block_id"]
            for b in before_blocks
            if b["block_id"] not in {a["block_id"] for a in after_blocks}
            or next((a["text"] for a in after_blocks if a["block_id"] == b["block_id"]), None)
            != b["text"]
        }
        changed_ids.update(
            {b["block_id"] for b in after_blocks if b["block_id"] not in {x["block_id"] for x in before_blocks}}
        )

        before_text = "\n".join(
            f"{b['block_id']}: {b['text']}" for b in before_blocks if b["block_id"] in changed_ids
        )
        after_text = "\n".join(
            f"{b['block_id']}: {b['text']}" for b in after_blocks if b["block_id"] in changed_ids
        )

        semantic_diff: str | None = None
        try:
            semantic_diff = await edit_llm.complete_text(
                diff_system_prompt(),
                diff_user_prompt(before_text, after_text),
            )
        except Exception as exc:
            logger.warning("LLM diff generation failed, using simple diff: %s", exc)

        if not semantic_diff or semantic_diff.strip() in {
            "No changes detected",
            "No semantic changes detected",
        }:
            semantic_diff = compute_simple_diff(doc_json, patched)

        metrics["execution_time_ms"] = int((time.perf_counter() - started) * 1000)
        metrics["operations_count"] = len(operations)

        return {
            "document": patched,
            "diff": semantic_diff,
            "metrics": metrics,
            "operations": operations,
            "operation_id": operation_id,
            "no_effective_change": False,
            "no_effective_change_reason": None,
        }

    def _no_change_result(
        self,
        doc_json: dict[str, Any],
        operation_id: str,
        metrics: dict[str, Any],
        started: float,
        reason: str,
        operations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        metrics["execution_time_ms"] = int((time.perf_counter() - started) * 1000)
        return {
            "document": deepcopy(doc_json),
            "diff": reason,
            "metrics": metrics,
            "operations": operations or [],
            "operation_id": operation_id,
            "no_effective_change": True,
            "no_effective_change_reason": reason,
        }

    def _error_result(
        self,
        doc_json: dict[str, Any],
        operation_id: str,
        metrics: dict[str, Any],
        started: float,
        error: str,
    ) -> dict[str, Any]:
        metrics["execution_time_ms"] = int((time.perf_counter() - started) * 1000)
        return {
            "document": deepcopy(doc_json),
            "diff": f"Error occurred during processing: {error}. Document remains unchanged.",
            "metrics": metrics,
            "operations": [],
            "operation_id": operation_id,
            "no_effective_change": True,
            "no_effective_change_reason": error,
            "error": error,
        }
