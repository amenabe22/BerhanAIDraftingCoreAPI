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
from app.services.drafting.editing.retrieval import neighbor_blocks, rank_blocks_for_instruction
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
        if output_language not in ("en", "am"):
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

            # LOCATE: rerank + LLM selector
            t0 = time.perf_counter()
            ranked = rank_blocks_for_instruction(instruction, all_blocks)
            metrics["stages"]["rerank_ms"] = int((time.perf_counter() - t0) * 1000)

            locate_result = await edit_llm.complete_json(
                locate_system_prompt(),
                locate_user_prompt(instruction, ranked[: settings.EDIT_SELECTOR_TOP_K]),
            )
            metrics["stages"]["locate"] = locate_result

            targets = (locate_result or {}).get("targets") or []
            scope = (locate_result or {}).get("scope", "single")
            locate_confidence = float((locate_result or {}).get("confidence") or 0.0)

            if not targets or locate_confidence < settings.EDIT_MIN_CONFIDENCE:
                return self._no_change_result(
                    doc_json,
                    operation_id,
                    metrics,
                    started,
                    "Could not confidently locate target block(s) for the instruction.",
                )

            target_ids = {t["block_id"] for t in targets if t.get("block_id")}
            edit_context = neighbor_blocks(all_blocks, target_ids, window=1)
            if scope == "global":
                edit_context = [
                    b
                    for b in all_blocks
                    if any(
                        tok in (b.get("text") or "").lower()
                        for tok in instruction.lower().split()
                        if len(tok) > 3
                    )
                ] or ranked[:10]

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
            logger.exception("Semantic edit failed op=%s", operation_id)
            return self._error_result(doc_json, operation_id, metrics, started, str(exc))

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
