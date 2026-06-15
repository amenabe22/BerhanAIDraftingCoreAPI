"""Deterministic content hashing for compliance cache and diff anchoring."""

from __future__ import annotations

import hashlib
import re


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def compute_block_hash(block_id: str, text: str) -> str:
    """SHA-256 of normalized block_id + text."""
    normalized = f"{block_id}|{_normalize_text(text)}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_per_block_hashes(blocks: list[dict]) -> dict[str, str]:
    """Map block_id -> content hash for each block with text."""
    out: dict[str, str] = {}
    for b in blocks or []:
        bid = str(b.get("block_id") or "").strip()
        if not bid:
            continue
        text = b.get("text") or ""
        out[bid] = compute_block_hash(bid, text)
    return out


def compute_document_content_hash(blocks: list[dict]) -> str:
    """SHA-256 over sorted block_id+text pairs (normalized)."""
    parts: list[str] = []
    for b in sorted(blocks or [], key=lambda x: str(x.get("block_id") or "")):
        bid = str(b.get("block_id") or "").strip()
        text = _normalize_text(b.get("text") or "")
        if bid or text:
            parts.append(f"{bid}|{text}")
    payload = "\n".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
