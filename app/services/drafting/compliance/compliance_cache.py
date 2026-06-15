"""Compliance analysis cache keys and diff-anchoring persistence."""

from __future__ import annotations

from app.config import settings
from app.services.cache import cache_get, cache_set
from app.services.drafting.compliance.rubric import RUBRIC_VERSION


def compliance_cache_key(
    *,
    content_hash: str,
    check_level: str,
    language: str,
    rubric_version: str = RUBRIC_VERSION,
) -> str:
    return f"compliance:v{rubric_version}:{check_level}:{language}:{content_hash}"


def last_result_key(doc_id: str) -> str:
    return f"compliance:last:{doc_id}"


def get_cached_compliance(cache_key: str) -> dict | None:
    return cache_get(cache_key)


def store_cached_compliance(cache_key: str, response: dict) -> bool:
    return cache_set(cache_key, response, ttl=settings.COMPLIANCE_CACHE_TTL)


def get_last_rubric_result(doc_id: str) -> dict | None:
    return cache_get(last_result_key(doc_id))


def store_last_rubric_result(
    doc_id: str,
    *,
    content_hash: str,
    per_block_hashes: dict[str, str],
    checks: list[dict],
) -> bool:
    return cache_set(
        last_result_key(doc_id),
        {
            "content_hash": content_hash,
            "per_block_hashes": per_block_hashes,
            "checks": checks,
            "rubric_version": RUBRIC_VERSION,
        },
        ttl=settings.COMPLIANCE_CACHE_TTL,
    )
