"""Lazy Redis client for compliance caching and diff anchoring."""

from __future__ import annotations

import json
from typing import Any

from app.config import settings
from app.logging_config import get_logger

log = get_logger("cache")

_redis_client: Any | None = None
_redis_unavailable: bool = False


def get_redis():
    """Return a Redis client or None if unavailable."""
    global _redis_client, _redis_unavailable
    if _redis_unavailable:
        return None
    if _redis_client is not None:
        return _redis_client
    if not settings.REDIS_URL:
        _redis_unavailable = True
        return None
    try:
        import redis

        _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        _redis_client.ping()
        return _redis_client
    except Exception as e:
        log.warning("redis_unavailable", extra={"event": "redis_unavailable", "error": str(e)})
        _redis_unavailable = True
        return None


def cache_get(key: str) -> dict | None:
    """Fetch and parse JSON value from Redis."""
    client = get_redis()
    if client is None:
        return None
    try:
        raw = client.get(key)
        if not raw:
            return None
        return json.loads(raw)
    except Exception as e:
        log.warning("cache_get_error", extra={"key": key, "error": str(e)})
        return None


def cache_set(key: str, value: dict, *, ttl: int | None = None) -> bool:
    """Store JSON-serializable value in Redis."""
    client = get_redis()
    if client is None:
        return False
    try:
        payload = json.dumps(value, default=str)
        if ttl is not None:
            client.setex(key, ttl, payload)
        else:
            client.set(key, payload)
        return True
    except Exception as e:
        log.warning("cache_set_error", extra={"key": key, "error": str(e)})
        return False
