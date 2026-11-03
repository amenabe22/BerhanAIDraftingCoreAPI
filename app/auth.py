"""Service-to-service auth for the Knowledge / search-api.

CoreAPI must send ``Authorization: Bearer <BERHAN_SEARCH_API_KEY>`` (or
``X-Berhan-Service-Key``). When the key is unset, only localhost is allowed
(local development). In production the key must be set.
"""

from __future__ import annotations

import hmac
import os
import secrets
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

# Paths that stay public (probes / CORS preflight).
_PUBLIC_PATHS = frozenset({"/health", "/ready", "/favicon.ico"})


def _expected_key() -> str:
    # Prefer process env (docker compose env_file) then Settings — Settings may
    # miss fields if an older image/config is mixed during rolling deploys.
    return (
        (os.getenv("BERHAN_SEARCH_API_KEY") or "").strip()
        or (getattr(settings, "BERHAN_SEARCH_API_KEY", None) or "").strip()
    )


def _extract_key(request: Request) -> str:
    auth = (request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (request.headers.get("X-Berhan-Service-Key") or "").strip()


def _is_loopback(request: Request) -> bool:
    host = ""
    if request.client:
        host = request.client.host or ""
    return host in {"127.0.0.1", "::1", "localhost"}


class ServiceAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path.rstrip("/") or "/"
        if path in _PUBLIC_PATHS or path == "/":
            # Root is handled separately (no HTML test client in prod).
            if path == "/" and request.method == "GET":
                return await call_next(request)
            if path in _PUBLIC_PATHS:
                return await call_next(request)

        expected = _expected_key()
        provided = _extract_key(request)

        env = (os.getenv("ENVIRONMENT") or getattr(settings, "ENVIRONMENT", "") or "development").lower()
        if not expected:
            if env in {"production", "prod"}:
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Service authentication is not configured"},
                )
            # Fail open only for local loopback when no key configured (dev).
            if _is_loopback(request) and env in {
                "development",
                "dev",
                "local",
                "",
            }:
                return await call_next(request)
            return JSONResponse(
                status_code=503,
                content={"detail": "Service authentication is not configured"},
            )

        if not provided or not hmac.compare_digest(provided, expected):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
            )

        return await call_next(request)


def require_mcp_secret() -> str:
    """Fail closed for MCP shared secret outside explicit local/dev."""
    secret = (os.getenv("MCP_SHARED_SECRET") or "").strip()
    env = (os.getenv("ENVIRONMENT") or "development").lower()
    if secret and secret != "dev-docgen-secret":
        return secret
    if env in {"development", "dev", "local"} and secrets.compare_digest(
        secret or "dev-docgen-secret", "dev-docgen-secret"
    ):
        return secret or "dev-docgen-secret"
    if not secret or secret == "dev-docgen-secret":
        raise RuntimeError(
            "MCP_SHARED_SECRET must be set to a non-default value in this environment"
        )
    return secret
