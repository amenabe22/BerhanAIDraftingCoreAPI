"""Contabo Object Storage client (S3-compatible via boto3)."""

from __future__ import annotations

from functools import lru_cache

import boto3
from botocore.client import BaseClient
from botocore.config import Config

from app.config import settings
from app.logging_config import get_logger

log = get_logger("contabo")


class ContaboNotConfiguredError(RuntimeError):
    """Raised when Contabo/S3 credentials are missing."""


def contabo_configured() -> bool:
    return bool(
        settings.S3_ENDPOINT_URL
        and settings.S3_ACCESS_KEY_ID
        and settings.S3_SECRET_ACCESS_KEY
        and settings.S3_BUCKET_NAME
    )


@lru_cache(maxsize=1)
def _s3_client() -> BaseClient:
    if not contabo_configured():
        raise ContaboNotConfiguredError(
            "Contabo/S3 is not configured. Set S3_ENDPOINT_URL, "
            "S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, and S3_BUCKET_NAME."
        )
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL.rstrip("/"),
        aws_access_key_id=settings.S3_ACCESS_KEY_ID,
        aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
        region_name=settings.S3_REGION or "default",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )


def upload_bytes(key: str, data: bytes, content_type: str) -> str:
    """Upload bytes to Contabo and return a downloadable URL."""
    client = _s3_client()
    bucket = settings.S3_BUCKET_NAME
    log.info(
        "contabo_upload",
        extra={"event": "contabo_upload", "bucket": bucket, "key": key, "bytes": len(data)},
    )
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )

    public_base = (settings.S3_PUBLIC_BASE_URL or "").rstrip("/")
    if public_base:
        return f"{public_base}/{key.lstrip('/')}"

    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=settings.S3_PRESIGN_EXPIRY_SECONDS,
    )


def reset_client_cache() -> None:
    """Clear cached client (for tests)."""
    _s3_client.cache_clear()
