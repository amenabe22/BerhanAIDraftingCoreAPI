"""httpx client for CoreAPI one-shot POST /drafting/generate (SSE)."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urljoin

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 300.0
TERMINAL_TYPES = frozenset({"export_ready", "export_skipped", "error"})
DEFAULT_ROLES = ("Disclosing Party", "Receiving Party")
HTTP_ERROR_BODY_LIMIT = 500


def normalize_parties(parties: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Keep name+role only. Never copy ``type`` into role; fill defaults by index."""
    out: list[dict[str, str]] = []
    for i, raw in enumerate(parties or []):
        name = str(raw.get("name") or "").strip()
        role = str(raw.get("role") or "").strip()
        if not role:
            role = DEFAULT_ROLES[i] if i < len(DEFAULT_ROLES) else f"Party {i + 1}"
        out.append({"name": name, "role": role})
    return out


def build_generate_payload(
    *,
    doc_type: str,
    title: str,
    parties: list[dict[str, str]],
    governing_law: str = "Ethiopian law",
    language: str = "en",
    num_pages: int = 2,
    purpose: str | None = None,
    instructions: str | None = None,
    formats: list[str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "title": title,
        "parties": normalize_parties(parties),
        "governingLaw": governing_law,
        "numPages": num_pages,
    }
    if purpose:
        metadata["purpose"] = purpose
    body: dict[str, Any] = {
        "doc_type": doc_type,
        "type": list(formats or ["pdf", "docx"]),
        "language": language,
        "metadata": metadata,
    }
    if instructions:
        body["instructions"] = instructions
    return body


def parse_sse_events(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in raw.split("\n\n"):
        for line in block.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if not payload:
                continue
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
    return events


def result_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    thread_id = None
    for ev in events:
        et = ev.get("type")
        if et == "thread_id":
            thread_id = ev.get("thread_id") or thread_id
        elif et == "export_ready":
            return {
                "ok": True,
                "thread_id": ev.get("thread_id") or thread_id,
                "filename": ev.get("filename"),
                "pdf_url": ev.get("pdf_url"),
                "docx_url": ev.get("docx_url"),
            }
        elif et == "export_skipped":
            return {
                "ok": False,
                "error": "export_skipped",
                "message": ev.get("message") or "Export skipped",
                "thread_id": ev.get("thread_id") or thread_id,
            }
        elif et == "error":
            return {
                "ok": False,
                "error": "generation_failed",
                "message": ev.get("message") or "Document generation failed",
                "thread_id": ev.get("thread_id") or thread_id,
            }
    return {
        "ok": False,
        "error": "no_terminal_event",
        "message": "Stream ended without export_ready, export_skipped, or error",
        "thread_id": thread_id,
    }


def generate_document(
    payload: dict[str, Any],
    *,
    api_url: str = DEFAULT_API_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    url = urljoin(api_url.rstrip("/") + "/", "drafting/generate")
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream(
                "POST",
                url,
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                if resp.status_code >= 400:
                    body = ""
                    try:
                        raw = resp.read()
                        if isinstance(raw, (bytes, bytearray)):
                            body = raw.decode("utf-8", errors="replace")
                        elif raw:
                            body = str(raw)
                    except Exception:
                        body = ""
                    body = body.strip()[:HTTP_ERROR_BODY_LIMIT]
                    message = f"HTTP {resp.status_code}"
                    if body:
                        message = f"{message}: {body}"
                    return {
                        "ok": False,
                        "error": "http_error",
                        "message": message,
                    }
                buffer = ""
                events: list[dict[str, Any]] = []
                for chunk in resp.iter_text():
                    buffer += chunk
                    while "\n\n" in buffer:
                        part, buffer = buffer.split("\n\n", 1)
                        events.extend(parse_sse_events(part + "\n\n"))
                        if events and events[-1].get("type") in TERMINAL_TYPES:
                            return result_from_events(events)
                if buffer.strip():
                    events.extend(parse_sse_events(buffer))
                return result_from_events(events)
    except httpx.HTTPError as exc:
        return {"ok": False, "error": "http_error", "message": str(exc)}
