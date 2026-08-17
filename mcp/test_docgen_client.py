"""Unit tests for the Doc-Gen SSE client (no live LLM / CoreAPI)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from docgen_client import (
    build_generate_payload,
    generate_document,
    normalize_parties,
    parse_sse_events,
    result_from_events,
)
from print_mcp_url import public_mcp_url


def _sse(*events: dict) -> str:
    return "".join(f"data: {json.dumps(ev)}\n\n" for ev in events)


def test_parse_sse_events_extracts_json_payloads():
    raw = _sse(
        {"type": "status", "message": "Generating…"},
        {"type": "export_ready", "pdf_url": "https://cdn.test/nda.pdf"},
    )
    events = parse_sse_events(raw)
    assert [e["type"] for e in events] == ["status", "export_ready"]
    assert events[1]["pdf_url"] == "https://cdn.test/nda.pdf"


def test_result_from_export_ready():
    result = result_from_events(
        [
            {"type": "thread_id", "thread_id": "t1"},
            {
                "type": "export_ready",
                "thread_id": "t1",
                "filename": "nda",
                "pdf_url": "https://cdn.test/nda.pdf",
                "docx_url": "https://cdn.test/nda.docx",
            },
        ]
    )
    assert result == {
        "ok": True,
        "thread_id": "t1",
        "filename": "nda",
        "pdf_url": "https://cdn.test/nda.pdf",
        "docx_url": "https://cdn.test/nda.docx",
    }


def test_result_from_export_skipped():
    result = result_from_events(
        [
            {"type": "document_generated", "thread_id": "t2", "document": {}},
            {"type": "export_skipped", "thread_id": "t2", "message": "Contabo/S3 is not configured"},
        ]
    )
    assert result["ok"] is False
    assert result["error"] == "export_skipped"
    assert "Contabo" in result["message"]
    assert result["thread_id"] == "t2"


def test_result_from_error_event():
    result = result_from_events([{"type": "error", "message": "boom"}])
    assert result == {
        "ok": False,
        "error": "generation_failed",
        "message": "boom",
        "thread_id": None,
    }


def test_result_from_empty_stream():
    result = result_from_events([])
    assert result["ok"] is False
    assert result["error"] == "no_terminal_event"


def test_build_generate_payload_matches_coreapi_schema():
    body = build_generate_payload(
        doc_type="NDA",
        title="Mutual NDA",
        parties=[{"name": "Acme PLC", "role": "Disclosing Party"}],
        governing_law="Ethiopian law",
        language="en",
        num_pages=2,
        purpose="Keep secrets",
        instructions="Keep it practical",
        formats=["pdf", "docx"],
    )
    assert body["doc_type"] == "NDA"
    assert body["type"] == ["pdf", "docx"]
    assert body["language"] == "en"
    assert body["instructions"] == "Keep it practical"
    assert body["metadata"] == {
        "title": "Mutual NDA",
        "parties": [{"name": "Acme PLC", "role": "Disclosing Party"}],
        "governingLaw": "Ethiopian law",
        "numPages": 2,
        "purpose": "Keep secrets",
    }


def test_normalize_parties_strips_type_and_fills_roles():
    out = normalize_parties(
        [
            {"type": "Company", "name": "Acme PLC"},
            {"type": "Company", "name": "Beta LLC"},
        ]
    )
    assert out == [
        {"name": "Acme PLC", "role": "Disclosing Party"},
        {"name": "Beta LLC", "role": "Receiving Party"},
    ]


def test_normalize_parties_keeps_explicit_roles():
    out = normalize_parties(
        [{"name": "Acme PLC", "role": "Employer", "type": "Company"}]
    )
    assert out == [{"name": "Acme PLC", "role": "Employer"}]


def test_build_generate_payload_normalizes_parties():
    body = build_generate_payload(
        doc_type="NDA",
        title="Mutual NDA",
        parties=[{"type": "Company", "name": "Acme PLC"}],
    )
    assert body["metadata"]["parties"] == [
        {"name": "Acme PLC", "role": "Disclosing Party"}
    ]


def test_build_generate_payload_omits_empty_optionals():
    body = build_generate_payload(
        doc_type="NDA",
        title="Mutual NDA",
        parties=[{"name": "Acme PLC", "role": "Disclosing Party"}],
    )
    assert "instructions" not in body
    assert "purpose" not in body["metadata"]
    assert body["metadata"]["governingLaw"] == "Ethiopian law"
    assert body["type"] == ["pdf", "docx"]
    assert body["language"] == "en"
    assert body["metadata"]["numPages"] == 2


class _FakeStream:
    def __init__(self, status_code: int, text: str, headers: dict | None = None):
        self.status_code = status_code
        self._text = text
        self.headers = headers or {"content-type": "text/event-stream"}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("POST", "http://test/drafting/generate"),
                response=httpx.Response(self.status_code),
            )

    def read(self):
        return self._text.encode() if isinstance(self._text, str) else self._text

    def iter_text(self):
        yield self._text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_generate_document_returns_export_ready():
    sse = _sse(
        {"type": "thread_id", "thread_id": "abc"},
        {
            "type": "export_ready",
            "thread_id": "abc",
            "filename": "nda",
            "pdf_url": "https://cdn.test/nda.pdf",
            "docx_url": None,
        },
    )
    fake_client = MagicMock()
    fake_client.stream.return_value = _FakeStream(200, sse)

    with patch("docgen_client.httpx.Client", return_value=fake_client):
        fake_client.__enter__.return_value = fake_client
        fake_client.__exit__.return_value = False
        result = generate_document(
            {
                "doc_type": "NDA",
                "type": ["pdf"],
                "language": "en",
                "metadata": {
                    "title": "NDA",
                    "parties": [{"name": "A", "role": "Party"}],
                    "governingLaw": "Ethiopian law",
                },
            },
            api_url="http://127.0.0.1:8000",
        )

    assert result["ok"] is True
    assert result["pdf_url"] == "https://cdn.test/nda.pdf"
    assert result["thread_id"] == "abc"
    fake_client.stream.assert_called_once()
    args, kwargs = fake_client.stream.call_args
    assert args[0] == "POST"
    assert args[1] == "http://127.0.0.1:8000/drafting/generate"


def test_generate_document_http_error():
    fake_client = MagicMock()
    fake_client.stream.return_value = _FakeStream(500, "nope")
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = False

    with patch("docgen_client.httpx.Client", return_value=fake_client):
        result = generate_document({"doc_type": "NDA"}, api_url="http://127.0.0.1:8000")

    assert result["ok"] is False
    assert result["error"] == "http_error"
    assert "HTTP 500" in result["message"]
    assert "nope" in result["message"]


def test_generate_document_http_422_includes_body():
    fake_client = MagicMock()
    fake_client.stream.return_value = _FakeStream(
        422, '{"detail":[{"loc":["body","metadata","parties",0,"role"],"msg":"Field required"}]}'
    )
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = False

    with patch("docgen_client.httpx.Client", return_value=fake_client):
        result = generate_document({"doc_type": "NDA"}, api_url="http://127.0.0.1:8000")

    assert result["ok"] is False
    assert result["error"] == "http_error"
    assert "HTTP 422" in result["message"]
    assert "role" in result["message"]


def test_public_mcp_url_prefers_https():
    url = public_mcp_url(
        {
            "tunnels": [
                {"public_url": "http://abc.ngrok-free.app"},
                {"public_url": "https://abc.ngrok-free.app"},
            ]
        }
    )
    assert url == "https://abc.ngrok-free.app/mcp"


def test_public_mcp_url_empty():
    assert public_mcp_url({"tunnels": []}) is None
