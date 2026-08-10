"""Tests for structured one-shot Doc-Gen payload API."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from jsonschema import ValidationError

from app.main import app
from app.models.drafting.generate import StructuredGenerateRequest
from app.services.export.contabo import ContaboNotConfiguredError
from app.services.generation.metadata_schema import (
    load_drafting_metadata_schema,
    validate_drafting_metadata,
)
from app.services.generation.requirements import (
    build_requirements,
    build_synthetic_prompt,
)

SAMPLE_TIPTAP = {
    "type": "doc",
    "content": [
        {
            "type": "page",
            "content": [
                {
                    "type": "heading",
                    "attrs": {"level": 1},
                    "content": [{"type": "text", "text": "NDA"}],
                },
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "Confidentiality terms."}],
                },
            ],
        }
    ],
}

VALID_METADATA = {
    "title": "Mutual Non-Disclosure Agreement",
    "parties": [
        {"name": "Acme PLC", "role": "Disclosing Party"},
        {"name": "Beta LLC", "role": "Receiving Party"},
    ],
    "governingLaw": "Ethiopian law",
    "jurisdiction": "Ethiopian courts",
    "numPages": 2,
    "purpose": "Protect confidential business information",
}


@pytest.fixture
def client():
    return TestClient(app)


def test_structured_route_in_openapi(client: TestClient):
    paths = client.app.openapi()["paths"]
    assert "/drafting/generate" in paths
    assert "/drafting/generate/stream" in paths
    assert "/drafting/generate/metadata-schema" in paths


def test_metadata_schema_endpoint(client: TestClient):
    r = client.get("/drafting/generate/metadata-schema")
    assert r.status_code == 200
    body = r.json()
    assert body["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "title" in body["required"]
    assert "parties" in body["required"]
    assert "governingLaw" in body["required"]


def test_json_schema_file_validates_canonical_example():
    validate_drafting_metadata(VALID_METADATA)
    schema = load_drafting_metadata_schema()
    assert schema["properties"]["parties"]["items"]["required"] == ["name", "role"]


def test_json_schema_rejects_string_parties():
    bad = {
        **VALID_METADATA,
        "parties": ["Acme", "Beta"],
    }
    with pytest.raises(ValidationError):
        validate_drafting_metadata(bad)


def test_structured_requires_doc_type(client: TestClient):
    r = client.post("/drafting/generate", json={"type": ["pdf"], "metadata": VALID_METADATA})
    assert r.status_code == 422


def test_structured_requires_metadata(client: TestClient):
    r = client.post("/drafting/generate", json={"doc_type": "NDA", "type": ["pdf"]})
    assert r.status_code == 422


def test_structured_requires_type_formats(client: TestClient):
    r = client.post(
        "/drafting/generate",
        json={"doc_type": "NDA", "type": [], "metadata": VALID_METADATA},
    )
    assert r.status_code == 422


def test_structured_rejects_bad_format(client: TestClient):
    r = client.post(
        "/drafting/generate",
        json={"doc_type": "NDA", "type": ["rtf"], "metadata": VALID_METADATA},
    )
    assert r.status_code == 422


def test_structured_rejects_party_without_role(client: TestClient):
    r = client.post(
        "/drafting/generate",
        json={
            "doc_type": "NDA",
            "type": ["pdf"],
            "metadata": {
                "title": "NDA",
                "governingLaw": "Ethiopian law",
                "parties": [{"name": "Acme"}],
            },
        },
    )
    assert r.status_code == 422


def test_structured_rejects_snake_case_only_when_missing_required_camel(client: TestClient):
    """Missing camelCase required keys (e.g. governingLaw) → 422."""
    r = client.post(
        "/drafting/generate",
        json={
            "doc_type": "NDA",
            "type": ["pdf"],
            "metadata": {
                "title": "NDA",
                "parties": [{"name": "Acme", "role": "Discloser"}],
                # missing governingLaw
            },
        },
    )
    assert r.status_code == 422


def test_build_requirements_maps_payload():
    req = StructuredGenerateRequest(
        doc_type="Employment Contract",
        type=["pdf", "docx"],
        language="am",
        instructions="Include probation clause",
        metadata={
            "title": "Employment Agreement",
            "parties": [
                {"name": "Acme", "role": "Employer"},
                {"name": "Abebe Kebede", "role": "Employee"},
            ],
            "governingLaw": "Ethiopian law",
            "numPages": 3,
        },
    )
    reqs = build_requirements(req)
    assert reqs["document_type"] == "Employment Contract"
    assert reqs["language"] == "am"
    assert reqs["num_pages"] == 3
    assert reqs["title"] == "Employment Agreement"
    assert "Acme" in reqs["parties"]
    assert "Abebe Kebede" in reqs["parties"]
    assert reqs["governing_law"] == "Ethiopian law"
    assert "probation" in reqs["instructions"].lower()

    prompt = build_synthetic_prompt(req, reqs)
    assert "Employment Contract" in prompt
    assert "Do not ask clarifying questions" in prompt


def test_structured_generate_passes_metadata_json_and_language(client: TestClient):
    """API accepts nested metadata JSON objects; language preference is pinned."""
    captured: dict = {}

    async def fake_gen(thread_id, emitter, **kwargs):
        captured["requirements"] = kwargs.get("requirements")
        captured["synthetic_prompt"] = kwargs.get("synthetic_prompt")
        await emitter.document_generated(thread_id, SAMPLE_TIPTAP, metadata={})
        return dict(SAMPLE_TIPTAP)

    with (
        patch(
            "app.api.v1.endpoints.drafting.generate._agent.generate_from_requirements",
            new=AsyncMock(side_effect=fake_gen),
        ),
        patch(
            "app.api.v1.endpoints.drafting.generate.export_document",
            return_value={
                "filename": "nda",
                "pdf_url": "https://cdn.test/nda.pdf",
                "docx_url": None,
                "keys": {"pdf": "k.pdf"},
            },
        ),
    ):
        r = client.post(
            "/drafting/generate",
            json={
                "doc_type": "NDA",
                "type": ["pdf"],
                "language": "am",
                "instructions": "Keep practical",
                "metadata": VALID_METADATA,
            },
        )

    assert r.status_code == 200
    reqs = captured["requirements"]
    assert reqs["language"] == "am"
    assert reqs["document_type"] == "NDA"
    assert reqs["num_pages"] == 2
    assert reqs["title"] == "Mutual Non-Disclosure Agreement"
    assert "Acme PLC" in reqs["parties"]
    assert "Disclosing Party" in reqs["parties"]
    assert "Beta LLC" in reqs["parties"]
    assert "Amharic" in captured["synthetic_prompt"]
    assert 'MUST be "am"' in captured["synthetic_prompt"]
    assert "clarification_needed" not in r.text


def test_structured_rejects_non_object_metadata(client: TestClient):
    r = client.post(
        "/drafting/generate",
        json={"doc_type": "NDA", "type": ["pdf"], "metadata": "parties as text"},
    )
    assert r.status_code == 422


def test_structured_accepts_amh_language_alias(client: TestClient):
    captured: dict = {}

    async def fake_gen(thread_id, emitter, **kwargs):
        captured["requirements"] = kwargs.get("requirements")
        await emitter.document_generated(thread_id, SAMPLE_TIPTAP, metadata={})
        return dict(SAMPLE_TIPTAP)

    with (
        patch(
            "app.api.v1.endpoints.drafting.generate._agent.generate_from_requirements",
            new=AsyncMock(side_effect=fake_gen),
        ),
        patch(
            "app.api.v1.endpoints.drafting.generate.export_document",
            return_value={"filename": "x", "pdf_url": "https://cdn.test/x.pdf", "keys": {}},
        ),
    ):
        r = client.post(
            "/drafting/generate",
            json={
                "doc_type": "NDA",
                "type": ["pdf"],
                "language": "amh",
                "metadata": VALID_METADATA,
            },
        )

    assert r.status_code == 200
    assert captured["requirements"]["language"] == "am"


def test_structured_export_skipped_when_contabo_missing(client: TestClient):
    async def fake_gen(thread_id, emitter, **kwargs):
        await emitter.document_generated(thread_id, SAMPLE_TIPTAP, metadata={})
        return dict(SAMPLE_TIPTAP)

    with (
        patch(
            "app.api.v1.endpoints.drafting.generate._agent.generate_from_requirements",
            new=AsyncMock(side_effect=fake_gen),
        ),
        patch(
            "app.api.v1.endpoints.drafting.generate.export_document",
            side_effect=ContaboNotConfiguredError("Contabo/S3 is not configured"),
        ),
    ):
        r = client.post(
            "/drafting/generate",
            json={"doc_type": "NDA", "type": ["pdf"], "metadata": VALID_METADATA},
        )

    assert r.status_code == 200
    assert "document_generated" in r.text
    assert "export_skipped" in r.text
