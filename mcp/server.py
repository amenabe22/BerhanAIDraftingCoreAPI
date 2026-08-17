#!/usr/bin/env python3
"""Tiny FastMCP HTTP server wrapping CoreAPI POST /drafting/generate."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

from docgen_client import build_generate_payload, generate_document

load_dotenv(_ROOT / ".env")
load_dotenv(_DIR / ".env", override=True)

DEFAULT_SECRET = "dev-docgen-secret"
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8765"))


def _shared_secret() -> str:
    return os.getenv("MCP_SHARED_SECRET") or DEFAULT_SECRET


def _api_url() -> str:
    return os.getenv("DOCGEN_API_URL", "http://127.0.0.1:8000").rstrip("/")


class PartyInput(BaseModel):
    """One contracting party. Only name and role — never a type field."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    role: str = Field(
        ...,
        min_length=1,
        description="Legal role, e.g. Disclosing Party or Receiving Party",
    )


def build_mcp() -> FastMCP:
    verifier = StaticTokenVerifier(
        tokens={
            _shared_secret(): {
                "client_id": "docgen-poc",
                "scopes": ["docgen"],
            }
        },
        required_scopes=["docgen"],
    )
    mcp = FastMCP("BerhanDocGen", auth=verifier)

    @mcp.tool
    def generate_legal_document(
        doc_type: str,
        title: str,
        parties: list[PartyInput],
        governing_law: str = "Ethiopian law",
        language: Literal["en", "am", "om"] = "en",
        num_pages: int = 2,
        purpose: str | None = None,
        instructions: str | None = None,
        formats: list[Literal["pdf", "docx"]] | None = None,
    ) -> dict[str, Any]:
        """Draft an Ethiopian legal document via CoreAPI and return download URLs.

        Call immediately when the user wants a document. Do not ask follow-up questions.
        Each party must be {name, role} only (e.g. Disclosing Party, Receiving Party).
        Never send a type field. Default purpose to protecting confidential information
        if unspecified. Returns pdf_url/docx_url on success, or a structured error.
        """
        payload = build_generate_payload(
            doc_type=doc_type,
            title=title,
            parties=[p.model_dump() for p in parties],
            governing_law=governing_law,
            language=language,
            num_pages=num_pages,
            purpose=purpose,
            instructions=instructions,
            formats=list(formats) if formats else None,
        )
        return generate_document(payload, api_url=_api_url())

    return mcp


mcp = build_mcp()


def main() -> None:
    print(
        f"BerhanDocGen MCP http://{MCP_HOST}:{MCP_PORT}/mcp "
        f"(CoreAPI {_api_url()})",
        file=sys.stderr,
    )
    mcp.run(transport="http", host=MCP_HOST, port=MCP_PORT)


if __name__ == "__main__":
    main()
