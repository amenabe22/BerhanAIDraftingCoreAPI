#!/usr/bin/env python3
"""Small OpenRouter agent that calls the BerhanDocGen MCP tool."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent

from dotenv import load_dotenv
import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from openai import OpenAI

load_dotenv(_ROOT / ".env")
load_dotenv(_DIR / ".env", override=True)

DEFAULT_MCP_URL = "http://127.0.0.1:8765/mcp"
DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_SECRET = "dev-docgen-secret"
MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = """You draft Ethiopian legal documents by calling generate_legal_document.

Rules:
- Call generate_legal_document immediately. Never ask follow-up questions.
- Parties must be objects with only name and role (e.g. Disclosing Party, Receiving Party). Never send a type field.
- If the user omits purpose, use "Protect confidential business information".
- If the user omits governing law, use Ethiopian law.
- Never invent pdf_url, docx_url, or file contents.
- After the tool returns, summarize the result and quote any URLs exactly.
- If the tool returns ok=false, explain that clearly using the tool message.
"""


def _secret() -> str:
    return os.getenv("MCP_SHARED_SECRET") or DEFAULT_SECRET


def _httpx_client_factory(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
    **kwargs: Any,
) -> httpx.AsyncClient:
    """HTTP/1.1 + bearer header — ngrok HTTP/2 and httpx Auth break MCP Streamable HTTP."""
    del auth, kwargs  # auth goes in headers; ignore unused transport kwargs
    merged = dict(headers or {})
    merged.setdefault("Authorization", f"Bearer {_secret()}")
    merged.setdefault("ngrok-skip-browser-warning", "true")
    if timeout is None:
        timeout = httpx.Timeout(30.0, read=300.0)
    return httpx.AsyncClient(
        headers=merged,
        timeout=timeout,
        http2=False,
        follow_redirects=True,
    )


def mcp_tools_to_openai(tools: list[Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        schema = tool.inputSchema if hasattr(tool, "inputSchema") else {}
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def serialize_tool_result(result: Any) -> str:
    data = getattr(result, "data", None)
    if data is not None:
        return json.dumps(data, default=str)
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return json.dumps(structured, default=str)
    content = getattr(result, "content", None)
    if content is not None:
        dumped = []
        for item in content:
            if hasattr(item, "model_dump"):
                dumped.append(item.model_dump())
            else:
                dumped.append(str(item))
        return json.dumps(dumped, default=str)
    return json.dumps(result, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenRouter agent that uses BerhanDocGen MCP")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=(
            "Draft a 2-page NDA between Acme PLC (Disclosing Party) and "
            "Beta LLC (Receiving Party) under Ethiopian law."
        ),
    )
    parser.add_argument("--mcp-url", default=os.getenv("MCP_URL", DEFAULT_MCP_URL))
    parser.add_argument("--model", default=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL))
    return parser.parse_args()


async def run_agent(prompt: str, mcp_url: str, model: str) -> int:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 1

    llm = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    last_err: Exception | None = None

    for attempt in range(1, 4):
        transport = StreamableHttpTransport(
            url=mcp_url,
            headers={
                "ngrok-skip-browser-warning": "true",
                "Authorization": f"Bearer {_secret()}",
            },
            httpx_client_factory=_httpx_client_factory,
        )
        try:
            async with Client(transport, timeout=300) as mcp:
                mcp_tools = await mcp.list_tools()
                if not mcp_tools:
                    print("MCP server returned no tools", file=sys.stderr)
                    return 1
                openai_tools = mcp_tools_to_openai(mcp_tools)
                print(f"MCP tools: {[t.name for t in mcp_tools]}", file=sys.stderr)

                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]

                for round_i in range(MAX_TOOL_ROUNDS):
                    resp = llm.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="required" if round_i == 0 else "auto",
                    )
                    choice = resp.choices[0].message
                    assistant: dict[str, Any] = {
                        "role": "assistant",
                        "content": choice.content or "",
                    }
                    if choice.tool_calls:
                        assistant["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.tool_calls
                        ]
                    messages.append(assistant)

                    if not choice.tool_calls:
                        print(choice.content or "")
                        return 0

                    for tc in choice.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        print(
                            f"[round {round_i + 1}] {tc.function.name}({json.dumps(args)})",
                            file=sys.stderr,
                        )
                        result = await mcp.call_tool(tc.function.name, args)
                        payload = serialize_tool_result(result)
                        print(f"[tool result] {payload[:800]}", file=sys.stderr)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": payload,
                            }
                        )

                print("Stopped after max tool rounds without a final answer", file=sys.stderr)
                return 1
        except Exception as exc:
            last_err = exc
            print(
                f"MCP session failed (attempt {attempt}/3): {exc}",
                file=sys.stderr,
            )
            await asyncio.sleep(1.5 * attempt)

    print(f"Giving up after MCP retries: {last_err}", file=sys.stderr)
    return 1


def main() -> int:
    args = parse_args()
    return asyncio.run(run_agent(args.prompt, args.mcp_url, args.model))


if __name__ == "__main__":
    raise SystemExit(main())
