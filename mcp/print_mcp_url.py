#!/usr/bin/env python3
"""Print the public MCP URL from a running ngrok inspect API."""

from __future__ import annotations

import json
import sys
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

NGROK_INSPECT = "http://127.0.0.1:4040/api/tunnels"
DEFAULT_PROMPT = (
    "Draft a 2-page NDA between Acme PLC and Beta LLC under Ethiopian law."
)


def public_mcp_url(payload: dict[str, Any]) -> str | None:
    https = None
    http = None
    for tunnel in payload.get("tunnels") or []:
        pub = str(tunnel.get("public_url") or "").rstrip("/")
        if not pub:
            continue
        mcp = pub + "/mcp"
        if pub.startswith("https://") and https is None:
            https = mcp
        elif pub.startswith("http://") and http is None:
            http = mcp
    return https or http


def fetch_tunnels(inspect_url: str = NGROK_INSPECT) -> dict[str, Any]:
    with urlopen(inspect_url, timeout=3) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    try:
        payload = fetch_tunnels()
    except URLError as exc:
        print(
            "ngrok inspect API is not reachable at "
            f"{NGROK_INSPECT} ({exc}). Start the tunnel first.",
            file=sys.stderr,
        )
        return 1
    except json.JSONDecodeError:
        print("ngrok inspect API returned invalid JSON", file=sys.stderr)
        return 1

    url = public_mcp_url(payload)
    if not url:
        print("No ngrok public_url found. Is the docgen-mcp tunnel up?", file=sys.stderr)
        return 1

    print(url)
    print(file=sys.stderr)
    print("Remote PC:", file=sys.stderr)
    print(
        f'MCP_URL={url} MCP_SHARED_SECRET=<same-as-host> \\\n'
        f'python mcp/client_agent.py "{DEFAULT_PROMPT}"',
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
