#!/usr/bin/env python3
"""Exit 0 if the MCP HTTP server is up. Auth 401 is healthy."""

from __future__ import annotations

import os
import urllib.error
import urllib.request

URL = f"http://127.0.0.1:{os.getenv('MCP_PORT', '8765')}/mcp"
_OK_WITHOUT_TOKEN = frozenset({401, 403, 405, 406})


def main() -> int:
    try:
        urllib.request.urlopen(URL, timeout=4)
    except urllib.error.HTTPError as exc:
        return 0 if exc.code in _OK_WITHOUT_TOKEN else 1
    except OSError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
