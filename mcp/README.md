# Berhan DocGen MCP

Tiny FastMCP HTTP server that wraps CoreAPI `POST /drafting/generate`, plus a small OpenRouter agent that calls it as a tool.

Does not change the product API. Install deps separately:

```bash
# from BerhanAIDraftingCoreAPI/
uv pip install -r mcp/requirements.txt
```

Copy [`mcp/.env.example`](.env.example) to `mcp/.env` and set `OPENROUTER_API_KEY` (or rely on the repo-root `.env`). Set the same `MCP_SHARED_SECRET` on server and client. CoreAPI also needs `OPENROUTER_API_KEY`. Contabo/S3 vars (`S3_*`) are required for real `pdf_url` / `docx_url`; otherwise the tool returns `export_skipped`.

> Note: this folder is named `mcp/` but is **not** a Python package (no `__init__.py`), so it does not shadow the MCP SDK package of the same name.

## Run locally (same machine)

Terminal 1 — CoreAPI:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 — MCP server (`http://127.0.0.1:8765/mcp`):

```bash
python mcp/server.py
```

Terminal 3 — OpenRouter agent:

```bash
python mcp/client_agent.py "Draft a 2-page NDA between Acme PLC and Beta LLC under Ethiopian law."
```

Success: stderr shows a `generate_legal_document(...)` tool call with `name` + `role` parties, then stdout quotes `pdf_url` / `docx_url` (or explains `export_skipped`).

## Remote PC (different network) via ngrok

Tunnel **only** MCP port `8765`. Do not tunnel CoreAPI `:8000`. ngrok 3 is already installed on this Mac; authtoken stays in `~/Library/Application Support/ngrok/ngrok.yml` (not in this repo).

### Host (this Mac)

Keep CoreAPI and the MCP server running, then start the named tunnel (user config first so the authtoken merges):

```bash
ngrok start \
  --config "$HOME/Library/Application Support/ngrok/ngrok.yml" \
  --config mcp/ngrok.yml \
  docgen-mcp
```

Equivalent fallback: `ngrok http 8765`.

Print the public MCP URL:

```bash
python mcp/print_mcp_url.py
```

Leave all three processes running. The free ngrok URL changes each time you restart ngrok.

Prove the public path **from this Mac first** (same route a remote PC will use):

```bash
MCP_URL=https://<sub>.ngrok-free.app/mcp \
python mcp/client_agent.py "Draft a 2-page NDA between Acme PLC and Beta LLC under Ethiopian law."
```

### Remote PC (other network)

Do **not** start CoreAPI or the MCP server on the remote PC. Copy `mcp/`, install deps, then:

```bash
uv pip install -r mcp/requirements.txt

MCP_URL=https://<sub>.ngrok-free.app/mcp \
MCP_SHARED_SECRET=<same-as-host> \
OPENROUTER_API_KEY=<remote-or-shared-key> \
python mcp/client_agent.py "Draft a 2-page NDA between Acme PLC and Beta LLC under Ethiopian law."
```

The client sends `Authorization: Bearer <MCP_SHARED_SECRET>` and `ngrok-skip-browser-warning: true`, uses HTTP/1.1 (ngrok HTTP/2 breaks MCP Streamable HTTP), and a 300s timeout so generation can finish through the tunnel.

Opening `https://<sub>.ngrok-free.app/mcp` in a browser returns **401**. That is expected; MCP is not a webpage.

## Tests (no live LLM)

```bash
python -m pytest mcp/test_docgen_client.py -q
```

## Production (docker compose)

The same `docker-compose.yml` that runs CoreAPI also runs MCP as a second container. MCP talks to CoreAPI on the internal Docker network (`http://api:8000`), not via localhost.

Put `MCP_SHARED_SECRET` in the repo-root `.env` (same file CoreAPI already uses). Do **not** keep the example `dev-docgen-secret` in production.

```bash
docker compose up -d --build
```

That starts:

| Service | Container     | Inside Docker        | Host (this compose file) |
| ------- | ------------- | -------------------- | ------------------------ |
| CoreAPI | `berhan-api`  | `http://api:8000`    | `http://127.0.0.1:8080`  |
| MCP     | `berhan-mcp`  | `http://mcp:8765/mcp`| `http://127.0.0.1:8765/mcp` |

Clients (Inspector, `mcp/client_agent.py`, Cursor) should use the **public HTTPS** URL you put on nginx, not the raw host port:

```
https://mcp.example.com/mcp
```

Point nginx at `berhan-mcp:8765` on `proxy-net` (same pattern as CoreAPI on `berhan-api:8000`). MCP already sends `Authorization: Bearer <MCP_SHARED_SECRET>`; keep that secret on both the server `.env` and every remote client.

Opening `/mcp` in a browser still returns **401**. That is expected.

Do not tunnel CoreAPI. Do not run ngrok in production — use your existing nginx/TLS stack.
