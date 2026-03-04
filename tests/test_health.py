"""
Tests for GET / and GET /health endpoints.
"""

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_health_returns_ok():
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_root_returns_html_when_file_exists(tmp_path, monkeypatch):
    """GET / should serve the HTML test client when it exists on disk."""
    html = tmp_path / "test_client.html"
    html.write_text("<html>test</html>")

    import app.main as main_module

    monkeypatch.setattr(main_module, "_TEST_CLIENT_PATH", html)

    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/")
    assert r.status_code == 200
    assert "html" in r.headers["content-type"]


@pytest.mark.asyncio
async def test_root_returns_json_when_file_missing(tmp_path, monkeypatch):
    """GET / should return JSON API info when test_client.html doesn't exist."""
    missing = tmp_path / "nonexistent.html"

    import app.main as main_module

    monkeypatch.setattr(main_module, "_TEST_CLIENT_PATH", missing)

    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "legal_search_stream" in body
    assert "legal_agent_stream" in body
