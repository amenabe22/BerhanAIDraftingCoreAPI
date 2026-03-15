"""
Compares real-time streaming behavior between local and production endpoints.
Measures time-to-first-token, inter-chunk intervals, and total chunks received.
"""

import asyncio
import time
import httpx

LOCAL = "http://127.0.0.1:8001/legal-search/stream"
PROD  = "https://search-api.berhan.ai/legal-search/stream"
PAYLOAD = {
    "message": "What are the steps to form a legal contract?",
    "language": None,
}


async def stream_endpoint(label: str, url: str, results: dict):
    chunks = []
    first_token_time = None
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=PAYLOAD) as resp:
                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    now = time.perf_counter()
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    elapsed = now - start

                    import json
                    try:
                        parsed = json.loads(data)
                    except Exception:
                        continue

                    event_type = parsed.get("type", "?")

                    if event_type == "token":
                        content = parsed.get("content", "")
                        if first_token_time is None:
                            first_token_time = elapsed
                            print(f"\n[{label}] ⚡ FIRST TOKEN at {elapsed:.3f}s — {repr(content[:60])}")
                        chunks.append((elapsed, len(content)))
                        # Show each chunk arrival with timing
                        print(f"[{label}] chunk #{len(chunks):>3} | t={elapsed:.3f}s | len={len(content):>5} chars | {repr(content[:50])}")

                    elif event_type == "thread_id":
                        print(f"[{label}] thread_id received at {elapsed:.3f}s")
                    elif event_type == "status":
                        print(f"[{label}] status='{parsed.get('message')}' at {elapsed:.3f}s")
                    elif event_type == "citations":
                        print(f"[{label}] citations received at {elapsed:.3f}s ({len(parsed.get('citations', []))} citations)")
                    elif event_type == "error":
                        print(f"[{label}] ERROR: {parsed.get('message')}")

    except Exception as e:
        print(f"[{label}] EXCEPTION: {e}")

    total = time.perf_counter() - start
    results[label] = {
        "total_time": total,
        "first_token_time": first_token_time,
        "num_chunks": len(chunks),
        "chunk_sizes": [c[1] for c in chunks],
        "chunk_times": [c[0] for c in chunks],
    }


def summarize(label: str, r: dict):
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {label}")
    print(f"{'='*60}")
    if r["first_token_time"] is None:
        print("  No token chunks received!")
        return

    print(f"  Total time:          {r['total_time']:.3f}s")
    print(f"  Time to first token: {r['first_token_time']:.3f}s")
    print(f"  Number of chunks:    {r['num_chunks']}")

    if r["num_chunks"] > 1:
        times = r["chunk_times"]
        gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
        print(f"  Avg inter-chunk gap: {sum(gaps)/len(gaps)*1000:.1f}ms")
        print(f"  Max inter-chunk gap: {max(gaps)*1000:.1f}ms")
        print(f"  Min inter-chunk gap: {min(gaps)*1000:.1f}ms")
        print(f"  Chunk sizes:         min={min(r['chunk_sizes'])} max={max(r['chunk_sizes'])} avg={sum(r['chunk_sizes'])//len(r['chunk_sizes'])}")

    if r["num_chunks"] == 1:
        print("  ⚠️  ONLY 1 CHUNK — response is BUFFERED, not real-time streaming!")
    else:
        print("  ✅ Multiple chunks — real-time streaming is working.")


async def main():
    results = {}

    print("=" * 60)
    print("  Streaming LOCAL first...")
    print("=" * 60)
    await stream_endpoint("LOCAL", LOCAL, results)

    print("\n" + "=" * 60)
    print("  Streaming PROD next...")
    print("=" * 60)
    await stream_endpoint("PROD", PROD, results)

    # Summaries
    for label, r in results.items():
        summarize(label, r)

    # Direct comparison
    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    local = results.get("LOCAL", {})
    prod  = results.get("PROD", {})
    lc = local.get("num_chunks", 0)
    pc = prod.get("num_chunks", 0)
    print(f"  LOCAL chunks: {lc}   |   PROD chunks: {pc}")
    if lc > 1 and pc == 1:
        print("  🚨 PROD is buffering the full response — streaming is broken on the server.")
        print("     Likely cause: reverse proxy (nginx/caddy) is buffering SSE.")
        print("     Fix: ensure proxy passes 'X-Accel-Buffering: no' and 'Cache-Control: no-cache'.")
    elif lc == 1 and pc == 1:
        print("  🚨 BOTH are buffering. The LLM itself is not streaming (thinking model or OpenRouter issue).")
    elif lc > 1 and pc > 1:
        print("  ✅ Both are streaming in real-time. Chunk counts match expectations.")


if __name__ == "__main__":
    asyncio.run(main())
