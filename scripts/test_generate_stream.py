#!/usr/bin/env python3
"""Smoke-test POST /drafting/generate/stream and print SSE events."""

from __future__ import annotations

import argparse
import json
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Test document generation SSE stream")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--message",
        default=(
            "Draft a simple 2-page NDA between Acme PLC and Beta LLC under "
            "Ethiopian law. Keep it simple and just generate it."
        ),
    )
    parser.add_argument("--language", default="en", choices=["en", "am", "om"])
    parser.add_argument("--action", default="start", choices=["start", "message", "finalize"])
    parser.add_argument("--thread-id", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--num-pages", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    body: dict = {
        "message": args.message,
        "language": args.language,
        "action": args.action,
        "num_pages": args.num_pages,
    }
    if args.thread_id:
        body["thread_id"] = args.thread_id
    if args.model:
        body["model"] = args.model

    url = args.base_url.rstrip("/") + "/drafting/generate/stream"
    print(f"POST {url}", file=sys.stderr)
    print(json.dumps(body, indent=2), file=sys.stderr)

    with httpx.Client(timeout=args.timeout) as client:
        with client.stream(
            "POST",
            url,
            json=body,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            print(f"status={resp.status_code} content-type={resp.headers.get('content-type')}", file=sys.stderr)
            if resp.status_code != 200:
                print(resp.read().decode(), file=sys.stderr)
                return 1
            buffer = ""
            for chunk in resp.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    part, buffer = buffer.split("\n\n", 1)
                    for line in part.split("\n"):
                        if line.startswith("data: "):
                            raw = line[6:]
                            try:
                                data = json.loads(raw)
                            except json.JSONDecodeError:
                                print(raw)
                                continue
                            etype = data.get("type")
                            if etype == "token":
                                sys.stdout.write(data.get("content") or "")
                                sys.stdout.flush()
                            else:
                                print(f"\n[{etype}] {json.dumps(data, default=str)[:500]}")
    print(file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
