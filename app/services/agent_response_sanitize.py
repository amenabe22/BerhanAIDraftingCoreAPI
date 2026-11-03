"""Strip internal document/block identifiers from user-facing agent text."""

from __future__ import annotations

import re

# Parenthetical dumps: (Block: uuid, uuid…)
_BLOCK_PAREN_RE = re.compile(
    r"\s*\(\s*Block:\s*[^)]+\)",
    re.IGNORECASE,
)
# Inline "Block: <id>" leftovers (comma-separated ids allowed)
_BLOCK_INLINE_RE = re.compile(
    r"\s*Block:\s*[A-Za-z0-9_\-]+(?:\s*,\s*[A-Za-z0-9_\-]+)*",
    re.IGNORECASE,
)
# Tool headers sometimes echoed: [Doc: … | Block: … | …], [Source: …], [Cassation: …]
_DOC_HEADER_RE = re.compile(
    r"\[(?:Doc|Document section|Source:|Cassation:)[^\]]*\]",
    re.IGNORECASE,
)
# Internal cassation document_id slugs (cite by ሰበር number instead)
_CASSATION_SLUG_RE = re.compile(
    r"\(?\s*\bcassation-(?:published|unpublished)-\d+-\d+\b\s*\)?",
    re.IGNORECASE,
)
_BLOCK_ID_EQ_RE = re.compile(
    r"\bblock_id\s*=\s*[A-Za-z0-9_\-]+",
    re.IGNORECASE,
)

# Hold-back markers for streaming (incomplete patterns near chunk boundary)
_HOLD_PREFIXES = (
    "(Block:",
    "(block:",
    "(BLOCK:",
    "Block:",
    "block:",
    "[Doc:",
    "[doc:",
    "[Document section",
    "[Source:",
    "[Cassation:",
)


def sanitize_user_facing_agent_text(text: str) -> str:
    """Remove internal block/doc ids that must never appear in chat replies."""
    if not text:
        return text
    out = _BLOCK_PAREN_RE.sub("", text)
    out = _DOC_HEADER_RE.sub("", out)
    out = _CASSATION_SLUG_RE.sub("", out)
    out = _BLOCK_ID_EQ_RE.sub("", out)
    out = _BLOCK_INLINE_RE.sub("", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" +\n", "\n", out)
    return out


def _hold_back_tail(buf: str) -> tuple[str, str]:
    """Split buffer into (emit_now, hold) so incomplete Block/Doc markers aren't cut mid-pattern."""
    if not buf:
        return "", ""
    lower = buf.lower()
    hold_from: int | None = None
    for prefix in _HOLD_PREFIXES:
        p = prefix.lower()
        # Incomplete prefix at end: "(Blo", "Block", "[Doc"
        for i in range(1, len(p) + 1):
            if lower.endswith(p[:i]) and not lower.endswith(p):
                start = len(buf) - i
                hold_from = start if hold_from is None else min(hold_from, start)
        # Full prefix opened but not closed yet
        idx = lower.rfind(p)
        if idx >= 0:
            rest = buf[idx:]
            if p.startswith("(") and ")" not in rest:
                hold_from = idx if hold_from is None else min(hold_from, idx)
            elif p.startswith("[") and "]" not in rest:
                hold_from = idx if hold_from is None else min(hold_from, idx)
            elif p.startswith("block:") and "\n" not in rest and ")" not in rest:
                # bare Block:… until newline / paren close / end
                hold_from = idx if hold_from is None else min(hold_from, idx)
    if hold_from is None:
        return buf, ""
    return buf[:hold_from], buf[hold_from:]


class StreamBlockIdSanitizer:
    """Sanitize token stream without leaking partial Block/UUID fragments."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        emit, self._buf = _hold_back_tail(self._buf)
        if not emit:
            return ""
        return sanitize_user_facing_agent_text(emit)

    def flush(self) -> str:
        out = sanitize_user_facing_agent_text(self._buf)
        self._buf = ""
        return out
