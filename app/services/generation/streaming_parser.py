"""Incremental JSON parser for streaming drafting-block generation."""

import json
import logging
from typing import List, Dict, Any, Set

logger = logging.getLogger(__name__)


class BlocksStreamingParser:
    """
    Incrementally parse drafting-blocks JSON and yield complete blocks in real-time.

    Expects LLM output shaped like ``{"blocks":[...], "metadata"?:{...}}`` where each
    block is ``{id, type, index, level?, html, plain_text}``.
    """

    def __init__(self):
        self.buffer = ""
        self.depth = 0
        self.in_string = False
        self.escape_next = False
        self.candidates = []
        self.parsed_node_hashes: Set[int] = set()
        self.total_parsed = 0
        self.total_streamed = 0
        self.last_structural_char = None
        self.last_structural_pos = -1
        self.commas_inserted = 0

    def feed(self, chunk: str) -> List[Dict[str, Any]]:
        if not chunk:
            return []

        self.buffer += chunk
        complete_nodes = []
        i = 0
        while i < len(self.buffer):
            char = self.buffer[i]

            if self.escape_next:
                self.escape_next = False
                i += 1
                continue

            if char == "\\":
                self.escape_next = True
                i += 1
                continue

            if char == '"':
                if not self.in_string:
                    self._check_and_repair_missing_comma(i, '"')
                else:
                    self.last_structural_char = '"'
                    self.last_structural_pos = i
                self.in_string = not self.in_string
                i += 1
                continue

            if not self.in_string:
                if char == "{":
                    self.candidates.append({"start": i, "depth": self.depth})
                    self.depth += 1
                    i += 1
                    continue
                elif char == "}":
                    self.depth -= 1
                    completed_candidates = [
                        c for c in self.candidates if c["depth"] == self.depth
                    ]
                    for candidate in completed_candidates:
                        obj_json = self.buffer[candidate["start"] : i + 1]
                        try:
                            obj = json.loads(obj_json.strip())
                            self.total_parsed += 1
                            if self._is_streamable_block(obj):
                                node_hash = hash(json.dumps(obj, sort_keys=True))
                                if node_hash not in self.parsed_node_hashes:
                                    self.parsed_node_hashes.add(node_hash)
                                    complete_nodes.append(obj)
                                    self.total_streamed += 1
                                    logger.debug(
                                        "✅ Streamed block #%s: %s",
                                        self.total_streamed,
                                        obj.get("type") or obj.get("id"),
                                    )
                        except json.JSONDecodeError:
                            pass
                        except Exception as e:
                            logger.warning("Unexpected parser error: %s", e, exc_info=False)
                        self.candidates.remove(candidate)
                    i += 1
                    continue
                elif char == "[":
                    self.depth += 1
                    self.last_structural_char = None
                    self.last_structural_pos = -1
                    i += 1
                    continue
                elif char == "]":
                    self.depth -= 1
                    self.last_structural_char = "]"
                    self.last_structural_pos = i
                    i += 1
                    continue
                elif char == ",":
                    self.last_structural_char = None
                    self.last_structural_pos = -1
                    i += 1
                    continue
            i += 1

        if len(self.buffer) > 20000:
            chars_to_trim = len(self.buffer) - 10000
            self.buffer = self.buffer[chars_to_trim:]
            self.candidates = [
                {"start": max(0, c["start"] - chars_to_trim), "depth": c["depth"]}
                for c in self.candidates
                if c["start"] >= chars_to_trim
            ]

        if complete_nodes:
            logger.info(
                "📦 Batch: %s blocks streamed (total: %s, parsed: %s)",
                len(complete_nodes),
                self.total_streamed,
                self.total_parsed,
            )
        return complete_nodes

    def _is_streamable_block(self, obj: Dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False
        # Canonical drafting block
        if obj.get("id") and obj.get("type") and (
            "plain_text" in obj or "html" in obj or "text" in obj
        ):
            return True
        # Legacy TipTap content nodes (inbound during transition)
        node_type = obj.get("type", "")
        streamable_types = {
            "heading",
            "paragraph",
            "bulletList",
            "orderedList",
            "listItem",
            "blockquote",
            "codeBlock",
            "table",
            "tableRow",
            "tableCell",
            "horizontalRule",
            "image",
            "hardBreak",
        }
        non_streamable = {"doc", "page", "pageBreak", "text"}
        if node_type in streamable_types:
            return True
        if node_type in non_streamable:
            return False
        return False

    def _check_and_repair_missing_comma(self, pos: int, next_char: str) -> None:
        if self.last_structural_char not in ["}", "]", '"']:
            return
        if self.last_structural_pos < 0:
            return
        between = self.buffer[self.last_structural_pos + 1 : pos]
        between_stripped = between.strip()
        if "," not in between_stripped and (not between_stripped or between_stripped.isspace()):
            needs_comma = False
            if next_char == '"':
                lookahead = pos + 1
                while lookahead < len(self.buffer) and self.buffer[lookahead].isspace():
                    lookahead += 1
                if lookahead < len(self.buffer) and self.buffer[lookahead] == ":":
                    needs_comma = True
            elif next_char in ["{", "["]:
                needs_comma = True
            if needs_comma:
                insert_pos = self.last_structural_pos + 1
                while insert_pos < len(self.buffer) and self.buffer[insert_pos].isspace():
                    insert_pos += 1
                self.buffer = self.buffer[:insert_pos] + "," + self.buffer[insert_pos:]
                self.commas_inserted += 1

    def reset(self):
        self.buffer = ""
        self.depth = 0
        self.in_string = False
        self.escape_next = False
        self.candidates = []
        self.parsed_node_hashes = set()
        self.total_parsed = 0
        self.total_streamed = 0
        self.last_structural_char = None
        self.last_structural_pos = -1
        self.commas_inserted = 0
        logger.info("🔄 Parser reset for new document")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_parsed": self.total_parsed,
            "total_streamed": self.total_streamed,
            "buffer_size": len(self.buffer),
            "active_candidates": len(self.candidates),
            "dedupe_cache_size": len(self.parsed_node_hashes),
            "commas_inserted": self.commas_inserted,
        }


# Backward-compatible alias for existing imports
TipTapStreamingParser = BlocksStreamingParser
