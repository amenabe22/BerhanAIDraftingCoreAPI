"""Document builder for constructing exact document structure."""

from typing import Dict, Any, List
from datetime import datetime
import json
import logging
from app.services.generation.json_utils import parse_json
from app.services.generation.schemas import DOCUMENT_SCHEMA

logger = logging.getLogger(__name__)


class DocumentBuilder:
    """Service for building document structure from LLM output."""

    def parse_llm_output(self, llm_output: str) -> Dict[str, Any]:
        """
        Parse LLM output and extract TipTap JSON document structure.
        Uses centralized robust JSON parser with enhanced error messages.
        
        Args:
            llm_output: Raw LLM response (may contain JSON wrapped in markdown or text)
        
        Returns:
            Parsed TipTap document structure dictionary
        
        Raises:
            ValueError: If JSON cannot be parsed, with detailed error context
        """
        try:
            # Pre-check syntax using syntax checker for better error messages
            from app.services.generation.json_syntax_checker import JSONSyntaxChecker
            syntax_checker = JSONSyntaxChecker()
            is_valid, syntax_errors = syntax_checker.check_syntax(llm_output)
            
            if not is_valid and syntax_errors:
                # Log syntax errors for debugging
                error_summary = syntax_checker.get_error_summary(syntax_errors)
                logger.warning(f"JSON syntax pre-check found {error_summary['total_errors']} errors: {error_summary['error_types']}")
                
                # If we have specific missing comma errors, provide helpful message
                missing_comma_errors = [e for e in syntax_errors if e.error_type == "missing_comma"]
                if missing_comma_errors:
                    first_error = missing_comma_errors[0]
                    error_msg = (
                        f"Missing comma detected at position {first_error.position}. "
                        f"Context: {first_error.context[:100]}... "
                        f"After every }} or ]] that is NOT the last element, you MUST add a comma."
                    )
                    logger.error(error_msg)
            
            # Use centralized JSON parser without old schema validation
            document = parse_json(
                llm_output,
                schema=None,  # Don't validate against old schema
                extract_from_markdown=True
            )
            logger.info("Successfully parsed TipTap JSON document structure")
            return self._normalize_tiptap_structure(document)
        except ValueError as e:
            error_msg = str(e)
            
            # Enhance error message with position context if available
            if "char" in error_msg.lower() or "position" in error_msg.lower():
                # Try to extract position from error message
                import re
                pos_match = re.search(r'(?:char|position)\s+(\d+)', error_msg, re.IGNORECASE)
                if pos_match:
                    error_pos = int(pos_match.group(1))
                    if error_pos < len(llm_output):
                        # Get context around error
                        start = max(0, error_pos - 100)
                        end = min(len(llm_output), error_pos + 100)
                        context = llm_output[start:end]
                        relative_pos = error_pos - start
                        
                        # Mark error position
                        marked_context = (
                            context[:relative_pos] + 
                            " ⬅️ ERROR HERE ⬅️ " + 
                            context[relative_pos:]
                        )
                        
                        enhanced_error = (
                            f"{error_msg}\n\n"
                            f"Error Context (position {error_pos}):\n"
                            f"{marked_context}\n\n"
                            f"Common fixes:\n"
                            f"- If you see }} or ]] followed by \" or {{ or [, add a comma\n"
                            f"- If you see a property name without opening quote, add \"\n"
                            f"- If you see unescaped quotes in strings, escape them as \\\""
                        )
                        logger.error(f"Failed to parse LLM output as JSON:\n{enhanced_error}")
                        raise ValueError(enhanced_error) from e
            
            logger.error(f"Failed to parse LLM output as JSON: {error_msg}")
            raise
    
    def _parse_with_repair(self, json_str: str) -> Dict[str, Any]:
        """Parse JSON with basic repair (trailing commas, etc.)."""
        import re
        
        # Fix trailing commas
        fixed = re.sub(r',\s*}', '}', json_str)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # Fix missing commas between objects/arrays
        # Pattern: } { or ] [ or } [ or ] {
        fixed = re.sub(r'}\s*{', '},{', fixed)
        fixed = re.sub(r']\s*\[', '],[', fixed)
        fixed = re.sub(r'}\s*\[', '},[', fixed)
        fixed = re.sub(r']\s*{', '],{', fixed)
        
        return json.loads(fixed)
    
    def _parse_with_control_char_escape(self, json_str: str) -> Dict[str, Any]:
        """
        Parse JSON by escaping control characters (newlines, tabs, etc.) in string values.
        This handles cases where HTML content contains unescaped newlines.
        """
        import re
        
        fixed = json_str
        
        # Strategy: Find all JSON string values and escape control characters
        # We need to be careful to only escape inside string values, not in keys or structure
        
        result = []
        i = 0
        in_string = False
        escape_next = False
        
        while i < len(fixed):
            char = fixed[i]
            
            # Handle escape sequences first
            if escape_next:
                # The previous char was \, so this char is escaped
                result.append(char)
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                result.append(char)
                escape_next = True
                i += 1
                continue
            
            # Track string boundaries (only if not escaped)
            if char == '"':
                in_string = not in_string
                result.append(char)
                i += 1
                continue
            
            # If we're inside a string, escape control characters
            if in_string:
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                elif char == '\b':
                    result.append('\\b')
                elif char == '\f':
                    result.append('\\f')
                elif ord(char) < 32:  # Other control characters
                    result.append(f'\\u{ord(char):04x}')
                else:
                    result.append(char)
            else:
                result.append(char)
            
            i += 1
        
        fixed = ''.join(result)
        
        # Also fix trailing commas and other common issues
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        return json.loads(fixed)
    
    def _parse_with_aggressive_repair(self, json_str: str) -> Dict[str, Any]:
        """Parse JSON with aggressive repair (handles missing commas and unescaped quotes)."""
        import re
        
        fixed = json_str
        
        # CRITICAL: Fix unescaped quotes in HTML content first
        # Pattern: "content": "<p>"text"</p>" needs quotes escaped
        # Strategy: Use regex to find and escape quotes inside JSON string values
        def fix_unescaped_quotes_in_strings(text: str) -> str:
            """Escape unescaped quotes inside JSON string values using a more robust approach."""
            import re
            
            # Strategy: Find all JSON string values (pattern: "key": "value")
            # and escape any unescaped quotes inside the value part
            
            # Use a character-by-character state machine that properly handles:
            # - Escaped characters (\\, \")
            # - String boundaries
            # - Distinguishing between keys and values
            
            result = []
            i = 0
            in_string = False
            escape_next = False
            is_value = False  # True if we're in a value string (after :), False if in key (before :)
            stack = []  # Track object/array depth
            
            while i < len(text):
                char = text[i]
                
                # Handle escape sequences
                if escape_next:
                    result.append(char)
                    escape_next = False
                    i += 1
                    continue
                
                if char == '\\':
                    result.append(char)
                    escape_next = True
                    i += 1
                    continue
                
                # Track object/array boundaries (outside strings)
                if not in_string:
                    if char == '{':
                        stack.append('{')
                        result.append(char)
                        i += 1
                        continue
                    elif char == '}':
                        if stack and stack[-1] == '{':
                            stack.pop()
                        result.append(char)
                        i += 1
                        continue
                    elif char == '[':
                        stack.append('[')
                        result.append(char)
                        i += 1
                        continue
                    elif char == ']':
                        if stack and stack[-1] == '[':
                            stack.pop()
                        result.append(char)
                        i += 1
                        continue
                    elif char == ':':
                        # Colon indicates next string will be a value
                        is_value = True
                        result.append(char)
                        i += 1
                        continue
                
                # Handle quotes
                if char == '"':
                    if in_string:
                        # Inside a string - check if this ends it
                        # Look ahead (skip whitespace) to see what follows
                        j = i + 1
                        while j < len(text) and text[j] in ' \t\n\r':
                            j += 1
                        
                        if j >= len(text):
                            # End of text - closes string
                            result.append(char)
                            in_string = False
                            is_value = False
                        elif text[j] in ':,\}]':
                            # Followed by JSON delimiter - closes string
                            result.append(char)
                            in_string = False
                            is_value = False
                        else:
                            # This quote is inside the string content
                            # Escape it if we're in a value string
                            if is_value:
                                result.append('\\"')
                            else:
                                # In a key - shouldn't have quotes, but handle gracefully
                                result.append(char)
                                in_string = False
                                is_value = False
                    else:
                        # Starting a string
                        # If we just saw a colon, this is a value string
                        in_string = True
                        result.append(char)
                        # is_value flag is already set from colon detection
                    
                    i += 1
                    continue
                
                # Regular character
                result.append(char)
                i += 1
            
            return ''.join(result)
        
        # Fix unescaped quotes first
        fixed = fix_unescaped_quotes_in_strings(fixed)
        
        # Remove trailing commas (more aggressive)
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        
        # Fix missing commas between objects/arrays (most common issue)
        # Pattern: } { or ] [ or } [ or ] { (missing comma)
        fixed = re.sub(r'}\s*{', '},{', fixed)
        fixed = re.sub(r']\s*\[', '],[', fixed)
        fixed = re.sub(r'}\s*\[', '},[', fixed)
        fixed = re.sub(r']\s*{', '],{', fixed)
        
        # Fix missing commas after closing braces/brackets before strings
        # Pattern: } "key" or ] "key" (missing comma)
        fixed = re.sub(r'([}\]])"([^"]+)"\s*:', r'\1, "\2":', fixed)
        
        # Fix missing commas after string values before next key
        # Pattern: "value" "key": (missing comma)
        fixed = re.sub(r'"\s*"([^"]+)"\s*:', r'", "\1":', fixed)
        
        # Fix missing commas after values before closing braces/brackets
        # Pattern: "value" } or "value" ] (should have comma if in array/object)
        # But be careful - only if it's clearly in a list context
        fixed = re.sub(r'"\s*([}\]])', r'", \1', fixed)
        
        # Remove double commas that might have been created
        fixed = re.sub(r',\s*,\s*', ',', fixed)
        
        # Remove any comments (// or /* */)
        fixed = re.sub(r'//.*?$', '', fixed, flags=re.MULTILINE)
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        
        return json.loads(fixed)
    
    def _parse_with_smart_comma_insertion(self, json_str: str) -> Dict[str, Any]:
        """Parse JSON by intelligently inserting missing commas (handles 'Expecting , delimiter' errors)."""
        import re
        
        fixed = json_str
        
        # Strategy: Find patterns where comma is missing and insert it
        # Pattern 1: Closing brace/bracket followed by quote (missing comma before key)
        # } "key": or ] "key":
        fixed = re.sub(r'([}\]])"([^"]+)"\s*:', r'\1, "\2":', fixed)
        
        # Pattern 2: String value followed by quote (missing comma between key-value pairs)
        # "value" "key": - but be careful, this might be inside HTML content
        # Only fix if it's clearly a JSON structure (not HTML)
        # Look for: "key": "value" "key2": pattern
        fixed = re.sub(r'":\s*"([^"]*)"\s*"([^"]+)"\s*:', r'": "\1", "\2":', fixed)
        
        # Pattern 3: Number/boolean/null followed by quote (missing comma)
        fixed = re.sub(r'(\d+|true|false|null)\s*"([^"]+)"\s*:', r'\1, "\2":', fixed)
        
        # Pattern 4: Closing brace/bracket followed by opening brace/bracket (missing comma)
        fixed = re.sub(r'}\s*{', '},{', fixed)
        fixed = re.sub(r']\s*\[', '],[', fixed)
        fixed = re.sub(r'}\s*\[', '},[', fixed)
        fixed = re.sub(r']\s*{', '],{', fixed)
        
        # Pattern 5: Value followed by closing brace/bracket (missing comma in arrays/objects)
        # "value" } or "value" ] or number } or number ]
        # But only if we're clearly in a list/array context
        fixed = re.sub(r'(["\d\w])\s*([}\]])', r'\1, \2', fixed)
        # Then remove commas that were incorrectly added before closing braces at object/array end
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        
        # Pattern 6: Handle nested structures - } followed by "key": on next logical level
        # This handles cases like: } "key": where } closes a nested object
        lines = fixed.split('\n')
        fixed_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # If line starts with " after a closing brace on previous line
            if i > 0 and stripped.startswith('"') and ':' in stripped:
                prev_line = lines[i-1].rstrip()
                if prev_line.endswith(('}', ']')) and not prev_line.endswith(','):
                    # Check if we need to add comma - look at indentation
                    prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
                    curr_indent = len(line) - len(line.lstrip())
                    # If same or less indentation, we're at same level - need comma
                    if curr_indent <= prev_indent:
                        fixed_lines.append(lines[i-1] + ',')
                        fixed_lines.append(line)
                        continue
            fixed_lines.append(line)
        fixed = '\n'.join(fixed_lines)
        
        # Final cleanup - remove trailing commas
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # Remove double commas
        fixed = re.sub(r',\s*,\s*', ',', fixed)
        
        return json.loads(fixed)
    
    def _parse_with_line_by_line_fix(self, json_str: str) -> Dict[str, Any]:
        """Parse JSON by fixing issues line by line and character by character."""
        import re
        
        # First, try to fix common patterns
        fixed = json_str
        
        # Fix missing commas - look for patterns where comma is missing
        # Pattern 1: } followed by " (missing comma before next key)
        fixed = re.sub(r'}\s*"([^"]+)"\s*:', r'}, "\1":', fixed)
        
        # Pattern 2: ] followed by " (missing comma before next key)
        fixed = re.sub(r']\s*"([^"]+)"\s*:', r'], "\1":', fixed)
        
        # Pattern 3: "value" followed by "key": (missing comma)
        # This is tricky - need to ensure we're not inside HTML content
        # Look for pattern: "content": "<p>..." "key": (this shouldn't happen, but handle it)
        
        # Pattern 4: Number or boolean followed by "key": (missing comma)
        fixed = re.sub(r'(\d+|true|false|null)\s*"([^"]+)"\s*:', r'\1, "\2":', fixed)
        
        # Pattern 5: } or ] followed by { or [ (missing comma)
        fixed = re.sub(r'([}\]])"([^"]+)"\s*:', r'\1, "\2":', fixed)
        
        # Now try character-by-character repair for specific error positions
        # This is a fallback that tries to insert commas at likely positions
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            if hasattr(e, 'pos') and e.pos:
                # Try to insert comma at error position or nearby
                error_pos = e.pos
                # Look backwards for } or ] and forward for " or { or [
                before = fixed[:error_pos]
                after = fixed[error_pos:]
                
                # Find the last } or ] before error
                last_close = max(before.rfind('}'), before.rfind(']'))
                if last_close >= 0:
                    # Check if there's already a comma
                    between = before[last_close+1:].strip()
                    if between and not between.startswith(','):
                        # Insert comma after the closing brace/bracket
                        fixed = before[:last_close+1] + ',' + before[last_close+1:] + after
                        return json.loads(fixed)
            
            raise
    
    def _parse_with_json5_fallback(self, json_str: str) -> Dict[str, Any]:
        """Final fallback: Specifically escape quotes inside HTML content in JSON strings."""
        import re
        
        # Strategy: Find all "content": "..." or similar fields with HTML
        # and escape quotes that are clearly inside HTML tags
        
        fixed = json_str
        
        # Pattern 1: Escape quotes inside HTML tags: <tag>"text"</tag>
        # Match: <...>"..."<...> and escape the quotes
        fixed = re.sub(r'(<[^>]*>)"([^"]+)"(<[^>]*>)', r'\1\\"\2\\"\3', fixed)
        
        # Pattern 2: Escape quotes in text between tags: >"text"<
        fixed = re.sub(r'(>)([^<]*)"([^"]+)"([^<]*)(<)', r'\1\2\\"\3\\"\4\5', fixed)
        
        # But be careful not to double-escape
        fixed = re.sub(r'\\\\"', r'\\"', fixed)
        
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            # Log the specific error for debugging
            logger.debug(f"HTML quote escape strategy failed: {e}")
            raise

    def _normalize_tiptap_structure(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate TipTap document structure."""
        # Ensure root type is "doc"
        if document.get("type") != "doc":
            raise ValueError("Document must have 'type': 'doc' at root")
        
        # Ensure content exists
        if "content" not in document:
            raise ValueError("Document missing 'content' field")
        
        if not isinstance(document["content"], list):
            raise ValueError("'content' must be a list")
        
        # Add block_ids to paragraphs that don't have them
        block_counter = 1
        
        def add_block_ids(nodes: List[Dict[str, Any]]) -> None:
            nonlocal block_counter
            for node in nodes:
                node_type = node.get("type", "")
                
                # Add block_id to paragraphs and listItems
                if node_type in ["paragraph", "listItem"]:
                    attrs = node.get("attrs", {})
                    if "block_id" not in attrs:
                        attrs["block_id"] = f"b{block_counter}"
                        node["attrs"] = attrs
                        block_counter += 1
                
                # Recursively process content
                if "content" in node and isinstance(node["content"], list):
                    add_block_ids(node["content"])
        
        add_block_ids(document["content"])
        
        logger.info(f"Normalized TipTap document with {block_counter - 1} blocks")
        return document
    
    def _normalize_structure(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy: Normalize old format (kept for backward compatibility)."""
        return self._normalize_tiptap_structure(document)

    def convert_to_ingestion_format(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert TipTap document structure to ingestion format (HTML blocks).
        
        Args:
            document: TipTap JSON document structure
        
        Returns:
            List of sections/blocks in ingestion format
        """
        # Ingestion/format conversion stays on the main API for this phase.
        logger.info("convert_to_ingestion_format is not supported in core-service generation")
        return []

