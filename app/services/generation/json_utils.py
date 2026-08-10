"""Robust JSON parser for LLM outputs with multi-strategy repair pipeline."""

import json
import re
from typing import Dict, Any, Optional, List
from jsonschema import validate, ValidationError
import logging

logger = logging.getLogger(__name__)

try:
    import ssm_jsonrepair as jsonrepair
    # Check if the repair method actually exists
    if hasattr(jsonrepair, 'repair'):
        JSONREPAIR_AVAILABLE = True
    else:
        JSONREPAIR_AVAILABLE = False
        logger.warning("ssm-jsonrepair library imported but 'repair' method not found. Skipping jsonrepair strategy.")
except ImportError:
    JSONREPAIR_AVAILABLE = False
    logger.warning("ssm-jsonrepair library not available. Install it for better JSON repair capabilities.")


class RobustJSONParser:
    """Centralized robust JSON parser with multi-strategy repair pipeline."""

    def __init__(self):
        """Initialize the parser."""
        self.repair_strategies = [
            self._parse_with_basic_repair,
            self._parse_with_missing_opening_quote_fix,  # NEW: Fix missing opening quotes in property names (CRITICAL)
            self._parse_with_state_machine_comma_fix,  # NEW: State-machine based comma insertion (most reliable)
            self._parse_with_control_char_escape,
            self._parse_with_unicode_quote_escape,  # NEW: Fix unescaped quotes in Unicode/Amharic text
            self._parse_with_malformed_key_fix,  # Fix malformed keys like "type":g":
            self._parse_with_error_position_repair,  # NEW: Enhanced comma insertion using error position
            self._parse_with_consecutive_object_fix,  # NEW: Fix missing commas between consecutive objects/arrays
            self._parse_with_error_position_targeted_fix,  # NEW: Target-specific error positions for delimiter errors (runs after error pos captured)
            self._parse_with_aggressive_repair,
            self._parse_with_smart_comma_insertion,
            self._parse_with_missing_comma_after_brace,  # Specific fix for } followed by key
            self._parse_with_end_of_structure_fix,  # Fix missing commas at end of nested structures
            self._parse_with_line_by_line_fix,
        ]
        # Only add jsonrepair strategy if it's available
        if JSONREPAIR_AVAILABLE:
            self.repair_strategies.append(self._parse_with_jsonrepair)

    def parse_json(
        self,
        text: str,
        schema: Optional[Dict[str, Any]] = None,
        extract_from_markdown: bool = True,
    ) -> Dict[str, Any]:
        """
        Parse JSON from text with robust error handling and repair.

        Args:
            text: Input text that may contain JSON (possibly wrapped in markdown)
            schema: Optional JSON schema for validation
            extract_from_markdown: Whether to extract JSON from markdown code blocks

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON cannot be parsed after all repair attempts
        """
        # Extract JSON string from markdown or text
        json_str = self.extract_json_from_text(text) if extract_from_markdown else text

        # Try parsing with multiple repair strategies
        last_error = None
        error_pos = None
        for strategy in self.repair_strategies:
            try:
                # If this is the error_position_targeted_fix strategy and we have error position, use it
                if (error_pos is not None and 
                    hasattr(strategy, '__name__') and 
                    strategy.__name__ == '_parse_with_error_position_targeted_fix'):
                    document = strategy(json_str, error_pos)
                else:
                    document = strategy(json_str)
                logger.info(f"Successfully parsed JSON using {strategy.__name__}")
                
                # Validate against schema if provided
                if schema:
                    self.validate_schema(document, schema)
                
                return document
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                # Capture error position for targeted repair (only for delimiter errors)
                if hasattr(e, 'pos') and e.pos:
                    error_msg = str(e).lower()
                    if "expecting" in error_msg and "delimiter" in error_msg:
                        error_pos = e.pos
                        logger.debug(f"Captured error position {error_pos} for delimiter error")
                
                error_msg = str(e).lower()
                if "expecting" in error_msg and "delimiter" in error_msg:
                    logger.debug(f"Delimiter error with {strategy.__name__} at position {error_pos}, trying next strategy")
                elif "extra data" in error_msg:
                    # Handle "Extra data" error by extracting first complete JSON object
                    logger.debug(f"Extra data error with {strategy.__name__}, trying to extract first complete JSON")
                    try:
                        json_str = self._extract_first_complete_json(json_str)
                        # Retry with extracted JSON
                        document = strategy(json_str)
                        logger.info(f"Successfully parsed JSON using {strategy.__name__} after extracting first complete JSON")
                        if schema:
                            self.validate_schema(document, schema)
                        return document
                    except Exception:
                        continue
                continue

        # Final attempt: if error is "Extra data", try extracting first complete JSON
        if last_error and "extra data" in str(last_error).lower():
            logger.debug("Attempting final recovery: extracting first complete JSON object")
            try:
                extracted_json = self._extract_first_complete_json(json_str)
                if extracted_json:
                    # Try parsing the extracted JSON with basic repair
                    fixed = extracted_json
                    fixed = re.sub(r',\s*}', '}', fixed)
                    fixed = re.sub(r',\s*]', ']', fixed)
                    document = json.loads(fixed)
                    logger.info("Successfully parsed JSON by extracting first complete object")
                    if schema:
                        self.validate_schema(document, schema)
                    return document
            except Exception as e:
                logger.debug(f"Final recovery attempt failed: {e}")
        
        # Final recovery: try to complete incomplete JSON structure
        if last_error and hasattr(last_error, 'pos') and last_error.pos:
            error_pos = last_error.pos
            error_msg = str(last_error).lower()
            # If error is about delimiter or incomplete structure near the end
            if (("expecting" in error_msg and "delimiter" in error_msg) or 
                "unterminated" in error_msg) and error_pos > len(json_str) * 0.8:
                logger.debug("Attempting to complete incomplete JSON structure")
                try:
                    completed_json = self._complete_incomplete_json(json_str, error_pos)
                    if completed_json:
                        # Try parsing the completed JSON
                        fixed = completed_json
                        fixed = re.sub(r',\s*}', '}', fixed)
                        fixed = re.sub(r',\s*]', ']', fixed)
                        document = json.loads(fixed)
                        logger.info("Successfully parsed JSON after completing incomplete structure")
                        if schema:
                            self.validate_schema(document, schema)
                        return document
                except Exception as e:
                    logger.debug(f"JSON completion attempt failed: {e}")
        
        # If all strategies fail, log detailed error and raise
        logger.error(f"Failed to parse JSON after all repair attempts: {last_error}")
        if hasattr(last_error, 'pos') and last_error.pos:
            error_pos = last_error.pos
            start = max(0, error_pos - 300)
            end = min(len(json_str), error_pos + 300)
            logger.error(f"Context around error (char {error_pos}): ...{json_str[start:end]}...")
        
        raise ValueError(f"Invalid JSON in LLM output: {str(last_error)}")

    def extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON string from text, handling markdown code blocks.

        Args:
            text: Input text that may contain JSON

        Returns:
            Extracted JSON string
        """
        # Try to extract from markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
            else:
                # Markdown block not closed - extract from start to end of text
                # and try to find the last complete JSON object
                logger.warning("Markdown code block not closed - attempting to extract partial JSON")
                json_candidate = text[start:].strip()
                # Try to find the last complete JSON object
                last_brace = json_candidate.rfind('}')
                if last_brace > 100:  # Only if we have substantial content
                    return json_candidate[:last_brace + 1]
                return json_candidate
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()
            else:
                # Similar handling for incomplete generic code blocks
                logger.warning("Code block not closed - attempting to extract partial JSON")
                json_candidate = text[start:].strip()
                last_brace = json_candidate.rfind('}')
                if last_brace > 100:
                    return json_candidate[:last_brace + 1]
                return json_candidate

        # Try to find first complete JSON object in the text
        json_str = self._extract_first_complete_json(text)
        if json_str:
            return json_str

        # Fallback: try to find JSON object using simple method
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]

        return text.strip()
    
    def _extract_first_complete_json(self, text: str) -> str:
        """
        Extract the first complete JSON object from text by finding matching braces.
        
        Args:
            text: Input text that may contain JSON
            
        Returns:
            First complete JSON object string, or empty string if not found
        """
        start = text.find("{")
        if start < 0:
            return ""
        
        # Use a stack to track nested structures
        stack = []  # Track '{' and '[' to handle nested structures properly
        in_string = False
        escape_next = False
        i = start
        
        while i < len(text):
            char = text[i]
            
            # Handle escape sequences
            if escape_next:
                escape_next = False
                i += 1
                continue
            
            if char == '\\' and in_string:
                escape_next = True
                i += 1
                continue
            
            # Track string boundaries
            if char == '"':
                in_string = not in_string
                i += 1
                continue
            
            # Only track braces/brackets outside strings
            if not in_string:
                if char == '{':
                    stack.append('{')
                elif char == '}':
                    if stack and stack[-1] == '{':
                        stack.pop()
                        if len(stack) == 0:
                            # Found the matching closing brace for the root object
                            return text[start:i+1]
                elif char == '[':
                    stack.append('[')
                elif char == ']':
                    if stack and stack[-1] == '[':
                        stack.pop()
            
            i += 1
        
        # If we didn't find a complete JSON object, return empty
        return ""
    
    def _complete_incomplete_json(self, json_str: str, error_pos: int) -> Optional[str]:
        """
        Attempt to complete an incomplete JSON structure by balancing braces and brackets.
        
        Args:
            json_str: Incomplete JSON string
            error_pos: Position where error occurred
            
        Returns:
            Completed JSON string, or None if completion not possible
        """
        # Extract the portion up to the error
        truncated = json_str[:error_pos]
        
        # Count braces and brackets
        open_braces = truncated.count('{')
        close_braces = truncated.count('}')
        open_brackets = truncated.count('[')
        close_brackets = truncated.count(']')
        
        # Check if we're in the middle of a string (don't complete if so)
        # Simple check: if last quote before error is unclosed
        last_quote = truncated.rfind('"')
        if last_quote >= 0:
            # Count quotes before last quote to see if it's closed
            quotes_before = truncated[:last_quote].count('"')
            if quotes_before % 2 == 0:  # Even number means last quote opens a string
                # We're in the middle of a string - can't complete
                return None
        
        # Find the last complete structure before error
        # Look for the last complete clause/object
        last_complete_brace = truncated.rfind('},')
        if last_complete_brace > 1000:  # Only if we have substantial content
            # Extract up to the last complete structure
            completed = truncated[:last_complete_brace + 1]
            
            # Close any open arrays first (inner structures)
            missing_brackets = completed.count('[') - completed.count(']')
            for _ in range(missing_brackets):
                completed += '\n      ]'
            
            # Then close any open objects
            missing_braces = completed.count('{') - completed.count('}')
            for _ in range(missing_braces):
                completed += '\n    }'
            
            # Ensure root object is closed
            if completed.count('{') > completed.count('}'):
                completed += '\n}'
            
            # Final balance check
            if completed.count('{') == completed.count('}') and completed.count('[') == completed.count(']'):
                return completed
        
        return None

    def validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """
        Validate JSON data against a schema.

        Args:
            data: JSON data to validate
            schema: JSON schema definition

        Raises:
            ValidationError: If data doesn't match schema
        """
        try:
            validate(instance=data, schema=schema)
        except ValidationError as e:
            logger.error(f"Schema validation failed: {e.message}")
            logger.error(f"Failed at path: {'.'.join(str(p) for p in e.path)}")
            raise ValueError(f"Schema validation failed: {e.message}") from e

    def _parse_with_basic_repair(self, json_str: str) -> Dict[str, Any]:
        """Parse JSON with basic repair (trailing commas, etc.)."""
        fixed = json_str

        # Fix trailing commas
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)

        # Fix missing commas between objects/arrays - handle newlines
        fixed = re.sub(r'}[\s\n]*{', '}, {', fixed, flags=re.MULTILINE)
        fixed = re.sub(r'][\s\n]*\[', '], [', fixed, flags=re.MULTILINE)
        fixed = re.sub(r'}[\s\n]*\[', '}, [', fixed, flags=re.MULTILINE)
        fixed = re.sub(r'][\s\n]*{', '], {', fixed, flags=re.MULTILINE)

        return json.loads(fixed)

    def _parse_with_missing_opening_quote_fix(self, json_str: str) -> Dict[str, Any]:
        """
        Fix missing opening quotes in property names.
        
        Handles cases like:
        - content": -> "content":
        - type": -> "type":
        - block_id": -> "block_id":
        
        This is a common LLM error where the opening quote is missing.
        This repair strategy runs early in the pipeline to catch this critical error.
        """
        # Use a state machine to track when we're expecting a property name
        result = []
        i = 0
        in_string = False
        escape_next = False
        stack = []  # Track object/array depth
        expecting_property = False
        
        while i < len(json_str):
            char = json_str[i]
            
            # Handle escape sequences
            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue
            
            if char == '\\' and in_string:
                result.append(char)
                escape_next = True
                i += 1
                continue
            
            # Track string boundaries
            if char == '"' and not escape_next:
                in_string = not in_string
                result.append(char)
                if not in_string:
                    expecting_property = False
                i += 1
                continue
            
            # If we're inside a string, keep everything as-is
            if in_string:
                result.append(char)
                i += 1
                continue
            
            # Track object/array structure
            if char == '{':
                stack.append('{')
                result.append(char)
                expecting_property = True
                i += 1
                continue
            elif char == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
                result.append(char)
                expecting_property = False
                i += 1
                continue
            elif char == '[':
                stack.append('[')
                result.append(char)
                expecting_property = False
                i += 1
                continue
            elif char == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                result.append(char)
                expecting_property = False
                i += 1
                continue
            elif char == ':':
                result.append(char)
                expecting_property = False
                i += 1
                continue
            elif char == ',':
                result.append(char)
                # After comma, if we're in an object, expect property name
                if stack and stack[-1] == '{':
                    expecting_property = True
                else:
                    expecting_property = False
                i += 1
                continue
            
            # Handle whitespace
            if char.isspace():
                result.append(char)
                i += 1
                continue
            
            # If we're expecting a property name and we see a letter/underscore
            # (not a quote), this might be a missing opening quote
            if expecting_property and stack and stack[-1] == '{':
                if char.isalpha() or char == '_':
                    # Look ahead to see if this is followed by ": pattern
                    # Collect the identifier
                    identifier = []
                    j = i
                    while j < len(json_str) and (json_str[j].isalnum() or json_str[j] == '_'):
                        identifier.append(json_str[j])
                        j += 1
                    
                    # Check if followed by ": (missing opening quote pattern)
                    if j < len(json_str) and json_str[j:j+2] == '":':
                        # This is a missing opening quote! Add it
                        result.append('"')
                        result.append(''.join(identifier))
                        i = j  # Skip the identifier we just processed
                        continue
            
            # Default: keep the character
            result.append(char)
            i += 1
        
        fixed = ''.join(result)
        
        # Also apply basic repairs
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        return json.loads(fixed)

    def _parse_with_state_machine_comma_fix(self, json_str: str) -> Dict[str, Any]:
        """
        Use a state machine to track JSON structure and insert missing commas.
        This is more reliable than regex-based approaches for complex nested structures.
        
        Handles cases like:
        - } "key": (missing comma after closing brace)
        - ] { (missing comma after closing bracket)
        - "value" "key": (missing comma between string values)
        """
        result = []
        i = 0
        in_string = False
        escape_next = False
        stack = []  # Track nesting: '{' or '['
        last_structural_char_pos = -1  # Position in original string of last structural character
        last_structural_char = None  # The actual character
        
        while i < len(json_str):
            char = json_str[i]
            
            # Handle escape sequences
            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue
            
            if char == '\\' and in_string:
                result.append(char)
                escape_next = True
                i += 1
                continue
            
            # Track string boundaries
            if char == '"':
                if not in_string:
                    # Starting a string - check if we need comma before it
                    if last_structural_char in ['}', ']', '"'] and last_structural_char_pos >= 0:
                        # Check if there's already a comma between last structural char and this quote
                        between = json_str[last_structural_char_pos + 1:i]
                        between_stripped = between.strip()
                        # If between is empty/whitespace or doesn't contain a comma, we need one
                        if not between_stripped or (',' not in between_stripped):
                            # Missing comma - insert it
                            result.append(',')
                            # Preserve any whitespace that was there
                            if between_stripped:
                                result.append(between)
                            else:
                                result.append(' ')
                else:
                    # Ending a string - mark as structural
                    last_structural_char = '"'
                    last_structural_char_pos = i
                
                in_string = not in_string
                result.append(char)
                i += 1
                continue
            
            # If inside string, keep as-is
            if in_string:
                result.append(char)
                i += 1
                continue
            
            # Track structure
            if char == '{':
                # Before opening brace, check if we need comma
                if stack and stack[-1] == '[':
                    # In array, check if previous element ended
                    if last_structural_char in ['}', ']', '"'] and last_structural_char_pos >= 0:
                        between = json_str[last_structural_char_pos + 1:i]
                        between_stripped = between.strip()
                        if not between_stripped or (',' not in between_stripped):
                            result.append(',')
                            if between_stripped:
                                result.append(between)
                            else:
                                result.append(' ')
                # Also check if we're in an object and need comma before new object
                elif stack and stack[-1] == '{':
                    if last_structural_char in ['}', ']', '"'] and last_structural_char_pos >= 0:
                        between = json_str[last_structural_char_pos + 1:i]
                        between_stripped = between.strip()
                        if not between_stripped or (',' not in between_stripped):
                            result.append(',')
                            if between_stripped:
                                result.append(between)
                            else:
                                result.append(' ')
                stack.append('{')
                result.append(char)
                last_structural_char = None
                last_structural_char_pos = -1
                i += 1
                continue
            elif char == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
                result.append(char)
                last_structural_char = '}'
                last_structural_char_pos = i
                i += 1
                continue
            elif char == '[':
                # Before opening bracket, check if we need comma (in array)
                if stack and stack[-1] == '[':
                    if last_structural_char in ['}', ']', '"'] and last_structural_char_pos >= 0:
                        between = json_str[last_structural_char_pos + 1:i]
                        between_stripped = between.strip()
                        if not between_stripped or (',' not in between_stripped):
                            result.append(',')
                            if between_stripped:
                                result.append(between)
                            else:
                                result.append(' ')
                stack.append('[')
                result.append(char)
                last_structural_char = None
                last_structural_char_pos = -1
                i += 1
                continue
            elif char == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                result.append(char)
                last_structural_char = ']'
                last_structural_char_pos = i
                i += 1
                continue
            elif char == ':':
                result.append(char)
                i += 1
                continue
            elif char == ',':
                result.append(char)
                last_structural_char = None
                last_structural_char_pos = -1
                i += 1
                continue
            elif char.isspace():
                result.append(char)
                i += 1
                continue
            
            # Handle numbers, booleans, null (these are structural elements)
            if char.isdigit() or char == '-' or char in ['t', 'f', 'n']:
                # Try to parse a complete number/boolean/null
                j = i
                if char == '-':
                    j += 1
                if j < len(json_str):
                    # Check for number
                    if json_str[j].isdigit():
                        while j < len(json_str) and (json_str[j].isdigit() or json_str[j] in '.eE+-'):
                            j += 1
                        value = json_str[i:j]
                        result.append(value)
                        last_structural_char = value[0] if value else None
                        last_structural_char_pos = i
                        i = j
                        continue
                    # Check for boolean/null
                    elif json_str[j:j+4] == 'true':
                        result.append('true')
                        last_structural_char = 't'
                        last_structural_char_pos = i
                        i = j + 4
                        continue
                    elif json_str[j:j+5] == 'false':
                        result.append('false')
                        last_structural_char = 'f'
                        last_structural_char_pos = i
                        i = j + 5
                        continue
                    elif json_str[j:j+4] == 'null':
                        result.append('null')
                        last_structural_char = 'n'
                        last_structural_char_pos = i
                        i = j + 4
                        continue
            
            # Check for missing comma: } or ] followed by " (key) or { or [
            # Note: This is a fallback check for cases not caught above
            if char in ['"', '{', '['] and last_structural_char in ['}', ']'] and last_structural_char_pos >= 0:
                # Check if there's already a comma between
                between = json_str[last_structural_char_pos + 1:i]
                between_stripped = between.strip()
                if not between_stripped or (',' not in between_stripped):
                    # Insert comma
                    result.append(',')
                    if between_stripped:
                        result.append(between)
                    else:
                        result.append(' ')
            
            result.append(char)
            i += 1
        
        fixed = ''.join(result)
        
        # Apply basic cleanup
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        fixed = re.sub(r',\s*,', ',', fixed)
        
        return json.loads(fixed)

    def _parse_with_error_position_targeted_fix(self, json_str: str, error_pos: Optional[int] = None) -> Dict[str, Any]:
        """
        Target-specific error positions to fix missing comma delimiters.
        This strategy uses the error position from previous parse attempts to fix the exact issue.
        
        Args:
            json_str: JSON string to parse
            error_pos: Optional error position from previous parse attempt
        """
        # First try basic repair
        fixed = json_str
        
        # Fix trailing commas
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # Apply general comma fixes first
        # Pattern: } or ] followed by whitespace/newlines then "key":
        fixed = re.sub(r'([}\]])[\s\n]+"([^"]+)"\s*:', r'\1, "\2":', fixed, flags=re.MULTILINE)
        
        # Pattern: } or ] directly followed by "key":
        fixed = re.sub(r'([}\]])"([^"]+)"\s*:', r'\1, "\2":', fixed)
        
        # Pattern: } or ] followed by { or [
        fixed = re.sub(r'([}\]])[\s\n]*([\[{])', r'\1, \2', fixed, flags=re.MULTILINE)
        
        # Clean up any double commas
        fixed = re.sub(r',\s*,', ',', fixed)
        
        # If we have an error position, target that area specifically with more aggressive fixes
        if error_pos is not None and error_pos < len(fixed):
            # Look backwards from error position to find where comma should be inserted
            # Search window: 500 chars before error position (larger window for complex structures)
            search_start = max(0, error_pos - 500)
            search_end = min(len(fixed), error_pos + 100)
            search_window = fixed[search_start:search_end]
            
            # Find the last closing brace/bracket before the error
            last_close_pos = -1
            for i in range(len(search_window) - 1, -1, -1):
                char = search_window[i]
                if char in ['}', ']']:
                    last_close_pos = search_start + i
                    break
            
            # Find the next structural element after the error (quote, brace, bracket)
            next_struct_pos = -1
            for i in range(error_pos, min(len(fixed), error_pos + 200)):
                char = fixed[i]
                if char in ['"', '{', '[']:
                    next_struct_pos = i
                    break
            
            # If we found both positions, check if comma is missing
            if last_close_pos >= 0 and next_struct_pos >= 0:
                between = fixed[last_close_pos + 1:next_struct_pos].strip()
                # If there's no comma and it's mostly whitespace, insert one
                if ',' not in between and (not between or between.isspace()):
                    # Insert comma after the closing brace/bracket
                    fixed = fixed[:last_close_pos + 1] + ',' + fixed[last_close_pos + 1:]
                    logger.debug(f"Inserted comma at position {last_close_pos + 1} based on error position {error_pos}")
                # Also try to fix if there's a quote or brace/bracket immediately after }
                elif next_struct_pos == last_close_pos + 1:
                    # Directly adjacent - definitely needs comma
                    fixed = fixed[:last_close_pos + 1] + ',' + fixed[last_close_pos + 1:]
                    logger.debug(f"Inserted comma at position {last_close_pos + 1} (directly adjacent structures)")
        
        # Try parsing - if it still fails with delimiter error, try one more targeted fix
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            error_msg = str(e).lower()
            if "expecting" in error_msg and "delimiter" in error_msg and hasattr(e, 'pos') and e.pos:
                # Try one more targeted fix at the new error position
                new_error_pos = e.pos
                if new_error_pos < len(fixed):
                    # Look for } or ] before the error
                    for i in range(new_error_pos - 1, max(0, new_error_pos - 300), -1):
                        if fixed[i] in ['}', ']']:
                            # Check if there's a comma after it
                            j = i + 1
                            while j < len(fixed) and fixed[j].isspace():
                                j += 1
                            if j < len(fixed) and fixed[j] != ',':
                                # Insert comma
                                fixed = fixed[:i + 1] + ',' + fixed[i + 1:]
                                logger.debug(f"Inserted comma at position {i + 1} after retry (error at {new_error_pos})")
                                break
                    # Try parsing again
                    return json.loads(fixed)
            raise

    def _parse_with_control_char_escape(self, json_str: str) -> Dict[str, Any]:
        """
        Parse JSON by escaping control characters (newlines, tabs, etc.) in string values.
        This handles cases where HTML content contains unescaped newlines.
        """
        result = []
        i = 0
        in_string = False
        escape_next = False

        while i < len(json_str):
            char = json_str[i]

            # Handle escape sequences first
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

    def _parse_with_unicode_quote_escape(self, json_str: str) -> Dict[str, Any]:
        """
        Parse JSON by escaping unescaped quotes in Unicode/Amharic text strings.
        
        This handles cases where LLM generates JSON with unescaped quotes inside
        string values containing Unicode characters (like Amharic text).
        """
        result = []
        i = 0
        in_string = False
        escape_next = False
        string_start = -1
        
        while i < len(json_str):
            char = json_str[i]
            
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
            
            # Track string boundaries
            if char == '"':
                if not in_string:
                    # Starting a string
                    in_string = True
                    string_start = i
                    result.append(char)
                else:
                    # Inside a string - check if this should close it
                    # Look ahead to see what comes after this quote
                    j = i + 1
                    # Skip whitespace
                    while j < len(json_str) and json_str[j] in ' \t\n\r':
                        j += 1
                    
                    # Check if this quote is followed by a JSON delimiter
                    if j >= len(json_str):
                        # End of string - closes it
                        result.append(char)
                        in_string = False
                    elif json_str[j] in ':},]':
                        # Followed by JSON delimiter - this closes the string
                        result.append(char)
                        in_string = False
                    elif json_str[j] == '"':
                        # Two quotes in a row - might be escaped quote or empty string
                        # Check if previous char was backslash
                        if i > 0 and json_str[i-1] == '\\':
                            # Already escaped
                            result.append(char)
                        else:
                            # Empty string or escaped quote - close current, start new
                            result.append(char)
                            in_string = False
                    else:
                        # Quote is inside string content (unescaped) - escape it
                        # But only if we're actually in a string value (not a key)
                        # Check if we're in a value by looking backwards for ':'
                        is_value = False
                        for k in range(string_start - 1, max(0, string_start - 50), -1):
                            if json_str[k] == ':':
                                is_value = True
                                break
                            elif json_str[k] in '{[,':
                                break
                        
                        if is_value:
                            # Escape the quote
                            result.append('\\"')
                        else:
                            # In a key - this might close it
                            result.append(char)
                            in_string = False
                
                i += 1
                continue
            
            # If we're inside a string, keep everything as-is (except quotes which we handle above)
            if in_string:
                result.append(char)
                i += 1
                continue
            
            # Outside string - keep as-is
            result.append(char)
            i += 1
        
        fixed = ''.join(result)
        
        # Also apply basic repairs
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        return json.loads(fixed)

    def _parse_with_malformed_key_fix(self, json_str: str) -> Dict[str, Any]:
        """Parse JSON by fixing malformed keys and stray characters using a systematic approach."""
        # Use a character-by-character state machine to remove invalid characters
        fixed = self._remove_stray_characters(json_str)
        
        # Fix pattern: "key":X": where X is a single character (likely typo)
        # This fixes cases like "type":g": -> "type":
        fixed = re.sub(r'"([^"]+)":([a-zA-Z0-9_]):"', r'"\1":', fixed)
        
        # More specific: Fix "key":X": pattern where X is any single character
        # This handles the exact error case: "type":g":
        fixed = re.sub(r'":([a-zA-Z0-9_]):"', '":', fixed)
        
        # Fix pattern: "key":X" where X is a single character before a quote (but not colon)
        # This fixes cases where there's an extra char before closing quote in value
        fixed = re.sub(r'":([a-zA-Z0-9_])"([^:,\}\]])', r'": "\2', fixed)
        
        # Also apply basic repairs before parsing
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # Try to parse
        return json.loads(fixed)
    
    def _remove_stray_characters(self, json_str: str) -> str:
        """
        Systematically remove stray characters that appear where property names or values should be.
        Uses a state machine to track JSON structure and remove invalid characters.
        """
        result = []
        i = 0
        in_string = False
        escape_next = False
        stack = []  # Track object/array depth: '{' or '['
        expecting_property = False  # True when we're expecting a property name (after comma or opening brace)
        expecting_value = False  # True when we're expecting a value (after colon)
        last_char = None
        
        while i < len(json_str):
            char = json_str[i]
            
            # Handle escape sequences in strings
            if escape_next:
                result.append(char)
                escape_next = False
                last_char = char
                i += 1
                continue
            
            if char == '\\' and in_string:
                result.append(char)
                escape_next = True
                last_char = char
                i += 1
                continue
            
            # Track string boundaries
            if char == '"' and not escape_next:
                in_string = not in_string
                result.append(char)
                if not in_string:
                    expecting_property = False
                    expecting_value = False
                last_char = char
                i += 1
                continue
            
            # If we're inside a string, keep everything as-is
            if in_string:
                result.append(char)
                last_char = char
                i += 1
                continue
            
            # Track object/array structure
            if char == '{':
                stack.append('{')
                result.append(char)
                expecting_property = True  # After opening brace, expect property name
                expecting_value = False
                last_char = char
                i += 1
                continue
            elif char == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
                result.append(char)
                expecting_property = False
                expecting_value = False
                last_char = char
                i += 1
                continue
            elif char == '[':
                stack.append('[')
                result.append(char)
                expecting_property = False
                expecting_value = False
                last_char = char
                i += 1
                continue
            elif char == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                result.append(char)
                expecting_property = False
                expecting_value = False
                last_char = char
                i += 1
                continue
            
            # Track colons and commas
            if char == ':':
                result.append(char)
                expecting_property = False
                expecting_value = True  # After colon, expect value
                last_char = char
                i += 1
                continue
            elif char == ',':
                result.append(char)
                expecting_property = True  # After comma, expect property name (in object) or value (in array)
                expecting_value = False
                last_char = char
                i += 1
                continue
            
            # Handle whitespace - always allowed
            if char.isspace():
                result.append(char)
                last_char = char
                i += 1
                continue
            
            # If we're expecting a property name (after comma or opening brace in object)
            if expecting_property and stack and stack[-1] == '{':
                # Property name must start with quote
                if char == '"':
                    result.append(char)
                    expecting_property = False
                    last_char = char
                    i += 1
                    continue
                # Skip stray characters that aren't valid JSON (like -, _, =, +, etc.)
                # But '-' could be start of negative number, so we need to be careful
                elif char in ['_', '=', '+', '|', '&', '*', '#', '@', '!', '%', '^']:
                    # Skip this character - it's a stray character
                    i += 1
                    continue
                # Handle '-' specially - could be negative number or stray dash
                elif char == '-':
                    # Look ahead to see if it's followed by a digit
                    if i + 1 < len(json_str) and json_str[i + 1].isdigit():
                        # It's a negative number - but we're expecting a property name, so this is malformed
                        # Skip it as a stray character
                        i += 1
                        continue
                    else:
                        # It's a stray dash
                        i += 1
                        continue
                # If it's a valid JSON value start (number, boolean, null, object, array), that's also OK
                elif char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 't', 'f', 'n', '{', '[']:
                    # This might be a valid value, but we expected a property name
                    # In JSON, property names must be strings, so this is likely an error
                    # But let's be lenient and allow it (might be a malformed JSON that we'll fix elsewhere)
                    result.append(char)
                    expecting_property = False
                    last_char = char
                    i += 1
                    continue
            
            # If we're expecting a value (after colon)
            if expecting_value:
                # Value can start with quote, number, boolean, null, object, array
                if char == '"':
                    # String value - most common case
                    result.append(char)
                    expecting_value = False
                    last_char = char
                    i += 1
                    continue
                elif char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 't', 'f', 'n', '{', '[']:
                    # Number, boolean, null, object, or array
                    result.append(char)
                    expecting_value = False
                    last_char = char
                    i += 1
                    continue
                elif char == '-':
                    # Could be negative number or stray dash
                    # Look ahead to see if it's followed by a digit or quote
                    if i + 1 < len(json_str):
                        next_char = json_str[i + 1]
                        if next_char.isdigit():
                            # It's a negative number
                            result.append(char)
                            expecting_value = False
                            last_char = char
                            i += 1
                            continue
                        elif next_char == '"':
                            # Stray dash before quote: "key":- "value" -> "key": "value"
                            # Skip the dash
                            i += 1
                            continue
                        else:
                            # Stray dash - skip it
                            i += 1
                            continue
                    else:
                        # End of string - skip dash
                        i += 1
                        continue
                # Skip other stray characters
                elif char in ['_', '=', '+', '|', '&', '*', '#', '@', '!', '%', '^']:
                    # Skip stray character after colon
                    i += 1
                    continue
            
            # Default: keep the character
            result.append(char)
            last_char = char
            i += 1
        
        return ''.join(result)

    def _parse_with_error_position_repair(self, json_str: str) -> Dict[str, Any]:
        """
        Enhanced comma insertion using error position from JSONDecodeError.
        
        This strategy attempts to parse the JSON, catches the error, and uses
        the exact error position to intelligently insert missing commas.
        """
        # First try - parse as-is to get error position
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            error_msg = str(e).lower()
            
            # Only handle delimiter errors (missing commas)
            if "expecting" not in error_msg or "delimiter" not in error_msg:
                raise
            
            if not hasattr(e, 'pos') or not e.pos:
                raise
            
            logger.debug(f"Delimiter error at position {e.pos}: {error_msg}")
            
            # Strategy 1: Find and fix missing comma using error context
            fixed = self._insert_comma_at_error_position(json_str, e.pos)
            if fixed:
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass  # Try next strategy
            
            # Strategy 2: Scan backwards from error position for structural markers
            fixed = self._fix_comma_by_structure_scan(json_str, e.pos)
            if fixed:
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass  # Try next strategy
            
            # Strategy 3: Look for common patterns around error position
            fixed = self._fix_comma_by_pattern_match(json_str, e.pos)
            if fixed:
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass
            
            # If all strategies fail, re-raise the original error
            raise
    
    def _insert_comma_at_error_position(self, json_str: str, error_pos: int) -> Optional[str]:
        """Insert comma at the most likely position based on error context."""
        # Get context around error
        start = max(0, error_pos - 200)
        context_before = json_str[start:error_pos]
        context_after = json_str[error_pos:min(len(json_str), error_pos + 200)]
        
        # Find the last structural character (}, ], or ")
        last_brace = context_before.rfind('}')
        last_bracket = context_before.rfind(']')
        last_quote = context_before.rfind('"')
        
        # Find which is closest to error
        positions = []
        if last_brace >= 0:
            positions.append((last_brace, '}'))
        if last_bracket >= 0:
            positions.append((last_bracket, ']'))
        if last_quote >= 0:
            positions.append((last_quote, '"'))
        
        if not positions:
            return None
        
        # Get the closest structural character
        pos, char = max(positions, key=lambda x: x[0])
        abs_pos = start + pos
        
        # Check what comes after this character
        after_char = json_str[abs_pos + 1:abs_pos + 50].lstrip()
        
        # Case 1: } or ] followed by " (start of next key)
        if char in ['}', ']'] and after_char.startswith('"'):
            # Insert comma after the } or ]
            return json_str[:abs_pos + 1] + ',' + json_str[abs_pos + 1:]
        
        # Case 2: } or ] followed by another } or ]
        if char in ['}', ']'] and (after_char.startswith('}') or after_char.startswith(']')):
            # This might be end of nested structures - add comma
            return json_str[:abs_pos + 1] + ',' + json_str[abs_pos + 1:]
        
        # Case 3: " (end of string value) followed by " (start of next key)
        if char == '"':
            # Look for pattern: "value" followed by "key":
            if after_char.startswith('"') and ':' in after_char[:50]:
                return json_str[:abs_pos + 1] + ',' + json_str[abs_pos + 1:]
        
        return None
    
    def _fix_comma_by_structure_scan(self, json_str: str, error_pos: int) -> Optional[str]:
        """Scan structure backwards from error to find where comma should be inserted."""
        # Track nesting depth by scanning backwards
        depth = 0
        in_string = False
        escape_next = False
        
        # Scan backwards from error position
        for i in range(error_pos - 1, max(0, error_pos - 500), -1):
            char = json_str[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if in_string:
                continue
            
            # Track depth
            if char in ['}', ']']:
                depth += 1
            elif char in ['{', '[']:
                depth -= 1
            
            # When we return to depth 0, we found a complete structure
            if depth == 0 and char in ['}', ']']:
                # Check what follows this closing bracket
                after = json_str[i + 1:error_pos].strip()
                next_char = json_str[i + 1:].lstrip()[:1] if i + 1 < len(json_str) else ''
                
                # If next char is " or { or [, we need a comma
                if next_char and next_char in ['"', '{', '['] and not after.startswith(','):
                    return json_str[:i + 1] + ',' + json_str[i + 1:]
        
        return None
    
    def _fix_comma_by_pattern_match(self, json_str: str, error_pos: int) -> Optional[str]:
        """Use regex patterns to find and fix missing commas near error position."""
        # Get a window around the error
        start = max(0, error_pos - 500)  # Increased window for Unicode text
        end = min(len(json_str), error_pos + 200)
        window = json_str[start:end]
        
        # Pattern 1: } followed by whitespace/newlines then "
        matches = list(re.finditer(r'([}\]])([\s\n]+)(")', window))
        if matches:
            # Find the match closest to the error position
            for match in matches:
                match_pos = start + match.start(2)  # Position of whitespace
                if abs(match_pos - error_pos) < 200:  # Within 200 chars of error
                    # Insert comma after } or ]
                    insert_pos = start + match.end(1)
                    return json_str[:insert_pos] + ',' + json_str[insert_pos:]
        
        # Pattern 2: ] or } followed by ] or }
        matches = list(re.finditer(r'([}\]])([\s\n]+)([}\]])', window))
        if matches:
            for match in matches:
                match_pos = start + match.start(2)
                if abs(match_pos - error_pos) < 200:
                    insert_pos = start + match.end(1)
                    return json_str[:insert_pos] + ',' + json_str[insert_pos:]
        
        # Pattern 3: " (end of value) followed by whitespace then " (start of key with :)
        matches = list(re.finditer(r'(")([\s\n]+)(")([^"]*?)(:)', window))
        if matches:
            for match in matches:
                match_pos = start + match.start(2)
                if abs(match_pos - error_pos) < 200:
                    insert_pos = start + match.end(1)
                    return json_str[:insert_pos] + ',' + json_str[insert_pos:]
        
        # Pattern 4: } ] followed by whitespace then { (missing comma between array items)
        matches = list(re.finditer(r'([}\]])[\s\n]+([{])', window))
        if matches:
            for match in matches:
                match_pos = start + match.start()
                if abs(match_pos - error_pos) < 200:
                    insert_pos = start + match.end(1)
                    return json_str[:insert_pos] + ',' + json_str[insert_pos:]
        
        # Pattern 5: Look for } ] pattern followed by { (common in arrays of objects)
        # This handles: } ] } { pattern where comma is missing
        matches = list(re.finditer(r'([}\]])[\s\n]*([}\]])[\s\n]*([{])', window))
        if matches:
            for match in matches:
                match_pos = start + match.start()
                if abs(match_pos - error_pos) < 200:
                    # Insert comma after the second closing bracket
                    insert_pos = start + match.end(2)
                    return json_str[:insert_pos] + ',' + json_str[insert_pos:]
        
        return None

    def _parse_with_consecutive_object_fix(self, json_str: str) -> Dict[str, Any]:
        """
        Fix missing commas between consecutive objects/arrays in lists.
        
        Common LLM pattern:
        { "attr": value }
        { "next": value }  <- missing comma after first }
        """
        fixed = json_str
        
        # Pattern 1: } followed by newline(s) and whitespace, then { 
        # This is common in arrays of objects where comma is missing
        fixed = re.sub(
            r'}(\s*\n\s*)(\s*){',  # } followed by newlines/whitespace then {
            r'},\1{',              # Insert comma after }
            fixed
        )
        
        # Pattern 2: ] followed by newline(s) and whitespace, then [
        # Arrays within arrays
        fixed = re.sub(
            r'](\s*\n\s*)(\s*)\[',
            r'],\1[',
            fixed
        )
        
        # Pattern 3: } followed by newline(s) and whitespace, then [
        # Object followed by array in list
        fixed = re.sub(
            r'}(\s*\n\s*)(\s*)\[',
            r'},\1[',
            fixed
        )
        
        # Pattern 4: ] followed by newline(s) and whitespace, then {
        # Array followed by object in list
        fixed = re.sub(
            r'](\s*\n\s*)(\s*){',
            r'],\1{',
            fixed
        )
        
        # Clean up any double commas we might have created
        fixed = re.sub(r',\s*,', ',', fixed)
        
        # Remove trailing commas
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            # If this strategy fails, let the error bubble up to try next strategy
            raise

    def _parse_with_aggressive_repair(self, json_str: str) -> Dict[str, Any]:
        """Parse JSON with aggressive repair (handles missing commas and unescaped quotes)."""
        fixed = json_str

        # Fix unescaped quotes in HTML content
        fixed = self._fix_unescaped_quotes_in_strings(fixed)

        # Remove trailing commas (more aggressive)
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)

        # Fix missing commas between objects/arrays - handle newlines aggressively
        fixed = re.sub(r'}[\s\n]*{', '}, {', fixed, flags=re.MULTILINE)
        fixed = re.sub(r'][\s\n]*\[', '], [', fixed, flags=re.MULTILINE)
        fixed = re.sub(r'}[\s\n]*\[', '}, [', fixed, flags=re.MULTILINE)
        fixed = re.sub(r'][\s\n]*{', '], {', fixed, flags=re.MULTILINE)

        # Fix missing commas after closing braces/brackets before strings
        # Handle both with and without whitespace/newlines
        fixed = re.sub(r'([}\]])"([^"]+)"\s*:', r'\1, "\2":', fixed)
        # More aggressive: handle newlines and whitespace between } and "
        fixed = re.sub(r'([}\]])[\s\n]+"([^"]+)"\s*:', r'\1, "\2":', fixed)

        # Fix missing commas after string values before next key
        fixed = re.sub(r'"\s*"([^"]+)"\s*:', r'", "\1":', fixed)
        
        # Fix missing commas after closing quote followed by closing bracket/brace
        # Handles: "...text"\n          }\n          ] -> "...text",\n          }\n          ]
        # This is common when a text node ends and is followed by closing brackets
        fixed = re.sub(r'"([^"]*)"[\s\n]+([}\]])', r'"\1", \2', fixed, flags=re.MULTILINE)
        
        # Fix missing commas between closing brackets/braces (more aggressive)
        # Handles: }\n          ]\n        } -> },\n          ],\n        }
        fixed = re.sub(r'([}\]])[\s\n]+([\]}])', r'\1, \2', fixed, flags=re.MULTILINE)

        # Remove double commas that might have been created
        fixed = re.sub(r',\s*,\s*', ',', fixed)

        # Remove any comments (// or /* */)
        fixed = re.sub(r'//.*?$', '', fixed, flags=re.MULTILINE)
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)

        return json.loads(fixed)

    def _fix_unescaped_quotes_in_strings(self, text: str) -> str:
        """Escape unescaped quotes inside JSON string values using a state machine."""
        result = []
        i = 0
        in_string = False
        escape_next = False
        is_value = False  # True if we're in a value string (after :), False if in key
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
                    is_value = True
                    result.append(char)
                    i += 1
                    continue

            # Handle quotes
            if char == '"':
                if in_string:
                    # Inside a string - check if this ends it
                    j = i + 1
                    while j < len(text) and text[j] in ' \t\n\r':
                        j += 1

                    if j >= len(text):
                        result.append(char)
                        in_string = False
                        is_value = False
                    elif text[j] in ':,}]':
                        # Followed by JSON delimiter - closes string
                        result.append(char)
                        in_string = False
                        is_value = False
                    else:
                        # This quote is inside the string content - escape it if in value
                        if is_value:
                            result.append('\\"')
                        else:
                            result.append(char)
                            in_string = False
                            is_value = False
                else:
                    # Starting a string
                    in_string = True
                    result.append(char)

                i += 1
                continue

            # Regular character
            result.append(char)
            i += 1

        return ''.join(result)

    def _parse_with_smart_comma_insertion(self, json_str: str) -> Dict[str, Any]:
        """Parse JSON by intelligently inserting missing commas."""
        fixed = json_str

        # Pattern 1: Closing brace/bracket followed by quote (missing comma before key)
        fixed = re.sub(r'([}\]])"([^"]+)"\s*:', r'\1, "\2":', fixed)

        # Pattern 2: String value followed by quote (missing comma between key-value pairs)
        fixed = re.sub(r'":\s*"([^"]*)"\s*"([^"]+)"\s*:', r'": "\1", "\2":', fixed)

        # Pattern 3: Number/boolean/null followed by quote (missing comma)
        fixed = re.sub(r'(\d+|true|false|null)\s*"([^"]+)"\s*:', r'\1, "\2":', fixed)

        # Pattern 4: Closing brace/bracket followed by opening brace/bracket (missing comma)
        fixed = re.sub(r'}\s*{', '},{', fixed)
        fixed = re.sub(r']\s*\[', '],[', fixed)
        fixed = re.sub(r'}\s*\[', '},[', fixed)
        fixed = re.sub(r']\s*{', '],{', fixed)

        # Pattern 5: Handle nested structures
        lines = fixed.split('\n')
        fixed_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if i > 0 and stripped.startswith('"') and ':' in stripped:
                prev_line = lines[i-1].rstrip()
                if prev_line.endswith(('}', ']')) and not prev_line.endswith(','):
                    prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
                    curr_indent = len(line) - len(line.lstrip())
                    if curr_indent <= prev_indent:
                        fixed_lines.append(lines[i-1] + ',')
                        fixed_lines.append(line)
                        continue
            fixed_lines.append(line)
        fixed = '\n'.join(fixed_lines)

        # Final cleanup
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        fixed = re.sub(r',\s*,\s*', ',', fixed)

        return json.loads(fixed)

    def _parse_with_missing_comma_after_brace(self, json_str: str) -> Dict[str, Any]:
        """Parse JSON by fixing missing commas after closing braces/brackets before keys."""
        fixed = json_str
        
        # Pattern 1: } followed by whitespace/newlines then a quoted key
        # Matches: }\n          "type": -> },\n          "type":
        fixed = re.sub(r'([}\]])[\s\n]+"([^"]+)"\s*:', r'\1, "\2":', fixed, flags=re.MULTILINE)
        
        # Pattern 2: } directly followed by a quoted key (no whitespace)
        # Matches: }"type": -> },"type":
        fixed = re.sub(r'([}\]])"([^"]+)"\s*:', r'\1, "\2":', fixed)
        
        # Pattern 3: Fix missing commas between closing brackets/braces in nested structures
        # This handles cases where we have } followed by ] or } followed by }
        # But ONLY if they're at the same nesting level (not parent-child relationship)
        # We'll be conservative and only add commas when there's whitespace/newlines
        fixed = re.sub(r'([}\]])[\s\n]+([\]}])', r'\1, \2', fixed, flags=re.MULTILINE)
        
        # Pattern 4: Remove any double commas we might have created
        fixed = re.sub(r',\s*,\s*', ',', fixed)
        
        # Pattern 5: Remove trailing commas before final closing brackets/braces
        # BUT be careful - only remove if it's clearly a trailing comma (not between elements)
        # We'll only remove trailing commas at the very end of the JSON
        # This is safer than removing all trailing commas which could break valid nested structures
        if fixed.rstrip().endswith('}') or fixed.rstrip().endswith(']'):
            # Only remove trailing comma if it's right before the final closing bracket
            fixed = re.sub(r',(\s*[}\]])$', r'\1', fixed, flags=re.MULTILINE)
        
        return json.loads(fixed)

    def _parse_with_end_of_structure_fix(self, json_str: str) -> Dict[str, Any]:
        """Parse JSON by fixing missing commas at the end of nested structures."""
        fixed = json_str
        
        # Pattern 1: Fix missing commas between closing brackets/braces in arrays/objects
        # Handles: }\n          ]\n        } -> },\n          ],\n        }
        # This is common at the end of nested structures
        fixed = re.sub(r'([}\]])[\s\n]+([\]}])', r'\1, \2', fixed, flags=re.MULTILINE)
        
        # Pattern 2: Fix cases where closing brace/bracket is followed by another closing
        # without proper comma separation (especially in nested structures)
        # More aggressive: handle multiple consecutive closing brackets
        # Matches: }] -> },]
        fixed = re.sub(r'([}\]])[\s\n]*([\]}])', r'\1, \2', fixed, flags=re.MULTILINE)
        
        # Pattern 3: Remove any double commas we might have created
        fixed = re.sub(r',\s*,\s*', ',', fixed)
        
        # Pattern 4: Remove trailing commas before final closing brackets (but keep internal ones)
        # Only remove if it's the absolute last thing before closing
        # We need to be careful here - only remove trailing commas that are clearly wrong
        # Check if comma is before final closing bracket/brace at the end
        if fixed.rstrip().endswith('}') or fixed.rstrip().endswith(']'):
            # Remove trailing comma only if it's right before the final closing
            fixed = re.sub(r',(\s*[}\]])$', r'\1', fixed, flags=re.MULTILINE)
        
        return json.loads(fixed)

    def _parse_with_line_by_line_fix(self, json_str: str) -> Dict[str, Any]:
        """Parse JSON by fixing issues line by line."""
        fixed = json_str

        # Fix missing commas - handle newlines and whitespace explicitly
        fixed = re.sub(r'}\s*"([^"]+)"\s*:', r'}, "\1":', fixed)
        fixed = re.sub(r']\s*"([^"]+)"\s*:', r'], "\1":', fixed)
        fixed = re.sub(r'(\d+|true|false|null)\s*"([^"]+)"\s*:', r'\1, "\2":', fixed)
        # More aggressive pattern that handles newlines
        fixed = re.sub(r'([}\]])[\s\n]+"([^"]+)"\s*:', r'\1, "\2":', fixed)
        fixed = re.sub(r'([}\]])"([^"]+)"\s*:', r'\1, "\2":', fixed)

        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            error_msg = str(e).lower()
            # Check if this is a delimiter error (missing comma)
            if "expecting" in error_msg and ("delimiter" in error_msg or "," in error_msg):
                if hasattr(e, 'pos') and e.pos:
                    # Try to insert comma at error position
                    error_pos = e.pos
                    before = fixed[:error_pos]
                    after = fixed[error_pos:]

                    # Find the last } or ] before error
                    last_close = max(before.rfind('}'), before.rfind(']'))
                    if last_close >= 0:
                        between = before[last_close+1:error_pos].strip()
                        # If there's content between } and error position, add comma
                        if between and not between.startswith(',') and not between.startswith('"'):
                            # Check if it looks like a key (starts with quote)
                            if '"' in between or between.strip():
                                fixed = before[:last_close+1] + ',' + before[last_close+1:] + after
                                try:
                                    return json.loads(fixed)
                                except:
                                    pass
                        # If between is empty or just whitespace, try adding comma after }
                        elif not between or between.strip() == '':
                            # Look ahead to see if next non-whitespace is a quote or closing bracket
                            lookahead = after.lstrip()
                            if lookahead.startswith('"') or lookahead.startswith('}') or lookahead.startswith(']'):
                                fixed = before[:last_close+1] + ',' + before[last_close+1:] + after
                                try:
                                    return json.loads(fixed)
                                except:
                                    pass
                        # More aggressive: if error is about delimiter and we have } followed by quote/bracket
                        # Try inserting comma right after the }
                        if after.lstrip().startswith('"') or after.lstrip().startswith('}') or after.lstrip().startswith(']'):
                            fixed = before[:last_close+1] + ',' + before[last_close+1:] + after
                            try:
                                return json.loads(fixed)
                            except:
                                pass
                        
                        # Special case: if we're near the end of the JSON and error is at a closing bracket
                        # Try to find if we're missing a comma between array/object elements
                        if error_pos > len(fixed) * 0.9:  # Near the end (last 10%)
                            # Look for pattern: } followed by } or ] followed by ]
                            # This suggests missing comma between elements
                            lookback = before[max(0, last_close-50):]
                            if re.search(r'[}\]][\s\n]*[}\]][\s\n]*[}\]][\s\n]*[}\]][\s\n]*$', lookback):
                                # Multiple closing brackets/braces suggest nested structure
                                # Try adding comma before the last closing bracket
                                if last_close > 0:
                                    fixed = before[:last_close+1] + ',' + before[last_close+1:] + after
                                    try:
                                        return json.loads(fixed)
                                    except:
                                        pass
                        
                        # Even more aggressive: if error is at delimiter and we're in a nested structure
                        # Try inserting comma at multiple positions before the error
                        # Check if we have a pattern like: }"text" followed by } or ]
                        # This suggests we need a comma after the closing quote
                        if '"' in before[last_close:error_pos]:
                            # Find the last quote before error
                            last_quote = before.rfind('"', last_close, error_pos)
                            if last_quote > last_close:
                                # Check if there's whitespace/newline between quote and error
                                between_quote_error = before[last_quote+1:error_pos].strip()
                                if not between_quote_error or between_quote_error in ['}', ']']:
                                    # Try adding comma after the quote
                                    fixed = before[:last_quote+1] + ',' + before[last_quote+1:] + after
                                    try:
                                        return json.loads(fixed)
                                    except:
                                        pass
                        
                        # Final attempt: if we have } followed by whitespace then ] or }
                        # and error is about delimiter, try adding comma after }
                        if last_close >= 0:
                            after_close = before[last_close+1:error_pos].strip()
                            if not after_close or after_close in ['}', ']']:
                                lookahead = after.lstrip()
                                if lookahead.startswith(']') or lookahead.startswith('}'):
                                    fixed = before[:last_close+1] + ',' + before[last_close+1:] + after
                                    try:
                                        return json.loads(fixed)
                                    except:
                                        pass

            raise

    def _parse_with_jsonrepair(self, json_str: str) -> Dict[str, Any]:
        """Parse JSON using ssm-jsonrepair library as final fallback."""
        if not JSONREPAIR_AVAILABLE:
            raise ValueError("ssm-jsonrepair library not available")

        try:
            # Check if repair method exists (defensive check)
            if not hasattr(jsonrepair, 'repair'):
                raise ValueError("ssm-jsonrepair.repair method not found")
            
            # jsonrepair.repair() returns a string, so we need to parse it
            repaired_json_str = jsonrepair.repair(json_str)
            return json.loads(repaired_json_str)
        except AttributeError as e:
            logger.warning(f"ssm-jsonrepair API issue: {e}. Skipping this strategy.")
            raise ValueError(f"ssm-jsonrepair API not available: {str(e)}") from e
        except Exception as e:
            logger.warning(f"ssm-jsonrepair failed: {e}. This is expected if JSON is malformed.")
            raise ValueError(f"ssm-jsonrepair failed: {str(e)}") from e


# Global instance for convenience
_parser_instance = None


def get_json_parser() -> RobustJSONParser:
    """Get the global JSON parser instance."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = RobustJSONParser()
    return _parser_instance


def parse_json(
    text: str,
    schema: Optional[Dict[str, Any]] = None,
    extract_from_markdown: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to parse JSON using the global parser.

    Args:
        text: Input text that may contain JSON
        schema: Optional JSON schema for validation
        extract_from_markdown: Whether to extract JSON from markdown code blocks

    Returns:
        Parsed JSON dictionary
    """
    return get_json_parser().parse_json(text, schema, extract_from_markdown)

