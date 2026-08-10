"""JSON syntax checker for pre-parse validation.

Validates JSON syntax before attempting full parse to provide
better error messages and faster failure detection.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class JSONSyntaxError:
    """Represents a JSON syntax error with position and context."""
    
    def __init__(
        self,
        error_type: str,
        position: int,
        message: str,
        context: str = ""
    ):
        self.error_type = error_type
        self.position = position
        self.message = message
        self.context = context
    
    def __str__(self):
        return f"{self.error_type} at position {self.position}: {self.message}"


class JSONSyntaxChecker:
    """
    Pre-parse JSON syntax checker.
    
    Validates JSON syntax before attempting full parse to catch common errors early.
    """
    
    def __init__(self):
        """Initialize the syntax checker."""
        pass
    
    def check_syntax(self, json_str: str) -> Tuple[bool, List[JSONSyntaxError]]:
        """
        Check JSON syntax and return errors if any.
        
        Args:
            json_str: JSON string to check
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check bracket/brace matching
        bracket_errors = self._check_bracket_matching(json_str)
        errors.extend(bracket_errors)
        
        # Check comma placement
        comma_errors = self._check_comma_placement(json_str)
        errors.extend(comma_errors)
        
        # Check string escaping
        escape_errors = self._check_string_escaping(json_str)
        errors.extend(escape_errors)
        
        # Check property name quotes
        quote_errors = self._check_property_quotes(json_str)
        errors.extend(quote_errors)
        
        return len(errors) == 0, errors
    
    def _check_bracket_matching(self, json_str: str) -> List[JSONSyntaxError]:
        """Check if all brackets and braces are properly matched."""
        errors = []
        stack = []
        in_string = False
        escape_next = False
        
        for i, char in enumerate(json_str):
            # Handle escape sequences
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\' and in_string:
                escape_next = True
                continue
            
            # Track string boundaries
            if char == '"':
                in_string = not in_string
                continue
            
            # Only check brackets outside strings
            if not in_string:
                if char == '{':
                    stack.append(('{', i))
                elif char == '}':
                    if not stack or stack[-1][0] != '{':
                        errors.append(JSONSyntaxError(
                            "unmatched_brace",
                            i,
                            "Unmatched closing brace '}'",
                            self._get_context(json_str, i, 50, 50)
                        ))
                    else:
                        stack.pop()
                elif char == '[':
                    stack.append(('[', i))
                elif char == ']':
                    if not stack or stack[-1][0] != '[':
                        errors.append(JSONSyntaxError(
                            "unmatched_bracket",
                            i,
                            "Unmatched closing bracket ']'",
                            self._get_context(json_str, i, 50, 50)
                        ))
                    else:
                        stack.pop()
        
        # Check for unclosed brackets
        for bracket_type, pos in stack:
            errors.append(JSONSyntaxError(
                "unclosed_bracket",
                pos,
                f"Unclosed {bracket_type}",
                self._get_context(json_str, pos, 50, 50)
            ))
        
        return errors
    
    def _check_comma_placement(self, json_str: str) -> List[JSONSyntaxError]:
        """Check for missing commas between elements."""
        errors = []
        in_string = False
        escape_next = False
        last_structural_char = None
        last_structural_pos = -1
        
        for i, char in enumerate(json_str):
            # Handle escape sequences
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\' and in_string:
                escape_next = True
                continue
            
            # Track string boundaries
            if char == '"':
                if not in_string:
                    # Starting string - check for missing comma
                    if last_structural_char in ['}', ']', '"']:
                        between = json_str[last_structural_pos + 1:i]
                        if ',' not in between.strip() and (not between.strip() or between.strip().isspace()):
                            # Check if this looks like a property name (followed by :)
                            lookahead = i + 1
                            while lookahead < len(json_str) and json_str[lookahead].isspace():
                                lookahead += 1
                            if lookahead < len(json_str) and json_str[lookahead] == ':':
                                errors.append(JSONSyntaxError(
                                    "missing_comma",
                                    i,
                                    f"Missing comma after {last_structural_char} before property name",
                                    self._get_context(json_str, i, 50, 50)
                                ))
                in_string = not in_string
                if not in_string:
                    last_structural_char = '"'
                    last_structural_pos = i
                continue
            
            # Only check structure outside strings
            if not in_string:
                if char in ['{', '[']:
                    # Check for missing comma before structural element
                    if last_structural_char in ['}', ']']:
                        between = json_str[last_structural_pos + 1:i]
                        if ',' not in between.strip() and (not between.strip() or between.strip().isspace()):
                            errors.append(JSONSyntaxError(
                                "missing_comma",
                                i,
                                f"Missing comma after {last_structural_char} before {char}",
                                self._get_context(json_str, i, 50, 50)
                            ))
                    last_structural_char = None
                    last_structural_pos = -1
                elif char in ['}', ']']:
                    last_structural_char = char
                    last_structural_pos = i
                elif char == ',':
                    last_structural_char = None
                    last_structural_pos = -1
        
        return errors
    
    def _check_string_escaping(self, json_str: str) -> List[JSONSyntaxError]:
        """Check for unescaped quotes and control characters in strings."""
        errors = []
        in_string = False
        escape_next = False
        string_start = -1
        
        for i, char in enumerate(json_str):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\' and in_string:
                escape_next = True
                continue
            
            if char == '"':
                if not in_string:
                    string_start = i
                in_string = not in_string
                continue
            
            # Check for unescaped control characters in strings
            if in_string and char in ['\n', '\r', '\t']:
                # Check if it's escaped
                if i == 0 or json_str[i-1] != '\\':
                    errors.append(JSONSyntaxError(
                        "unescaped_control_char",
                        i,
                        f"Unescaped control character in string: {repr(char)}",
                        self._get_context(json_str, i, 30, 30)
                    ))
        
        return errors
    
    def _check_property_quotes(self, json_str: str) -> List[JSONSyntaxError]:
        """Check for missing opening quotes in property names."""
        errors = []
        # Pattern: property_name": (missing opening quote)
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*":'
        
        for match in re.finditer(pattern, json_str):
            # Check if this is actually inside a string (false positive)
            pos = match.start()
            # Simple check: if there's a quote before this, it's probably fine
            before = json_str[max(0, pos - 10):pos]
            if '"' not in before:
                errors.append(JSONSyntaxError(
                    "missing_opening_quote",
                    pos,
                    f"Property name '{match.group(1)}' missing opening quote",
                    self._get_context(json_str, pos, 50, 50)
                ))
        
        return errors
    
    def _get_context(self, json_str: str, pos: int, before: int, after: int) -> str:
        """Get context around a position for error messages."""
        start = max(0, pos - before)
        end = min(len(json_str), pos + after)
        context = json_str[start:end]
        # Mark the error position
        relative_pos = pos - start
        if relative_pos >= 0:
            marked = context[:relative_pos] + " ⬅️ " + context[relative_pos:]
            return marked
        return context
    
    def get_error_summary(self, errors: List[JSONSyntaxError]) -> Dict[str, Any]:
        """Get a summary of syntax errors."""
        error_counts = {}
        for error in errors:
            error_type = error.error_type
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(errors),
            "error_types": error_counts,
            "errors": [str(e) for e in errors[:10]]  # First 10 errors
        }

