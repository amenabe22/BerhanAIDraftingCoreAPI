"""Real-time JSON validator for streaming LLM responses.

Validates JSON incrementally during streaming to catch errors early,
especially missing commas and malformed structures.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationErrorType(Enum):
    """Types of JSON validation errors."""
    MISSING_COMMA = "missing_comma"
    UNCLOSED_STRING = "unclosed_string"
    UNMATCHED_BRACKET = "unmatched_bracket"
    UNESCAPED_QUOTE = "unescaped_quote"
    TRAILING_COMMA = "trailing_comma"
    INCOMPLETE_STRUCTURE = "incomplete_structure"


class JSONValidationError:
    """Represents a JSON validation error with position and context."""
    
    def __init__(
        self,
        error_type: ValidationErrorType,
        position: int,
        message: str,
        context_before: str = "",
        context_after: str = ""
    ):
        self.error_type = error_type
        self.position = position
        self.message = message
        self.context_before = context_before
        self.context_after = context_after
    
    def __str__(self):
        return f"{self.error_type.value} at position {self.position}: {self.message}"


class StreamingJSONValidator:
    """
    Validates JSON incrementally during streaming to catch errors early.
    
    Tracks:
    - Bracket/brace depth and matching
    - String boundaries and escaping
    - Comma placement between elements
    - Structural completeness
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.buffer = ""
        self.depth = 0
        self.in_string = False
        self.escape_next = False
        self.last_structural_char = None  # '}', ']', or '"' (end of string)
        self.last_structural_pos = -1
        
        # Track bracket/brace stack
        self.stack = []  # List of ('{', pos) or ('[', pos)
        
        # Track validation errors
        self.errors: List[JSONValidationError] = []
        self.warnings: List[JSONValidationError] = []
        
        # Statistics
        self.total_chars_validated = 0
        self.commas_inserted = 0
    
    def feed(self, chunk: str) -> List[JSONValidationError]:
        """
        Feed a chunk of JSON and return any validation errors found.
        
        Args:
            chunk: String chunk from LLM stream
            
        Returns:
            List of validation errors found in this chunk
        """
        if not chunk:
            return []
        
        self.buffer += chunk
        new_errors = []
        
        # Validate from where we left off
        start_pos = self.total_chars_validated
        end_pos = len(self.buffer)
        
        i = start_pos
        while i < end_pos:
            char = self.buffer[i]
            
            # Handle escape sequences
            if self.escape_next:
                self.escape_next = False
                i += 1
                continue
            
            if char == '\\' and self.in_string:
                self.escape_next = True
                i += 1
                continue
            
            # Track string boundaries
            if char == '"':
                if not self.in_string:
                    # Starting a string - check for missing comma before it
                    error = self._check_missing_comma_before_string(i)
                    if error:
                        new_errors.append(error)
                        self.warnings.append(error)
                
                self.in_string = not self.in_string
                
                if not self.in_string:
                    # String ended - mark as structural
                    self.last_structural_char = '"'
                    self.last_structural_pos = i
                
                i += 1
                continue
            
            # Only validate structure outside strings
            if not self.in_string:
                # Check for missing comma before structural elements
                if char in ['{', '[', '"']:
                    error = self._check_missing_comma_before_structure(i, char)
                    if error:
                        new_errors.append(error)
                        self.warnings.append(error)
                
                # Track brackets/braces
                if char == '{':
                    self.stack.append(('{', i))
                    self.depth += 1
                    self.last_structural_char = None
                    self.last_structural_pos = -1
                    i += 1
                    continue
                
                elif char == '}':
                    if not self.stack or self.stack[-1][0] != '{':
                        error = JSONValidationError(
                            ValidationErrorType.UNMATCHED_BRACKET,
                            i,
                            "Unmatched closing brace '}'",
                            self._get_context(i, 30, 30)
                        )
                        new_errors.append(error)
                    else:
                        self.stack.pop()
                    
                    self.depth -= 1
                    self.last_structural_char = '}'
                    self.last_structural_pos = i
                    i += 1
                    continue
                
                elif char == '[':
                    self.stack.append(('[', i))
                    self.depth += 1
                    self.last_structural_char = None
                    self.last_structural_pos = -1
                    i += 1
                    continue
                
                elif char == ']':
                    if not self.stack or self.stack[-1][0] != '[':
                        error = JSONValidationError(
                            ValidationErrorType.UNMATCHED_BRACKET,
                            i,
                            "Unmatched closing bracket ']'",
                            self._get_context(i, 30, 30)
                        )
                        new_errors.append(error)
                    else:
                        self.stack.pop()
                    
                    self.depth -= 1
                    self.last_structural_char = ']'
                    self.last_structural_pos = i
                    i += 1
                    continue
                
                elif char == ',':
                    # Comma found - reset structural tracking
                    self.last_structural_char = None
                    self.last_structural_pos = -1
                    i += 1
                    continue
                
                elif char.isspace():
                    # Whitespace - skip
                    i += 1
                    continue
            
            i += 1
        
        self.total_chars_validated = end_pos
        self.errors.extend(new_errors)
        
        return new_errors
    
    def _check_missing_comma_before_string(self, pos: int) -> Optional[JSONValidationError]:
        """
        Check if a comma is missing before a string (property name).
        
        Pattern: } "key": or ] "key": or "value" "key":
        """
        if self.last_structural_char not in ['}', ']', '"']:
            return None
        
        if self.last_structural_pos < 0:
            return None
        
        # Check if there's a comma between last structural char and this quote
        between = self.buffer[self.last_structural_pos + 1:pos]
        between_stripped = between.strip()
        
        # If there's no comma and it's mostly whitespace, we likely need one
        if ',' not in between_stripped and (not between_stripped or between_stripped.isspace()):
            # Check if this looks like a property name (followed by ":")
            lookahead = pos + 1
            while lookahead < len(self.buffer) and self.buffer[lookahead].isspace():
                lookahead += 1
            
            if lookahead < len(self.buffer) and self.buffer[lookahead] == ':':
                # This is a property name - we need a comma
                context_before = self._get_context(pos, 50, 20)
                context_after = self._get_context(pos, 20, 50)
                
                return JSONValidationError(
                    ValidationErrorType.MISSING_COMMA,
                    pos,
                    f"Missing comma after {self.last_structural_char} before property name",
                    context_before,
                    context_after
                )
        
        return None
    
    def _check_missing_comma_before_structure(self, pos: int, char: str) -> Optional[JSONValidationError]:
        """
        Check if a comma is missing before a structural element ({, [, ").
        
        Pattern: } { or ] [ or } [ or ] {
        """
        if self.last_structural_char not in ['}', ']']:
            return None
        
        if self.last_structural_pos < 0:
            return None
        
        # Check if there's a comma between last structural char and this one
        between = self.buffer[self.last_structural_pos + 1:pos]
        between_stripped = between.strip()
        
        # If there's no comma and it's mostly whitespace, we likely need one
        if ',' not in between_stripped and (not between_stripped or between_stripped.isspace()):
            context_before = self._get_context(pos, 50, 20)
            context_after = self._get_context(pos, 20, 50)
            
            return JSONValidationError(
                ValidationErrorType.MISSING_COMMA,
                pos,
                f"Missing comma after {self.last_structural_char} before {char}",
                context_before,
                context_after
            )
        
        return None
    
    def _get_context(self, pos: int, before: int, after: int) -> str:
        """Get context around a position for error messages."""
        start = max(0, pos - before)
        end = min(len(self.buffer), pos + after)
        return self.buffer[start:end]
    
    def get_errors(self) -> List[JSONValidationError]:
        """Get all validation errors found so far."""
        return self.errors.copy()
    
    def get_warnings(self) -> List[JSONValidationError]:
        """Get all validation warnings (potential issues)."""
        return self.warnings.copy()
    
    def has_errors(self) -> bool:
        """Check if any errors were found."""
        return len(self.errors) > 0
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of validation errors."""
        error_counts = {}
        for error in self.errors:
            error_type = error.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.errors),
            "error_types": error_counts,
            "total_warnings": len(self.warnings),
            "chars_validated": self.total_chars_validated,
            "commas_inserted": self.commas_inserted,
        }
    
    def reset(self):
        """Reset validator state for new validation."""
        self.buffer = ""
        self.depth = 0
        self.in_string = False
        self.escape_next = False
        self.last_structural_char = None
        self.last_structural_pos = -1
        self.stack = []
        self.errors = []
        self.warnings = []
        self.total_chars_validated = 0
        self.commas_inserted = 0
        logger.debug("🔄 JSON validator reset")

