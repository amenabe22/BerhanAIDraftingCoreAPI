"""JSON schemas for generation LLM responses."""

DOCUMENT_SCHEMA = {
    "type": "object",
    "required": ["metadata", "pages"],
    "properties": {
        "metadata": {
            "type": "object",
            "required": ["document_type", "title", "language", "created_at"],
            "properties": {
                "document_type": {"type": "string"},
                "title": {"type": "string"},
                "language": {"type": "string", "enum": ["en", "am", "om"]},
                "created_at": {"type": "string", "format": "date-time"},
            },
        },
        "pages": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "page_number", "sections"],
                "properties": {
                    "id": {"type": "string"},
                    "page_number": {"type": "integer", "minimum": 1},
                    "sections": {"type": "array"},
                },
            },
        },
    },
}

ANALYSIS_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["document_type", "language", "questions", "ready_to_generate", "response_message"],
    "properties": {
        "document_type": {"type": "string"},
        "language": {"type": "string", "enum": ["en", "am", "om"]},
        "extracted_info": {"type": "object", "additionalProperties": True},
        "questions": {"type": "array", "items": {"type": "string"}},
        "ready_to_generate": {"type": "boolean"},
        "response_message": {"type": "string"},
    },
}

CLARIFICATION_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["updated_requirements", "ready_to_generate", "response_message", "questions"],
    "properties": {
        "updated_requirements": {"type": "object", "additionalProperties": True},
        "ready_to_generate": {"type": "boolean"},
        "response_message": {"type": "string"},
        "questions": {"type": "array", "items": {"type": "string"}},
    },
}
