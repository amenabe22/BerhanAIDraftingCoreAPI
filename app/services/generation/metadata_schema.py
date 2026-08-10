"""Load / validate Berhan drafting metadata against the canonical JSON Schema."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / "drafting_metadata.schema.json"
)


@lru_cache(maxsize=1)
def load_drafting_metadata_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_drafting_metadata(data: Any) -> None:
    """Raise ``jsonschema.ValidationError`` if ``data`` is not valid metadata."""
    Draft202012Validator(load_drafting_metadata_schema()).validate(data)
