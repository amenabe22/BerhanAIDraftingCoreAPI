"""
Unit tests for app/logging_config.py.
Covers: _default(), JsonFormatter.format(), setup_logging(), get_logger().
"""

import json
import logging
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# _default helper
# ---------------------------------------------------------------------------


def test_default_serialises_datetime():
    from app.logging_config import _default

    dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    result = _default(dt)
    assert result == dt.isoformat()


def test_default_raises_for_unknown_type():
    from app.logging_config import _default

    class _Unserializable:
        pass

    import pytest

    with pytest.raises(TypeError, match="_Unserializable"):
        _default(_Unserializable())


# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------


def _make_record(msg: str, **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_json_formatter_returns_valid_json():
    from app.logging_config import JsonFormatter

    formatter = JsonFormatter()
    record = _make_record("hello world")
    output = formatter.format(record)
    parsed = json.loads(output)
    assert isinstance(parsed, dict)


def test_json_formatter_has_required_keys():
    from app.logging_config import JsonFormatter

    formatter = JsonFormatter()
    record = _make_record("test message")
    parsed = json.loads(formatter.format(record))
    assert parsed["message"] == "test message"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test.logger"
    assert "timestamp" in parsed


def test_json_formatter_includes_extra_fields():
    from app.logging_config import JsonFormatter

    formatter = JsonFormatter()
    record = _make_record("event", event="rag_retrieve", query="contracts")
    parsed = json.loads(formatter.format(record))
    assert parsed["event"] == "rag_retrieve"
    assert parsed["query"] == "contracts"


def test_json_formatter_skips_none_extra_values():
    from app.logging_config import JsonFormatter

    formatter = JsonFormatter()
    record = _make_record("msg", nullable_field=None)
    parsed = json.loads(formatter.format(record))
    assert "nullable_field" not in parsed


def test_json_formatter_stringifies_non_serialisable_extra():
    from app.logging_config import JsonFormatter

    formatter = JsonFormatter()

    class _Weird:
        def __repr__(self):
            return "WeirdObject()"

    record = _make_record("msg", weird=_Weird())
    parsed = json.loads(formatter.format(record))
    assert "weird" in parsed
    assert "WeirdObject()" in parsed["weird"]


def test_json_formatter_timestamp_is_utc_iso():
    from app.logging_config import JsonFormatter

    formatter = JsonFormatter()
    record = _make_record("ts check")
    parsed = json.loads(formatter.format(record))
    # Should parse without error and end with +00:00
    dt = datetime.fromisoformat(parsed["timestamp"])
    assert dt.tzinfo is not None


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


def test_setup_logging_adds_handler_once():
    from app.logging_config import JsonFormatter, setup_logging

    root = logging.getLogger()
    original_handlers = root.handlers[:]
    try:
        # Temporarily clear handlers to test fresh setup
        root.handlers = []
        setup_logging(logging.DEBUG)
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        assert root.level == logging.DEBUG
    finally:
        root.handlers = original_handlers


def test_setup_logging_does_not_duplicate_handlers():
    """Calling setup_logging twice should not add a second handler."""
    from app.logging_config import setup_logging

    root = logging.getLogger()
    original_handlers = root.handlers[:]
    try:
        root.handlers = []
        setup_logging()
        count_after_first = len(root.handlers)
        setup_logging()
        count_after_second = len(root.handlers)
        assert count_after_second == count_after_first
    finally:
        root.handlers = original_handlers


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


def test_get_logger_returns_logger_with_correct_name():
    from app.logging_config import get_logger

    logger = get_logger("my.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "my.module"
