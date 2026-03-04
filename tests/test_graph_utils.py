"""
Unit tests for _trim() and _is_new_thread() in graph.py / main.py.
"""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph import _MAX_MESSAGES, LEGAL_AGENT_SYSTEM, _trim
from app.main import _is_new_thread

# ---------------------------------------------------------------------------
# _trim tests
# ---------------------------------------------------------------------------


def _sys() -> SystemMessage:
    return SystemMessage(content=LEGAL_AGENT_SYSTEM)


def _human(n: int = 1) -> HumanMessage:
    return HumanMessage(content=f"human {n}")


def _ai(n: int = 1) -> AIMessage:
    return AIMessage(content=f"ai {n}")


def test_trim_preserves_system_message_first():
    msgs = [_sys(), _human(1), _ai(1)]
    result = _trim(msgs, _sys())
    assert isinstance(result[0], SystemMessage)


def test_trim_below_limit_unchanged():
    msgs = [_sys()] + [_human(i) for i in range(5)] + [_ai(i) for i in range(5)]
    result = _trim(msgs, _sys())
    # System + 10 non-system messages, well below _MAX_MESSAGES=20
    assert len(result) == 11


def test_trim_above_limit_cuts_to_max():
    # Build 25 non-system messages (exceeds _MAX_MESSAGES=20)
    non_sys = []
    for i in range(13):
        non_sys.append(_human(i))
        non_sys.append(_ai(i))
    # 26 messages total; trim should cut to _MAX_MESSAGES=20 starting on human
    msgs = [_sys()] + non_sys
    result = _trim(msgs, _sys())
    # First message must be the system message
    assert isinstance(result[0], SystemMessage)
    # Must not exceed system + _MAX_MESSAGES
    assert len(result) <= _MAX_MESSAGES + 1


def test_trim_strips_extra_system_messages():
    """If state somehow has two system messages, trim should not include them."""
    msgs = [_sys(), _sys(), _human(1), _ai(1)]
    result = _trim(msgs, _sys())
    system_count = sum(1 for m in result if isinstance(m, SystemMessage))
    assert system_count == 1


def test_trim_result_starts_with_injected_system():
    custom_sys = SystemMessage(content="custom persona")
    msgs = [_human(1), _ai(1)]
    result = _trim(msgs, custom_sys)
    assert result[0].content == "custom persona"


# ---------------------------------------------------------------------------
# _is_new_thread tests
# ---------------------------------------------------------------------------


def test_is_new_thread_true_when_no_messages():
    graph = MagicMock()
    state = MagicMock()
    state.values = {"messages": []}
    graph.get_state.return_value = state
    assert _is_new_thread(graph, "some-thread-id") is True


def test_is_new_thread_false_when_messages_exist():
    graph = MagicMock()
    state = MagicMock()
    state.values = {"messages": [HumanMessage(content="hi")]}
    graph.get_state.return_value = state
    assert _is_new_thread(graph, "some-thread-id") is False


def test_is_new_thread_true_on_exception():
    graph = MagicMock()
    graph.get_state.side_effect = Exception("checkpointer error")
    assert _is_new_thread(graph, "bad-thread") is True


def test_is_new_thread_true_when_state_is_none():
    graph = MagicMock()
    graph.get_state.return_value = None
    assert _is_new_thread(graph, "some-id") is True
