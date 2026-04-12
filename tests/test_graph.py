"""
Unit tests for app/graph.py.
Covers: _should_continue, _agent_node, _llm/_tool/_llm_with_tools (via mocks),
build_graph, get_graph.

All external calls (LLM, Qdrant, Cohere) are mocked.
"""

import uuid
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(*msgs):
    """Build a minimal agent state dict."""
    return {"messages": list(msgs)}


def _ai_with_tool_calls(**kwargs):
    """Return an AIMessage that looks like it has pending tool calls."""
    msg = MagicMock(spec=AIMessage)
    msg.tool_calls = [{"name": "search_legal_knowledge", "args": {"query": "contracts"}}]
    return msg


def _ai_no_tool_calls():
    msg = MagicMock(spec=AIMessage)
    msg.tool_calls = []
    return msg


# ---------------------------------------------------------------------------
# _should_continue
# ---------------------------------------------------------------------------


def test_should_continue_returns_tools_when_tool_calls_present():
    from app.graph import _should_continue

    state = _state(_ai_with_tool_calls())
    assert _should_continue(state) == "tools"


def test_should_continue_returns_end_when_no_tool_calls():
    from app.graph import _should_continue

    state = _state(_ai_no_tool_calls())
    assert _should_continue(state) == "__end__"


def test_should_continue_returns_end_for_plain_ai_message():
    from app.graph import _should_continue

    state = _state(AIMessage(content="Here is your answer."))
    assert _should_continue(state) == "__end__"


def test_should_continue_returns_end_for_human_message():
    from app.graph import _should_continue

    state = _state(HumanMessage(content="What is contract law?"))
    assert _should_continue(state) == "__end__"


# ---------------------------------------------------------------------------
# _agent_node  (LLM is mocked)
# ---------------------------------------------------------------------------


def _patch_llm_with_tools(response):
    """Context manager: patch _llm_with_tools() so .invoke() returns *response*."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = response
    mock_llm.stream.return_value = []
    return patch("app.graph._llm_with_tools", return_value=mock_llm)


def test_agent_node_returns_messages_key():
    from app.graph import _agent_node

    fake_response = AIMessage(content="Answer without tools.")
    with _patch_llm_with_tools(fake_response):
        result = _agent_node(_state(SystemMessage(content="sys"), HumanMessage(content="hi")))
    assert "messages" in result
    assert result["messages"] == [fake_response]


def test_agent_node_uses_existing_system_message():
    """_agent_node must not inject a second system message when one already exists."""
    from app.graph import _agent_node

    custom_sys = SystemMessage(content="Custom legal persona")
    fake_response = AIMessage(content="Advice here.")
    with _patch_llm_with_tools(fake_response) as mock_factory:
        _agent_node(_state(custom_sys, HumanMessage(content="question")))
        call_args = mock_factory.return_value.invoke.call_args[0][0]
    # First message in the ainvoke call must be the custom system message
    assert isinstance(call_args[0], SystemMessage)
    assert call_args[0].content == "Custom legal persona"


def test_agent_node_injects_default_system_when_none_present():
    """When no system message exists in state, fall back to LEGAL_AGENT_SYSTEM."""
    from app.graph import LEGAL_AGENT_SYSTEM, _agent_node

    fake_response = AIMessage(content="Answer.")
    with _patch_llm_with_tools(fake_response) as mock_factory:
        _agent_node(_state(HumanMessage(content="tell me about property law")))
        call_args = mock_factory.return_value.invoke.call_args[0][0]
    assert isinstance(call_args[0], SystemMessage)
    assert call_args[0].content == LEGAL_AGENT_SYSTEM


def test_agent_node_logs_tool_calls_when_present(caplog):
    """_agent_node should log 'tool_calls' event when the LLM returns tool calls."""
    import logging

    from app.graph import _agent_node

    tc_response = MagicMock(spec=AIMessage)
    tc_response.tool_calls = [{"name": "search_legal_knowledge", "args": {"query": "family law"}}]
    with _patch_llm_with_tools(tc_response):
        with caplog.at_level(logging.INFO, logger="graph"):
            _agent_node(_state(HumanMessage(content="family law question")))
    # The log output (JSON) should mention the event
    assert any("tool_calls" in r.message or "retrieve" in str(r.__dict__) for r in caplog.records)


def test_agent_node_tool_call_args_as_json_string():
    """_tc_repr should safely parse JSON-string args."""
    import json as _json

    from app.graph import _agent_node

    args_dict = {"query": "inheritance rights"}
    tc_response = MagicMock(spec=AIMessage)
    tc_response.tool_calls = [{"name": "search_legal_knowledge", "args": _json.dumps(args_dict)}]
    with _patch_llm_with_tools(tc_response):
        # Should not raise
        _agent_node(_state(HumanMessage(content="inheritance question")))


def test_agent_node_tool_call_args_invalid_json_string():
    """_tc_repr must not crash on a non-JSON string in args."""
    from app.graph import _agent_node

    tc_response = MagicMock(spec=AIMessage)
    tc_response.tool_calls = [{"name": "search_legal_knowledge", "args": "not valid json {{"}]
    with _patch_llm_with_tools(tc_response):
        _agent_node(_state(HumanMessage(content="question")))


def test_compiled_graph_emits_multiple_custom_token_events():
    """Regression guard: compiled graph must expose real custom token chunks, not only a final message."""
    from app.graph import build_graph

    mock_tool = _mock_tool()
    mock_llm = MagicMock()
    mock_llm.stream.return_value = [
        AIMessageChunk(content="Hello "),
        AIMessageChunk(content="world"),
    ]
    mock_llm.invoke.return_value = AIMessage(content="Hello world")

    with (
        patch("app.graph._tool", return_value=mock_tool),
        patch("app.graph._llm_with_tools", return_value=mock_llm),
        patch("app.graph.ToolNode"),
    ):
        graph = build_graph()

        events = []
        for mode, payload in graph.stream(
            {"messages": [SystemMessage(content="sys"), HumanMessage(content="hi")]},
            stream_mode=["custom", "messages"],
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        ):
            events.append((mode, payload))

    custom_tokens = [payload["content"] for mode, payload in events if mode == "custom"]
    assert custom_tokens == ["Hello ", "world"]


# ---------------------------------------------------------------------------
# _llm, _tool, _llm_with_tools singletons
# ---------------------------------------------------------------------------


def test_llm_singleton_returns_same_instance():
    """_llm() should return the same cached object on repeated calls."""
    from app.graph import _llm

    with patch("app.graph.ChatOpenAI") as MockLLM:
        MockLLM.return_value = MagicMock()
        _llm.cache_clear()
        a = _llm()
        b = _llm()
        assert a is b
        _llm.cache_clear()


def test_tool_singleton_returns_same_instance():
    from app.graph import _tool

    mock_tool = MagicMock()
    with patch("app.graph.get_retriever_tool", return_value=mock_tool):
        _tool.cache_clear()
        a = _tool()
        b = _tool()
        assert a is b
        _tool.cache_clear()


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


def _mock_tool():
    t = MagicMock()
    t.name = "search_legal_knowledge"
    return t


def test_build_graph_returns_compiled_graph():
    """build_graph() should return a runnable CompiledStateGraph."""
    from langgraph.graph.state import CompiledStateGraph

    from app.graph import build_graph

    mock_tool = _mock_tool()
    with (
        patch("app.graph._tool", return_value=mock_tool),
        patch("app.graph._llm_with_tools", return_value=MagicMock()),
        patch("app.graph.ToolNode"),
    ):
        graph = build_graph()
    assert isinstance(graph, CompiledStateGraph)


def test_build_graph_has_agent_and_tools_nodes():
    from app.graph import build_graph

    mock_tool = _mock_tool()
    with (
        patch("app.graph._tool", return_value=mock_tool),
        patch("app.graph._llm_with_tools", return_value=MagicMock()),
        patch("app.graph.ToolNode"),
    ):
        graph = build_graph()
    node_names = set(graph.nodes)
    assert "agent" in node_names
    assert "tools" in node_names


# ---------------------------------------------------------------------------
# get_graph
# ---------------------------------------------------------------------------


def test_get_graph_returns_singleton():
    """get_graph() must return the same instance on second call."""
    import app.graph as graph_module
    from app.graph import get_graph

    original = graph_module._graph
    try:
        graph_module._graph = None
        mock_tool = _mock_tool()
        with (
            patch("app.graph._tool", return_value=mock_tool),
            patch("app.graph._llm_with_tools", return_value=MagicMock()),
            patch("app.graph.ToolNode"),
        ):
            g1 = get_graph()
            g2 = get_graph()
        assert g1 is g2
    finally:
        graph_module._graph = original


def test_get_graph_builds_when_none():
    """get_graph() must call build_graph exactly once when _graph is None."""
    import app.graph as graph_module
    from app.graph import get_graph

    original = graph_module._graph
    try:
        graph_module._graph = None
        mock_graph = MagicMock()
        with patch("app.graph.build_graph", return_value=mock_graph) as mock_build:
            get_graph()
            get_graph()
        mock_build.assert_called_once()
    finally:
        graph_module._graph = original
