"""
Unit tests for app/graph.py.
Covers: _llm/_tool singletons, build_graph, get_graph, system prompt constants.

All external calls (LLM, Qdrant, Cohere) are mocked.
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# _llm, _tool singletons
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
    """build_graph() should return a compiled graph with a callable astream."""
    from langgraph.graph.state import CompiledStateGraph

    from app.graph import LEGAL_AGENT_SYSTEM, build_graph

    mock_graph = MagicMock(spec=CompiledStateGraph)
    mock_tool = _mock_tool()
    mock_llm = MagicMock()
    with (
        patch("app.graph._llm", return_value=mock_llm),
        patch("app.graph._tool", return_value=mock_tool),
        patch("app.graph.create_react_agent", return_value=mock_graph),
    ):
        graph = build_graph(LEGAL_AGENT_SYSTEM)
    assert isinstance(graph, CompiledStateGraph)


def test_build_graph_has_agent_and_tools_nodes():
    from app.graph import LEGAL_AGENT_SYSTEM, build_graph

    mock_graph = MagicMock()
    mock_graph.nodes = {"agent": MagicMock(), "tools": MagicMock()}
    mock_tool = _mock_tool()
    mock_llm = MagicMock()
    with (
        patch("app.graph._llm", return_value=mock_llm),
        patch("app.graph._tool", return_value=mock_tool),
        patch("app.graph.create_react_agent", return_value=mock_graph),
    ):
        graph = build_graph(LEGAL_AGENT_SYSTEM)
    node_names = set(graph.nodes)
    assert "agent" in node_names
    assert "tools" in node_names


# ---------------------------------------------------------------------------
# get_graph
# ---------------------------------------------------------------------------


def test_get_graph_returns_same_instance_for_same_prompt():
    """get_graph() must return the same instance for the same system_prompt."""
    import app.graph as graph_module
    from app.graph import LEGAL_AGENT_SYSTEM, get_graph

    original = dict(graph_module._graphs)
    try:
        graph_module._graphs.clear()
        mock_graph = MagicMock()
        with patch("app.graph.build_graph", return_value=mock_graph):
            g1 = get_graph(LEGAL_AGENT_SYSTEM)
            g2 = get_graph(LEGAL_AGENT_SYSTEM)
        assert g1 is g2
    finally:
        graph_module._graphs.clear()
        graph_module._graphs.update(original)


def test_get_graph_builds_once_per_prompt():
    """get_graph() must call build_graph exactly once per unique system_prompt."""
    import app.graph as graph_module
    from app.graph import LEGAL_AGENT_SYSTEM, get_graph

    original = dict(graph_module._graphs)
    try:
        graph_module._graphs.clear()
        mock_graph = MagicMock()
        with patch("app.graph.build_graph", return_value=mock_graph) as mock_build:
            get_graph(LEGAL_AGENT_SYSTEM)
            get_graph(LEGAL_AGENT_SYSTEM)
        mock_build.assert_called_once()
    finally:
        graph_module._graphs.clear()
        graph_module._graphs.update(original)


# ---------------------------------------------------------------------------
# System prompt constants are exported
# ---------------------------------------------------------------------------


def test_system_prompt_constants_exported():
    from app.graph import DOC_CONSULTANT_SYSTEM, LEGAL_ADVISOR_SYSTEM, LEGAL_AGENT_SYSTEM

    assert "Ethiopian law" in LEGAL_AGENT_SYSTEM
    assert "Ethiopian law" in LEGAL_ADVISOR_SYSTEM
    assert "search_user_documents" in DOC_CONSULTANT_SYSTEM
