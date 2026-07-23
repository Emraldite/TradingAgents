from unittest.mock import MagicMock

from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup


def test_active_graph_skips_repetitive_debate_and_trader_nodes():
    setup = GraphSetup(
        MagicMock(),
        MagicMock(),
        {"insider": MagicMock()},
        ConditionalLogic(),
    )

    graph = setup.setup_graph(["insider"])

    assert "Portfolio Manager" in graph.nodes
    assert "Bull Researcher" not in graph.nodes
    assert "Trader" not in graph.nodes
    assert "Aggressive Analyst" not in graph.nodes
