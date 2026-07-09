from tradingagents.graph.setup import DEBATE_PATH_MAP, RISK_ANALYSIS_PATH_MAP


def test_shared_debate_path_map_contains_all_router_targets():
    assert set(DEBATE_PATH_MAP) == {
        "Bull Researcher",
        "Bear Researcher",
        "Research Manager",
    }


def test_shared_risk_path_map_contains_all_router_targets():
    assert set(RISK_ANALYSIS_PATH_MAP) == {
        "Aggressive Analyst",
        "Conservative Analyst",
        "Neutral Analyst",
        "Portfolio Manager",
    }
