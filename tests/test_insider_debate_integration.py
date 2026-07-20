from types import SimpleNamespace

import pytest

from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher


class CapturingLLM:
    def __init__(self):
        self.prompt = ""

    def invoke(self, prompt):
        self.prompt = prompt
        return SimpleNamespace(content="test argument")


@pytest.mark.parametrize("factory", [create_bull_researcher, create_bear_researcher])
def test_research_debate_receives_sec_insider_report(factory):
    llm = CapturingLLM()
    node = factory(llm)
    state = {
        "market_report": "market evidence",
        "sentiment_report": "sentiment evidence",
        "news_report": "news evidence",
        "fundamentals_report": "fundamental evidence",
        "insider_report": "SEC TEST EVIDENCE",
        "asset_type": "stock",
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_response": "",
            "count": 0,
        },
    }

    node(state)

    assert "Official SEC Form 4 insider activity: SEC TEST EVIDENCE" in llm.prompt
