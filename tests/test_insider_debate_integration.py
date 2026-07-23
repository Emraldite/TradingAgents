from unittest.mock import MagicMock

import pandas as pd

from tradingagents.agents.analysts import insider_analyst


def test_no_insider_activity_skips_llm(monkeypatch):
    activity = pd.DataFrame()
    activity.attrs["data_status"] = "no_activity"
    activity.attrs["data_reason"] = "No qualifying transactions"
    monkeypatch.setattr(
        insider_analyst,
        "get_sec_insider_activity",
        lambda *args, **kwargs: activity,
    )
    llm = MagicMock()
    node = insider_analyst.create_insider_analyst(llm)

    result = node(
        {
            "company_of_interest": "NVDA",
            "trade_date": "2026-07-22",
            "messages": [],
        }
    )

    assert result["insider_report"].startswith("No qualifying")
    llm.invoke.assert_not_called()
