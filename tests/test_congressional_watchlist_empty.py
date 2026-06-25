import pandas as pd

from tradingagents.dataflows import congressional_data


def test_score_trades_empty_frame_returns_expected_columns():
    scored = congressional_data.score_trades(pd.DataFrame())

    assert scored.empty
    assert "disclosure_date" in scored.columns
    assert "conviction_score" in scored.columns


def test_get_conviction_watchlist_handles_empty_source(monkeypatch):
    monkeypatch.setattr(
        congressional_data,
        "get_congressional_trades",
        lambda **kwargs: pd.DataFrame(),
    )

    watchlist = congressional_data.get_conviction_watchlist()

    assert watchlist.empty
    assert "disclosure_date" in watchlist.columns
    assert "conviction_score" in watchlist.columns
