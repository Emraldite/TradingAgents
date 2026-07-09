import pandas as pd

from tradingagents.scheduler import runner


def _price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "Volume": [1_000_000, 1_100_000],
        }
    )


def test_run_cycle_skips_when_graph_rating_is_hold(monkeypatch):
    monkeypatch.setattr(runner, "get_conviction_watchlist", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(runner, "_create_analysis_graph", lambda: object())
    monkeypatch.setattr(
        runner,
        "_run_graph_analysis",
        lambda graph, ticker, trade_date: ("Hold", {"final_trade_decision": "**Rating**: Hold"}),
    )
    monkeypatch.setattr(
        runner,
        "detect_manipulation",
        lambda ticker: {"recommendation": "ALLOW"},
    )
    monkeypatch.setattr(runner.yf, "download", lambda *args, **kwargs: _price_frame())
    monkeypatch.setattr(runner.executor, "get_account_info", lambda: None)
    monkeypatch.setattr(runner.executor, "get_portfolio", lambda: [])
    monkeypatch.setattr(runner, "validate_trade", lambda **kwargs: (True, "ok"))

    summary = runner.run_cycle(tickers=["AAPL"], dry_run=True)

    assert summary["signals"] == 0
    assert summary["simulated"] == 0
    assert summary["decisions"][0]["ticker"] == "AAPL"
    assert summary["decisions"][0]["reason"] == "graph_rating=Hold"


def test_run_cycle_uses_graph_buy_rating_for_entry(monkeypatch):
    monkeypatch.setattr(runner, "get_conviction_watchlist", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(runner, "_create_analysis_graph", lambda: object())
    monkeypatch.setattr(
        runner,
        "_run_graph_analysis",
        lambda graph, ticker, trade_date: ("Buy", {"final_trade_decision": "**Rating**: Buy"}),
    )
    monkeypatch.setattr(
        runner,
        "detect_manipulation",
        lambda ticker: {"recommendation": "ALLOW"},
    )
    monkeypatch.setattr(runner.yf, "download", lambda *args, **kwargs: _price_frame())
    monkeypatch.setattr(runner.executor, "get_account_info", lambda: None)
    monkeypatch.setattr(runner.executor, "get_portfolio", lambda: [])
    monkeypatch.setattr(runner, "validate_trade", lambda **kwargs: (True, "ok"))

    summary = runner.run_cycle(tickers=["AAPL"], dry_run=True)

    assert summary["signals"] == 1
    assert summary["simulated"] == 1
    assert summary["decisions"][0]["decision"] == "buy-signal"
    assert summary["decisions"][0]["rating"] == "Buy"
    assert summary["decisions"][0]["notional"] == 200.0
