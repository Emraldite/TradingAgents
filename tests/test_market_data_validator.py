import pandas as pd

from tradingagents.dataflows import market_data_validator


def test_verified_market_snapshot_excludes_future_rows(monkeypatch):
    data = pd.DataFrame(
        {
            "Date": ["2026-01-02", "2026-01-03", "2026-01-04"],
            "Open": [10.0, 11.0, 12.0],
            "High": [11.0, 12.0, 13.0],
            "Low": [9.0, 10.0, 11.0],
            "Close": [10.5, 11.5, 12.5],
            "Volume": [1000, 1100, 1200],
        }
    )
    monkeypatch.setattr(market_data_validator, "load_ohlcv", lambda symbol, curr_date: data)

    snapshot = market_data_validator.build_verified_market_snapshot(
        "AAPL",
        "2026-01-03",
        look_back_days=5,
        indicators=("close_10_ema",),
    )

    assert "Latest trading row used: 2026-01-03" in snapshot
    assert "| Close | 11.50 |" in snapshot
    assert "2026-01-04" not in snapshot
