import pandas as pd
import pytest

from backtests.strategy_replay import (
    ReplayConfig,
    replay_graph_decisions,
    summarize_replay_results,
)


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]),
            "Open": [100.0, 100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 115.0, 103.0],
            "Low": [99.0, 99.0, 100.0, 101.0],
            "Close": [100.0, 101.0, 110.0, 102.0],
            "Volume": [1_000_000] * 4,
        }
    )


def test_graph_decision_executes_on_next_bar_and_uses_shared_take_profit():
    result = replay_graph_decisions(
        _prices(),
        [{"trade_date": "2026-01-05", "rating": "Buy"}],
        config=ReplayConfig(initial_cash=10_000, scorecard_size_cap=0.02),
    )

    assert result["num_trades"] == 1
    assert result["trades"][0]["entry_date"] == "2026-01-06"
    assert result["trades"][0]["exit_date"] == "2026-01-07"
    assert result["trades"][0]["reason"] == "take_profit"
    assert "benchmark_return_pct" in result
    assert result["alpha_pct"] == pytest.approx(
        result["total_return_pct"] - result["benchmark_return_pct"], abs=0.02
    )


def test_hold_decision_never_opens_position():
    result = replay_graph_decisions(
        _prices(),
        [{"trade_date": "2026-01-05", "rating": "Hold"}],
    )

    assert result["num_trades"] == 0
    assert result["final_equity"] == 10_000


def test_cross_ticker_summary_uses_average_alpha_and_worst_drawdown():
    summary = summarize_replay_results(
        {
            "AAPL": {
                "num_trades": 20,
                "total_return_pct": 8,
                "benchmark_return_pct": 5,
                "alpha_pct": 3,
                "max_drawdown_pct": -4,
                "start_date": "2020-01-01",
                "end_date": "2024-01-01",
            },
            "MSFT": {
                "num_trades": 12,
                "total_return_pct": 4,
                "benchmark_return_pct": 3,
                "alpha_pct": 1,
                "max_drawdown_pct": -7,
                "start_date": "2021-01-01",
                "end_date": "2025-01-01",
            },
        }
    )

    assert summary["ticker_count"] == 2
    assert summary["num_trades"] == 32
    assert summary["alpha_pct"] == 2
    assert summary["max_drawdown_pct"] == -7
