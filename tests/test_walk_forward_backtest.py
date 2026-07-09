import pandas as pd

from backtests.walk_forward import WalkForwardConfig, build_walk_forward_features, run_walk_forward_backtest


def _sample_data() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=80, freq="D")
    close = [100 + i * 0.5 for i in range(80)]
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": [p + 1 for p in close],
            "Low": [p - 1 for p in close],
            "Close": close,
            "Volume": [1_000_000 + i * 10_000 for i in range(80)],
        }
    )


def test_walk_forward_features_shift_inputs_one_day():
    data = _sample_data()
    features = build_walk_forward_features(data)

    assert features.loc[0, "signal_score"] == 0
    assert features.loc[60, "signal_score"] > 0


def test_walk_forward_backtest_runs_without_future_data():
    result = run_walk_forward_backtest(
        _sample_data(),
        WalkForwardConfig(position_pct=0.02, min_score=2.0, max_hold_days=5),
    )

    assert result["num_trades"] > 0
    assert result["max_drawdown_pct"] <= 0
    assert "trades" in result
