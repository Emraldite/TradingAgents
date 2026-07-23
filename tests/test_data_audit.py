import pandas as pd

from backtests.data_audit import (
    audit_feature_lookahead,
    audit_market_data,
    trim_incomplete_trailing_rows,
)
from backtests.walk_forward import build_walk_forward_features


def _market_data(rows: int = 80) -> pd.DataFrame:
    close = pd.Series([100 + index * 0.5 for index in range(rows)], dtype="float64")
    return pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=rows, freq="D"),
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000,
        }
    )


def test_clean_market_data_and_walk_forward_features_pass_audit():
    data = _market_data()

    assert audit_market_data(data)["ok"] is True
    assert audit_feature_lookahead(data, build_walk_forward_features)["ok"] is True


def test_market_data_audit_reports_duplicate_and_invalid_range():
    data = _market_data(8)
    data.loc[1, "Date"] = data.loc[0, "Date"]
    data.loc[2, "High"] = data.loc[2, "Low"] - 1

    result = audit_market_data(data)
    codes = {issue["code"] for issue in result["issues"]}

    assert result["ok"] is False
    assert {"duplicate_dates", "invalid_ohlc_range"} <= codes


def test_only_short_incomplete_suffix_is_trimmed():
    data = _market_data(8)
    data.loc[7, "Close"] = float("nan")

    cleaned, removed = trim_incomplete_trailing_rows(data)

    assert removed == 1
    assert len(cleaned) == 7
    assert audit_market_data(cleaned)["ok"] is True


def test_incomplete_interior_row_is_never_silently_trimmed():
    data = _market_data(8)
    data.loc[3, "Close"] = float("nan")

    cleaned, removed = trim_incomplete_trailing_rows(data)

    assert removed == 0
    assert len(cleaned) == 8
    assert audit_market_data(cleaned)["ok"] is False


def test_lookahead_audit_finds_future_dependent_feature():
    def bad_builder(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["future_close"] = result["Close"].shift(-1)
        return result

    result = audit_feature_lookahead(_market_data(), bad_builder)

    assert result["ok"] is False
    assert result["issues"][0]["code"] == "future_dependent_features"
