from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.scheduler.runner import (
    _exit_reason,
    _resolve_mode,
    _select_target_tickers,
    _shadow_fill_price,
    _trading_days_between,
)


def test_no_manual_tickers_uses_watchlist():
    targets, skipped = _select_target_tickers(None, ["AAPL", "MSFT"])

    assert targets == ["AAPL", "MSFT"]
    assert skipped == []


def test_manual_tickers_are_filtered_to_watchlist_by_default():
    targets, skipped = _select_target_tickers(["aapl", "tsla"], ["AAPL"])

    assert targets == ["AAPL"]
    assert skipped == ["TSLA"]


def test_manual_ticker_bypass_requires_explicit_flag():
    targets, skipped = _select_target_tickers(
        ["aapl", "tsla"],
        ["MSFT"],
        allow_manual_tickers=True,
    )

    assert targets == ["AAPL", "TSLA"]
    assert skipped == []


def test_resolve_mode_preserves_old_dry_run_flag():
    assert _resolve_mode(None, True) == "dry-run"
    assert _resolve_mode(None, False) == "live"
    assert _resolve_mode("shadow", True) == "shadow"


def test_resolve_mode_rejects_unknown_mode():
    with pytest.raises(ValueError):
        _resolve_mode("paper", True)


def test_exit_reason_prefers_stop_loss_before_other_triggers():
    old_entry = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()

    assert _exit_reason(100, 92, old_entry, 0.9) == "stop_loss"


def test_exit_reason_take_profit_time_and_manipulation():
    old_entry = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    fresh_entry = datetime.now(timezone.utc).isoformat()

    assert _exit_reason(100, 112, fresh_entry, 0) == "take_profit"
    assert _exit_reason(100, 101, old_entry, 0) == "time_exit"
    assert _exit_reason(100, 101, fresh_entry, 0.9) == "manipulation_spike"
    assert _exit_reason(100, 101, fresh_entry, 0.1) is None


def test_shadow_fill_price_simulates_slippage():
    assert _shadow_fill_price("buy", 100) > 100
    assert _shadow_fill_price("sell", 100) < 100


def test_trading_days_between_ignores_weekend():
    friday = datetime(2026, 5, 22, tzinfo=timezone.utc)
    tuesday = datetime(2026, 5, 26, tzinfo=timezone.utc)

    assert _trading_days_between(friday.isoformat(), now=tuesday) == 2
