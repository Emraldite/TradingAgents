from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.scheduler.runner import (
    _exit_reason,
    _broker_reference_price,
    _manipulation_risk,
    _market_data_stale,
    _position_pct_for_rating,
    _portfolio_value_for_mode,
    _resolve_mode,
    _select_target_tickers,
    _shadow_fill_price,
    _trading_days_between,
    _validate_broker_mode,
)


def test_manipulation_risk_uses_detector_field():
    assert _manipulation_risk({"manipulation_risk": 7.5}) == 7.5
    assert _manipulation_risk({"manipulation_score": 4}) == 4.0


def test_market_data_staleness_is_explicit():
    import pandas as pd

    fresh = pd.DataFrame(
        {"Close": [100]}, index=[pd.Timestamp.now(tz="UTC").normalize()]
    )
    stale = pd.DataFrame(
        {"Close": [100]}, index=[pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=10)]
    )

    assert _market_data_stale(fresh) is False
    assert _market_data_stale(stale) is True


def test_broker_reference_uses_fresh_iex_ask_and_rejects_wide_spread(monkeypatch):
    from tradingagents.scheduler import runner

    monkeypatch.setattr(
        runner.executor,
        "get_stock_snapshot_checked",
        lambda ticker: (
            {
                "bid": 99.9,
                "ask": 100.1,
                "last": 100,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            None,
        ),
    )
    assert _broker_reference_price("AAPL", "buy") == (100.1, None)

    monkeypatch.setattr(
        runner.executor,
        "get_stock_snapshot_checked",
        lambda ticker: (
            {
                "bid": 90,
                "ask": 110,
                "last": 100,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            None,
        ),
    )
    price, error = _broker_reference_price("AAPL", "buy")
    assert price == 0
    assert "spread" in error


def test_broker_modes_fail_closed_without_account_value():
    assert _portfolio_value_for_mode("dry-run", None) == 10_000.0
    assert _portfolio_value_for_mode("shadow", None) is None
    assert _portfolio_value_for_mode("live", None) is None
    assert _portfolio_value_for_mode("live", {"portfolio_value": 25_000}) == 25_000.0


def test_no_manual_tickers_uses_watchlist():
    targets, skipped = _select_target_tickers(None, ["AAPL", "MSFT"])

    assert targets == ["AAPL", "MSFT"]
    assert skipped == []


def test_manual_tickers_are_filtered_to_watchlist_by_default():
    targets, skipped = _select_target_tickers(
        ["aapl", "tsla"],
        ["AAPL"],
        allow_manual_tickers=False,
    )

    assert targets == ["AAPL"]
    assert skipped == ["TSLA"]


def test_manual_ticker_bypass_is_default():
    targets, skipped = _select_target_tickers(
        ["aapl", "tsla"],
        ["MSFT"],
    )

    assert targets == ["AAPL", "TSLA"]
    assert skipped == []


def test_resolve_mode_preserves_old_dry_run_flag():
    assert _resolve_mode(None, True) == "dry-run"
    assert _resolve_mode(None, False) == "paper"
    assert _resolve_mode("shadow", True) == "shadow"
    assert _resolve_mode("live", True) == "paper"


def test_overweight_is_stronger_than_buy():
    assert _position_pct_for_rating("Overweight") > _position_pct_for_rating("Buy")


def test_paper_mode_refuses_real_endpoint(monkeypatch):
    from tradingagents.scheduler import runner

    monkeypatch.setattr(runner.executor, "is_paper_endpoint", lambda: False)

    error = _validate_broker_mode(
        "paper",
        {
            "account_id": "acct",
            "account_blocked": False,
            "trading_blocked": False,
            "trade_suspended_by_user": False,
        },
        None,
    )

    assert "refuses" in error


def test_real_mode_requires_every_unlock_gate(monkeypatch):
    from tradingagents.scheduler import runner

    monkeypatch.setattr(runner.executor, "is_paper_endpoint", lambda: False)
    monkeypatch.setitem(runner.DEFAULT_CONFIG, "allow_real_money", False)

    error = _validate_broker_mode(
        "real",
        {
            "account_id": "real-account",
            "account_blocked": False,
            "trading_blocked": False,
            "trade_suspended_by_user": False,
        },
        "ENABLE REAL MONEY",
    )

    assert "ALLOW_REAL_MONEY" in error


def test_real_mode_accepts_matching_approved_release_report(monkeypatch, tmp_path):
    import json

    from tradingagents.scheduler import runner
    from tradingagents.risk.release_gate import (
        REQUIRED_RELEASE_CHECKS,
        release_strategy_config,
        strategy_fingerprint,
    )

    report = tmp_path / "release.json"
    monkeypatch.setattr(runner.executor, "is_paper_endpoint", lambda: False)
    monkeypatch.setitem(runner.DEFAULT_CONFIG, "allow_real_money", True)
    monkeypatch.setitem(runner.DEFAULT_CONFIG, "expected_real_account_id", "real-account")
    monkeypatch.setitem(runner.DEFAULT_CONFIG, "max_real_money_notional", 25.0)
    monkeypatch.setitem(runner.DEFAULT_CONFIG, "real_money_validation_report", str(report))
    strategy_key = runner._scorecard_strategy_key()
    fingerprint = strategy_fingerprint(
        strategy_key,
        release_strategy_config(runner.DEFAULT_CONFIG),
    )
    report.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "approved": True,
                "checks": {name: True for name in REQUIRED_RELEASE_CHECKS},
                "account_id": "real-account",
                "strategy_key": strategy_key,
                "strategy_fingerprint": fingerprint,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    error = _validate_broker_mode(
        "real",
        {
            "account_id": "real-account",
            "account_blocked": False,
            "trading_blocked": False,
            "trade_suspended_by_user": False,
        },
        "ENABLE REAL MONEY",
    )

    assert error is None


def test_resolve_mode_rejects_unknown_mode():
    with pytest.raises(ValueError):
        _resolve_mode("margin", True)


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


def test_protective_order_rejection_is_audited_as_critical(monkeypatch):
    from tradingagents.scheduler import runner

    events = []
    fake_position = {
        "id": 7,
        "ticker": "AAPL",
        "broker_quantity": 1,
        "entry_price": 100,
        "entry_date": "2026-07-10",
    }
    monkeypatch.setattr(
        runner.state_store,
        "positions_needing_stop_orders",
        lambda: [fake_position],
    )
    monkeypatch.setattr(
        runner.state_store,
        "record_health_event",
        lambda severity, component, message, context=None: events.append(
            (severity, component, message)
        ),
    )
    monkeypatch.setattr(runner.executor, "execute_oco_sell", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.alerts, "critical", lambda *args, **kwargs: None)
    decisions = []

    created = runner._ensure_native_stop_orders(decisions, "paper")

    assert created == 0
    assert decisions[0]["decision"] == "native-stop-failed"
    assert events[0][0:2] == ("critical", "protective_orders")
