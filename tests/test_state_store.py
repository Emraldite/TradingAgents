from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.state_store import StrategyStateStore, summarize_decisions


def test_reconcile_creates_broker_only_position_and_blocks_it(tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")

    store.reconcile_broker_state(
        positions=[
            {
                "ticker": "AAPL",
                "qty": 2.0,
                "cost_basis": 300.0,
                "market_value": 310.0,
                "unrealized_pl": 10.0,
                "unrealized_plpc": 0.03,
            }
        ],
        orders=[],
        mode="live",
    )

    assert store.active_blocked_tickers()["AAPL"] == "already held"


def test_reconcile_blocks_pending_open_order(tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")

    store.reconcile_broker_state(
        positions=[],
        orders=[
            {
                "order_id": "ord-1",
                "ticker": "MSFT",
                "side": "buy",
                "type": "market",
                "status": "partially_filled",
                "qty": 10,
                "notional": 0,
                "filled_qty": 4,
                "filled_avg_price": 100,
            }
        ],
        mode="live",
    )

    assert store.active_blocked_tickers()["MSFT"] == "pending order: partially_filled"


def test_record_sell_closes_position_and_adds_cooldown(tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")
    store.record_buy_signal("AAPL", entry_price=100, position_size=1000, mode="shadow")
    position = store.open_positions()[0]

    store.record_sell(
        position_id=position["id"],
        ticker="AAPL",
        quantity=10,
        fill_price=110,
        reason="take_profit",
        mode="shadow",
    )

    assert store.open_positions() == []
    assert store.active_blocked_tickers()["AAPL"] == "sold: take_profit"


def test_apply_split_adjusts_entry_price_once(tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")
    store.record_buy_signal("NVDA", entry_price=100, position_size=1000, mode="shadow")
    position = store.open_positions()[0]

    assert store.apply_split(position["id"], "NVDA", "2026-05-20", 4) is True
    assert store.apply_split(position["id"], "NVDA", "2026-05-20", 4) is False

    adjusted = store.open_positions()[0]
    assert adjusted["entry_price"] == 25
    assert adjusted["quantity"] == 40


def test_cycle_snapshot_can_be_replayed(tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")
    cycle_id = store.start_cycle("shadow", ["AAPL"])
    store.complete_cycle(
        cycle_id,
        decisions=[{"ticker": "AAPL", "decision": "shadow-buy"}],
        portfolio=[{"ticker": "AAPL", "qty": 1}],
    )

    cycle = store.get_cycle(cycle_id)

    assert cycle["tickers"] == ["AAPL"]
    assert cycle["decisions"][0]["decision"] == "shadow-buy"


def test_strategy_status_read_helpers(tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")
    store.record_buy_signal("AAPL", entry_price=100, position_size=1000, mode="shadow")
    cycle_id = store.start_cycle("shadow", ["AAPL", "MSFT"])
    store.complete_cycle(
        cycle_id,
        decisions=[
            {"ticker": "AAPL", "decision": "shadow-buy"},
            {"ticker": "MSFT", "decision": "skip"},
        ],
        portfolio=[],
    )

    positions = store.active_positions()
    trades = store.recent_trades(limit=5)
    cycles = store.recent_cycles(limit=5)

    assert positions[0]["ticker"] == "AAPL"
    assert trades[0]["side"] == "buy"
    assert cycles[0]["ticker_count"] == 2
    assert cycles[0]["action_summary"] == "shadow-buy:1, skip:1"


def test_stop_order_tracking(tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")
    position_id = store.record_buy_signal(
        "AAPL",
        entry_price=100,
        position_size=1000,
        mode="live",
        order={
            "order_id": "buy-1",
            "ticker": "AAPL",
            "side": "buy",
            "type": "limit",
            "status": "filled",
            "qty": 10,
            "notional": 0,
            "filled_qty": 10,
            "filled_avg_price": 100,
        },
    )

    assert store.positions_needing_stop_orders()[0]["ticker"] == "AAPL"

    store.set_stop_order(position_id, "stop-1")

    assert store.positions_needing_stop_orders() == []
    assert store.active_positions()[0]["stop_order_id"] == "stop-1"


def test_unfilled_live_order_does_not_create_position_or_trade(tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")
    order = {
        "order_id": "buy-pending",
        "client_order_id": "ta-buy-one",
        "ticker": "AAPL",
        "side": "buy",
        "type": "limit",
        "status": "accepted",
        "qty": 10,
        "limit_price": 100,
        "filled_qty": 0,
        "filled_avg_price": 0,
    }

    result = store.record_order_update(order, "live", reason="graph_entry")

    assert result["fill_delta"] == 0
    assert store.active_positions() == []
    assert store.recent_trades() == []
    assert store.reserved_open_position_count() == 1


def test_order_updates_apply_only_incremental_fill_quantity(tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")
    order = {
        "order_id": "buy-partial",
        "ticker": "AAPL",
        "side": "buy",
        "type": "limit",
        "status": "partially_filled",
        "qty": 10,
        "limit_price": 100,
        "filled_qty": 4,
        "filled_avg_price": 100,
    }

    first = store.record_order_update(order, "live")
    repeated = store.record_order_update(order, "live")
    order.update(status="filled", filled_qty=10, filled_avg_price=101)
    final = store.record_order_update(order, "live")

    assert first["fill_delta"] == 4
    assert repeated["fill_delta"] == 0
    assert final["fill_delta"] == 6
    assert store.open_positions()[0]["quantity"] == 10
    assert [trade["quantity"] for trade in reversed(store.recent_trades())] == [4, 6]


def test_terminal_order_and_cumulative_fill_cannot_regress(tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")
    filled = {
        "order_id": "immutable-fill",
        "ticker": "AAPL",
        "side": "buy",
        "type": "limit",
        "status": "filled",
        "qty": 2,
        "filled_qty": 2,
        "filled_avg_price": 100,
    }
    store.record_order_update(filled, "paper")

    with pytest.raises(ValueError, match="cumulative fill regressed"):
        store.record_order_update(dict(filled, filled_qty=1), "paper")
    with pytest.raises(ValueError, match="terminal status"):
        store.record_order_update(dict(filled, status="accepted"), "paper")


def test_reconcile_clears_canceled_stop_and_requires_replacement(tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")
    buy = {
        "order_id": "buy-filled",
        "ticker": "AAPL",
        "side": "buy",
        "type": "limit",
        "status": "filled",
        "qty": 10,
        "filled_qty": 10,
        "filled_avg_price": 100,
    }
    active_stop = {
        "order_id": "stop-active",
        "ticker": "AAPL",
        "side": "sell",
        "type": "stop",
        "status": "new",
        "qty": 10,
        "filled_qty": 0,
        "filled_avg_price": 0,
    }
    broker_position = {"ticker": "AAPL", "qty": 10, "cost_basis": 1000}

    store.reconcile_broker_state([broker_position], [buy, active_stop], "live")
    assert store.positions_needing_stop_orders() == []

    canceled_stop = dict(active_stop, status="canceled")
    store.reconcile_broker_state([broker_position], [buy, canceled_stop], "live")
    assert store.positions_needing_stop_orders()[0]["ticker"] == "AAPL"


def test_persistent_daily_risk_halt_survives_restart(tmp_path):
    path = tmp_path / "state.db"
    start = datetime(2026, 7, 10, 14, tzinfo=timezone.utc)
    store = StrategyStateStore(path)
    assert not store.evaluate_persistent_risk(
        equity=10_000, mode="paper", now=start
    )["halted"]

    halted = store.evaluate_persistent_risk(
        equity=9_940, mode="paper", now=start + timedelta(hours=1)
    )
    restarted = StrategyStateStore(path).evaluate_persistent_risk(
        equity=9_940, mode="paper", now=start + timedelta(hours=2)
    )

    assert halted["halted"] is True
    assert restarted["halted"] is True
    assert "daily drawdown" in restarted["reason"]


def test_manual_halt_works_before_first_broker_cycle(tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")

    store.set_manual_halt("operator test")
    status = store.risk_status()

    assert status["manual_halt"] == 1
    assert status["halt_reason"] == "operator test"

    evaluated = store.evaluate_persistent_risk(equity=10_000, mode="paper")
    assert evaluated["halted"] is True
    assert evaluated["peak_equity"] == 10_000


def test_fill_based_performance_does_not_count_duplicate_order_update(tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")
    buy = {
        "order_id": "buy-performance",
        "ticker": "AAPL",
        "side": "buy",
        "type": "limit",
        "status": "filled",
        "qty": 2,
        "filled_qty": 2,
        "filled_avg_price": 100,
    }
    sell = {
        "order_id": "sell-performance",
        "ticker": "AAPL",
        "side": "sell",
        "type": "limit",
        "status": "filled",
        "qty": 2,
        "filled_qty": 2,
        "filled_avg_price": 110,
    }
    store.record_order_update(buy, "paper")
    store.record_order_update(buy, "paper")
    store.record_order_update(sell, "paper")

    summary = store.performance_summary()

    assert summary["fill_count"] == 2
    assert summary["realized_pnl"] == 20

    assert store.reconcile_account_activities(
        [
            {
                "id": "fee-1",
                "activity_type": "FEE",
                "symbol": "AAPL",
                "net_amount": "-1.25",
            }
        ]
    ) == 1
    assert store.reconcile_account_activities(
        [{"id": "fee-1", "activity_type": "FEE", "net_amount": "-1.25"}]
    ) == 0
    summary = store.performance_summary()
    assert summary["fees"] == 1.25
    assert summary["realized_pnl"] == 18.75


def test_health_snapshot_reports_unprotected_live_position(tmp_path):
    store = StrategyStateStore(tmp_path / "state.db")
    store.record_order_update(
        {
            "order_id": "health-buy",
            "ticker": "AAPL",
            "side": "buy",
            "type": "limit",
            "status": "filled",
            "qty": 1,
            "filled_qty": 1,
            "filled_avg_price": 100,
        },
        "paper",
    )

    health = store.health_snapshot()

    assert health["open_positions"] == 1
    assert health["unprotected_tickers"] == ["AAPL"]


def test_summarize_decisions_counts_actions():
    assert summarize_decisions([]) == "-"
    assert summarize_decisions(
        [
            {"decision": "skip"},
            {"decision": "skip"},
            {"decision": "shadow-buy"},
        ]
    ) == "shadow-buy:1, skip:2"
