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


def test_summarize_decisions_counts_actions():
    assert summarize_decisions([]) == "-"
    assert summarize_decisions(
        [
            {"decision": "skip"},
            {"decision": "skip"},
            {"decision": "shadow-buy"},
        ]
    ) == "shadow-buy:1, skip:2"
