from tradingagents.execution.alpaca_executor import _order_summary, _to_float


def test_to_float_handles_pending_alpaca_null_fields():
    assert _to_float(None) == 0.0
    assert _to_float("") == 0.0
    assert _to_float("16.5") == 16.5


def test_order_summary_handles_unfilled_market_order():
    order = {
        "id": "order-123",
        "symbol": "AAPL",
        "side": "buy",
        "notional": "5000",
        "qty": None,
        "status": "accepted",
        "filled_qty": None,
        "filled_avg_price": None,
        "submitted_at": "2026-05-20T15:16:19Z",
        "filled_at": None,
    }

    assert _order_summary(order) == {
        "order_id": "order-123",
        "ticker": "AAPL",
        "side": "buy",
        "type": "",
        "notional": 5000.0,
        "qty": 0.0,
        "status": "accepted",
        "filled_qty": 0.0,
        "filled_avg_price": 0.0,
        "limit_price": 0.0,
        "stop_price": 0.0,
        "submitted_at": "2026-05-20T15:16:19Z",
        "filled_at": None,
    }


def test_market_open_fails_closed_when_clock_request_fails(monkeypatch):
    from tradingagents.execution import alpaca_executor

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(alpaca_executor.requests, "get", boom)

    executor = alpaca_executor.AlpacaExecutor()

    assert executor.get_market_clock() is None
    assert executor.is_market_open() is False
    assert executor.require_market_open()[0] is False


def test_limit_buy_uses_quantity_and_limit_price(monkeypatch):
    from tradingagents.execution import alpaca_executor

    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "order-1",
                "symbol": "AAPL",
                "side": "buy",
                "type": "limit",
                "qty": "5",
                "notional": None,
                "status": "accepted",
                "filled_qty": None,
                "filled_avg_price": None,
                "submitted_at": None,
                "filled_at": None,
            }

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setattr(alpaca_executor.requests, "post", fake_post)

    order = alpaca_executor.AlpacaExecutor().execute_buy_limit("AAPL", 1000, 200)

    assert captured["type"] == "limit"
    assert captured["qty"] == 5
    assert captured["limit_price"] == 200
    assert order["type"] == "limit"


def test_stop_sell_uses_gtc_stop_order(monkeypatch):
    from tradingagents.execution import alpaca_executor

    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "stop-1",
                "symbol": "AAPL",
                "side": "sell",
                "type": "stop",
                "qty": "5",
                "notional": None,
                "status": "accepted",
                "filled_qty": None,
                "filled_avg_price": None,
                "limit_price": None,
                "stop_price": "186",
                "submitted_at": None,
                "filled_at": None,
            }

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setattr(alpaca_executor.requests, "post", fake_post)

    order = alpaca_executor.AlpacaExecutor().execute_stop_sell("AAPL", 5, 186)

    assert captured["type"] == "stop"
    assert captured["time_in_force"] == "gtc"
    assert captured["stop_price"] == 186
    assert order["type"] == "stop"
