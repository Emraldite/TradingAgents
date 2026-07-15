import logging

import requests

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
        "client_order_id": "",
        "ticker": "AAPL",
        "side": "buy",
        "type": "",
        "order_class": "",
        "parent_order_id": None,
        "notional": 5000.0,
        "qty": 0.0,
        "status": "accepted",
        "filled_qty": 0.0,
        "filled_avg_price": 0.0,
        "limit_price": 0.0,
        "stop_price": 0.0,
        "submitted_at": "2026-05-20T15:16:19Z",
        "filled_at": None,
        "legs": [],
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


def test_free_iex_snapshot_returns_bid_ask_and_timestamp(monkeypatch):
    from tradingagents.execution import alpaca_executor

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "latestQuote": {
                    "bp": 99.9,
                    "ap": 100.1,
                    "t": "2026-07-11T15:30:00Z",
                },
                "latestTrade": {"p": 100.0, "t": "2026-07-11T15:29:59Z"},
            }

    executor = alpaca_executor.AlpacaExecutor()
    response = Response()
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        return response

    monkeypatch.setattr("tradingagents.execution.alpaca_executor.requests.get", fake_get)

    snapshot, error = executor.get_stock_snapshot_checked("aapl")

    assert error is None
    assert snapshot["bid"] == 99.9
    assert snapshot["ask"] == 100.1
    assert captured["params"] == {"feed": "iex"}
    assert captured["url"].endswith("/v2/stocks/AAPL/snapshot")


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


def test_bracket_buy_contains_broker_side_stop_and_take_profit(monkeypatch):
    from tradingagents.execution import alpaca_executor

    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "bracket-1",
                "client_order_id": "ta-buy-1",
                "symbol": "AAPL",
                "side": "buy",
                "type": "limit",
                "order_class": "bracket",
                "qty": "5",
                "status": "accepted",
                "filled_qty": "0",
                "filled_avg_price": None,
            }

    executor = alpaca_executor.AlpacaExecutor()
    monkeypatch.setattr(executor, "get_order_by_client_id", lambda value: None)
    monkeypatch.setattr(
        alpaca_executor.requests,
        "post",
        lambda *args, **kwargs: captured.update(kwargs["json"]) or Response(),
    )

    order = executor.execute_buy_bracket("AAPL", 1000, 200, 186, 224, "ta-buy-1")

    assert captured["order_class"] == "bracket"
    assert captured["client_order_id"] == "ta-buy-1"
    assert captured["qty"] == 5
    assert captured["stop_loss"] == {"stop_price": 186}
    assert captured["take_profit"] == {"limit_price": 224}
    assert order["order_class"] == "bracket"


def test_bracket_buy_floors_to_whole_shares(monkeypatch):
    from tradingagents.execution import alpaca_executor

    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "bracket-whole",
                "symbol": "NVDA",
                "side": "buy",
                "type": "limit",
                "order_class": "bracket",
                "qty": "4",
                "status": "accepted",
                "filled_qty": "0",
                "filled_avg_price": None,
            }

    executor = alpaca_executor.AlpacaExecutor()
    monkeypatch.setattr(executor, "get_order_by_client_id", lambda value: None)
    monkeypatch.setattr(
        alpaca_executor.requests,
        "post",
        lambda *args, **kwargs: captured.update(kwargs["json"]) or Response(),
    )

    executor.execute_buy_bracket("NVDA", 999, 200, 186, 224, "ta-buy-whole")

    assert captured["qty"] == 4
    assert isinstance(captured["qty"], int)
    assert captured["qty"] * captured["limit_price"] <= 999


def test_bracket_buy_rejects_invalid_prices_before_submission(monkeypatch):
    from tradingagents.execution import alpaca_executor

    called = False

    def fake_post(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(alpaca_executor.requests, "post", fake_post)

    order = alpaca_executor.AlpacaExecutor().execute_buy_bracket(
        "NVDA", 1000, 200, 201, 224, "ta-buy-invalid"
    )

    assert order is None
    assert called is False


def test_bracket_buy_skips_when_budget_is_less_than_one_share(monkeypatch):
    from tradingagents.execution import alpaca_executor

    called = False

    def fake_post(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(alpaca_executor.requests, "post", fake_post)

    order = alpaca_executor.AlpacaExecutor().execute_buy_bracket(
        "NVDA", 199, 200, 186, 224, "ta-buy-too-small"
    )

    assert order is None
    assert called is False


def test_rejected_order_logs_safe_alpaca_message(monkeypatch, caplog):
    from tradingagents.execution import alpaca_executor

    response = requests.Response()
    response.status_code = 422
    response._content = b'{"code":42210000,"message":"qty must be integer"}'
    response.url = "https://paper-api.alpaca.markets/v2/orders"

    executor = alpaca_executor.AlpacaExecutor()
    monkeypatch.setattr(executor, "get_order_by_client_id", lambda value: None)
    monkeypatch.setattr(alpaca_executor.requests, "post", lambda *args, **kwargs: response)

    with caplog.at_level(logging.ERROR):
        order = executor.execute_buy_bracket(
            "NVDA", 1000, 200, 186, 224, "ta-buy-rejected"
        )

    assert order is None
    assert "status=422" in caplog.text
    assert "code=42210000 message=qty must be integer" in caplog.text
    assert "APCA-API-KEY-ID" not in caplog.text
    assert "APCA-API-SECRET-KEY" not in caplog.text


def test_submit_recovers_existing_order_after_timeout(monkeypatch):
    from tradingagents.execution import alpaca_executor

    executor = alpaca_executor.AlpacaExecutor()
    recovered = {
        "order_id": "accepted-1",
        "client_order_id": "ta-buy-recover",
        "ticker": "AAPL",
        "side": "buy",
        "type": "limit",
        "order_class": "",
        "parent_order_id": None,
        "notional": 0,
        "qty": 5,
        "status": "accepted",
        "filled_qty": 0,
        "filled_avg_price": 0,
        "limit_price": 200,
        "stop_price": 0,
        "submitted_at": None,
        "filled_at": None,
        "legs": [],
    }
    lookups = iter([None, recovered])
    monkeypatch.setattr(executor, "get_order_by_client_id", lambda value: next(lookups))
    monkeypatch.setattr(
        alpaca_executor.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("response lost")),
    )

    order = executor.execute_buy_limit(
        "AAPL", 1000, 200, client_order_id="ta-buy-recover"
    )

    assert order["order_id"] == "accepted-1"
