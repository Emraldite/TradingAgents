from __future__ import annotations

import logging
import math
import os
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ALPACA_BASE = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DATA_BASE = "https://data.alpaca.markets"


def _headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": os.environ.get("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", ""),
    }


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _order_summary(order: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "order_id": order["id"],
        "client_order_id": order.get("client_order_id", ""),
        "ticker": order.get("symbol", ""),
        "side": order.get("side", ""),
        "type": order.get("type", ""),
        "order_class": order.get("order_class", "") or "",
        "parent_order_id": order.get("parent_order_id"),
        "notional": _to_float(order.get("notional")),
        "qty": _to_float(order.get("qty")),
        "status": order.get("status", ""),
        "filled_qty": _to_float(order.get("filled_qty")),
        "filled_avg_price": _to_float(order.get("filled_avg_price")),
        "limit_price": _to_float(order.get("limit_price")),
        "stop_price": _to_float(order.get("stop_price")),
        "submitted_at": order.get("submitted_at"),
        "filled_at": order.get("filled_at"),
    }
    summary["legs"] = []
    for leg in order.get("legs") or []:
        leg_data = dict(leg)
        leg_data.setdefault("parent_order_id", order["id"])
        summary["legs"].append(_order_summary(leg_data))
    return summary


def _marketable_limit_price(side: str, reference_price: float, slippage_bps: int = 20) -> float:
    multiplier = 1 + slippage_bps / 10_000 if side == "buy" else 1 - slippage_bps / 10_000
    return round(reference_price * multiplier, 2)


def _broker_error_detail(response: Any) -> str:
    """Return only Alpaca's documented error code/message, never headers or keys."""
    if response is None:
        return ""
    try:
        payload = response.json()
    except Exception:
        return "non-JSON broker response"
    if not isinstance(payload, dict):
        return "unrecognized broker error response"
    code = str(payload.get("code", "unknown"))[:80]
    message = str(payload.get("message", "unknown")).replace("\n", " ")[:500]
    return f"code={code} message={message}"


def _safe_order_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Whitelist non-secret fields that are useful when Alpaca rejects an order."""
    allowed = (
        "symbol",
        "side",
        "type",
        "time_in_force",
        "order_class",
        "qty",
        "notional",
        "limit_price",
        "take_profit",
        "stop_loss",
        "client_order_id",
    )
    return {key: payload[key] for key in allowed if key in payload}


class AlpacaExecutor:
    def endpoint_host(self) -> str:
        return str(urlparse(ALPACA_BASE).hostname or "")

    def is_official_endpoint(self) -> bool:
        parsed = urlparse(ALPACA_BASE)
        return parsed.scheme == "https" and self.endpoint_host() in {
            "paper-api.alpaca.markets",
            "api.alpaca.markets",
        }

    def is_paper_endpoint(self) -> bool:
        return self.endpoint_host() == "paper-api.alpaca.markets"

    def get_market_clock(self) -> dict[str, Any] | None:
        try:
            resp = requests.get(
                f"{ALPACA_BASE}/v2/clock", headers=_headers(), timeout=10
            )
            resp.raise_for_status()
            clock = resp.json()
            return {
                "is_open": bool(clock.get("is_open", False)),
                "timestamp": clock.get("timestamp"),
                "next_open": clock.get("next_open"),
                "next_close": clock.get("next_close"),
            }
        except Exception as exc:
            logger.error("Could not check Alpaca market clock: %s", exc)
            return None

    def is_market_open(self) -> bool:
        clock = self.get_market_clock()
        if clock is None:
            return False
        return bool(clock["is_open"])

    def require_market_open(self) -> tuple[bool, dict[str, Any] | None, str | None]:
        clock = self.get_market_clock()
        if clock is None:
            return False, None, "Could not verify Alpaca market clock"
        if not clock["is_open"]:
            return False, clock, f"Market closed until {clock.get('next_open')}"
        return True, clock, None

    def get_stock_snapshot_checked(
        self,
        ticker: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Fetch the free IEX quote/trade snapshot used for broker order pricing."""
        symbol = ticker.strip().upper()
        if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
            return None, "Invalid ticker for Alpaca snapshot"
        try:
            resp = requests.get(
                f"{DATA_BASE}/v2/stocks/{symbol}/snapshot",
                headers=_headers(),
                params={"feed": "iex"},
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
            quote = payload.get("latestQuote") or {}
            trade = payload.get("latestTrade") or {}
            ask = _to_float(quote.get("ap"))
            bid = _to_float(quote.get("bp"))
            last = _to_float(trade.get("p"))
            if max(ask, bid, last) <= 0:
                return None, "Alpaca IEX snapshot had no positive quote or trade price"
            return {
                "ticker": symbol,
                "ask": ask,
                "bid": bid,
                "last": last,
                "timestamp": quote.get("t") or trade.get("t"),
                "feed": "iex",
            }, None
        except Exception as exc:
            logger.error("Failed to fetch Alpaca IEX snapshot for %s: %s", symbol, exc)
            return None, str(exc)

    def get_open_orders(self) -> list[dict[str, Any]]:
        return self.get_recent_orders(limit=100, status="open")

    def get_portfolio_checked(self) -> tuple[list[dict[str, Any]], str | None]:
        try:
            resp = requests.get(
                f"{ALPACA_BASE}/v2/positions", headers=_headers(), timeout=10
            )
            resp.raise_for_status()
            positions = resp.json()
            return [
                {
                    "ticker": p["symbol"],
                    "qty": float(p["qty"]),
                    "market_value": float(p["market_value"]),
                    "cost_basis": float(p["cost_basis"]),
                    "unrealized_pl": float(p["unrealized_pl"]),
                    "unrealized_plpc": float(p["unrealized_plpc"]),
                }
                for p in positions
            ], None
        except Exception as exc:
            logger.error("Failed to fetch portfolio: %s", exc)
            return [], str(exc)

    def get_portfolio(self) -> list[dict[str, Any]]:
        positions, _ = self.get_portfolio_checked()
        return positions

    def get_account_info(self) -> dict[str, Any] | None:
        try:
            resp = requests.get(
                f"{ALPACA_BASE}/v2/account", headers=_headers(), timeout=10
            )
            resp.raise_for_status()
            acct = resp.json()
            return {
                "account_id": str(acct.get("id", "")),
                "status": str(acct.get("status", "")),
                "cash": float(acct["cash"]),
                "portfolio_value": float(acct["portfolio_value"]),
                "buying_power": float(acct["buying_power"]),
                "day_trade_count": int(acct.get("day_trade_count", 0)),
                "account_blocked": bool(acct.get("account_blocked", False)),
                "trading_blocked": bool(acct.get("trading_blocked", False)),
                "trade_suspended_by_user": bool(acct.get("trade_suspended_by_user", False)),
                "pattern_day_trader": bool(acct.get("pattern_day_trader", False)),
            }
        except Exception as exc:
            logger.error("Failed to fetch account info: %s", exc)
            return None

    def get_account_activities_checked(
        self,
        activity_types: str = "FILL,FEE",
        page_size: int = 100,
    ) -> tuple[list[dict[str, Any]], str | None]:
        try:
            resp = requests.get(
                f"{ALPACA_BASE}/v2/account/activities",
                headers=_headers(),
                params={
                    "activity_types": activity_types,
                    "page_size": max(1, min(page_size, 100)),
                    "direction": "desc",
                },
                timeout=10,
            )
            resp.raise_for_status()
            activities = resp.json()
            return [dict(activity) for activity in activities], None
        except Exception as exc:
            logger.error("Failed to fetch account activities: %s", exc)
            return [], str(exc)

    def get_recent_orders(
        self,
        limit: int = 10,
        status: str = "all",
    ) -> list[dict[str, Any]]:
        try:
            resp = requests.get(
                f"{ALPACA_BASE}/v2/orders",
                headers=_headers(),
                params={
                    "status": status,
                    "limit": max(1, min(limit, 100)),
                    "direction": "desc",
                    "nested": "true",
                },
                timeout=10,
            )
            resp.raise_for_status()
            orders = resp.json()
            return [_order_summary(order) for order in orders]
        except Exception as exc:
            logger.error("Failed to fetch recent orders: %s", exc)
            return []

    def get_order_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        if not client_order_id:
            return None
        try:
            resp = requests.get(
                f"{ALPACA_BASE}/v2/orders:by_client_order_id",
                headers=_headers(),
                params={"client_order_id": client_order_id},
                timeout=10,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return _order_summary(resp.json())
        except Exception as exc:
            logger.error("Could not look up Alpaca order %s: %s", client_order_id, exc)
            return None

    def cancel_order(self, order_id: str) -> bool:
        try:
            resp = requests.delete(
                f"{ALPACA_BASE}/v2/orders/{order_id}", headers=_headers(), timeout=10
            )
            if resp.status_code in {204, 404}:
                return True
            resp.raise_for_status()
            return True
        except Exception as exc:
            logger.error("Could not cancel Alpaca order %s: %s", order_id, exc)
            return False

    def cancel_open_sell_orders(self, ticker: str) -> bool:
        matching = [
            order
            for order in self.get_open_orders()
            if order.get("ticker") == ticker and order.get("side") == "sell"
        ]
        return all(self.cancel_order(str(order["order_id"])) for order in matching)

    def _submit_order(
        self,
        payload: dict[str, Any],
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any] | None:
        if client_order_id:
            existing = self.get_order_by_client_id(client_order_id)
            if existing is not None:
                logger.info("Reusing Alpaca order %s for %s", existing["order_id"], client_order_id)
                return existing
            payload["client_order_id"] = client_order_id
        try:
            resp = requests.post(
                f"{ALPACA_BASE}/v2/orders",
                headers=_headers(),
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            return _order_summary(resp.json())
        except Exception as exc:
            # The broker may have accepted an order before the response timed out.
            recovered = self.get_order_by_client_id(client_order_id or "")
            if recovered is not None:
                logger.warning("Recovered accepted Alpaca order after submit error: %s", exc)
                return recovered
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", "unknown")
            detail = _broker_error_detail(response) or str(exc)
            logger.error(
                "Order submission failed for %s: status=%s detail=%s order=%s",
                payload.get("symbol"),
                status,
                detail,
                _safe_order_fields(payload),
            )
            return None

    def execute_buy(
        self,
        ticker: str,
        dollar_amount: float,
        client_order_id: str | None = None,
    ) -> dict[str, Any] | None:
        return self.execute_buy_limit(
            ticker, dollar_amount, limit_price=None, client_order_id=client_order_id
        )

    def execute_buy_limit(
        self,
        ticker: str,
        dollar_amount: float,
        limit_price: float | None,
        client_order_id: str | None = None,
    ) -> dict[str, Any] | None:
        order_type = "limit" if limit_price and limit_price > 0 else "market"
        payload: dict[str, Any] = {
            "symbol": ticker,
            "side": "buy",
            "type": order_type,
            "time_in_force": "day",
        }
        if order_type == "limit":
            qty = round(dollar_amount / float(limit_price), 6)
            payload.update({"qty": qty, "limit_price": round(float(limit_price), 2)})
        else:
            payload["notional"] = round(dollar_amount, 2)
        order = self._submit_order(payload, client_order_id=client_order_id)
        if order:
            logger.info("BUY %s: $%.2f via %s order %s", ticker, dollar_amount, order_type, order["order_id"])
        return order

    def execute_buy_bracket(
        self,
        ticker: str,
        dollar_amount: float,
        limit_price: float,
        stop_price: float,
        take_profit_price: float,
        client_order_id: str,
    ) -> dict[str, Any] | None:
        if dollar_amount <= 0 or min(limit_price, stop_price, take_profit_price) <= 0:
            logger.error("Bracket buy failed for %s: invalid price", ticker)
            return None
        entry = round(limit_price, 2)
        stop = round(stop_price, 2)
        target = round(take_profit_price, 2)
        if stop >= entry or target <= entry:
            logger.error(
                "Bracket buy failed for %s: require stop < entry < target "
                "(stop=%.2f entry=%.2f target=%.2f)",
                ticker,
                stop,
                entry,
                target,
            )
            return None

        # Whole shares are the most portable choice for Alpaca advanced orders.
        # Flooring also guarantees the order cannot exceed its dollar risk budget.
        qty = math.floor(dollar_amount / entry)
        if qty < 1:
            logger.info(
                "Bracket buy skipped for %s: $%.2f is below one share at $%.2f",
                ticker,
                dollar_amount,
                entry,
            )
            return None
        payload = {
            "symbol": ticker,
            "qty": qty,
            "side": "buy",
            "type": "limit",
            "limit_price": entry,
            "time_in_force": "day",
            "order_class": "bracket",
            "take_profit": {"limit_price": target},
            "stop_loss": {"stop_price": stop},
        }
        return self._submit_order(payload, client_order_id=client_order_id)

    def execute_sell(self, ticker: str, qty: float) -> dict[str, Any] | None:
        return self.execute_sell_market(ticker, qty)

    def execute_sell_market(
        self, ticker: str, qty: float, client_order_id: str | None = None
    ) -> dict[str, Any] | None:
        return self._submit_sell(
            ticker,
            qty,
            order_type="market",
            limit_price=None,
            client_order_id=client_order_id,
        )

    def execute_sell_limit(
        self,
        ticker: str,
        qty: float,
        limit_price: float | None,
        client_order_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not limit_price or limit_price <= 0:
            return self.execute_sell_market(ticker, qty, client_order_id=client_order_id)
        return self._submit_sell(
            ticker,
            qty,
            order_type="limit",
            limit_price=limit_price,
            client_order_id=client_order_id,
        )

    def execute_stop_sell(
        self,
        ticker: str,
        qty: float,
        stop_price: float,
        client_order_id: str | None = None,
    ) -> dict[str, Any] | None:
        if stop_price <= 0:
            logger.error("Stop sell failed for %s: invalid stop price %.2f", ticker, stop_price)
            return None
        return self._submit_sell(
            ticker,
            qty,
            order_type="stop",
            limit_price=None,
            stop_price=stop_price,
            time_in_force="gtc",
            client_order_id=client_order_id,
        )

    def execute_oco_sell(
        self,
        ticker: str,
        qty: float,
        take_profit_price: float,
        stop_price: float,
        client_order_id: str,
    ) -> dict[str, Any] | None:
        if qty <= 0 or min(take_profit_price, stop_price) <= 0:
            logger.error("OCO sell failed for %s: invalid quantity or price", ticker)
            return None
        payload = {
            "symbol": ticker,
            "qty": round(qty, 6),
            "side": "sell",
            "type": "limit",
            "limit_price": round(take_profit_price, 2),
            "time_in_force": "gtc",
            "order_class": "oco",
            "take_profit": {"limit_price": round(take_profit_price, 2)},
            "stop_loss": {"stop_price": round(stop_price, 2)},
        }
        return self._submit_order(payload, client_order_id=client_order_id)

    def _submit_sell(
        self,
        ticker: str,
        qty: float,
        order_type: str,
        limit_price: float | None,
        stop_price: float | None = None,
        time_in_force: str = "day",
        client_order_id: str | None = None,
    ) -> dict[str, Any] | None:
        payload: dict[str, Any] = {
            "symbol": ticker,
            "qty": round(qty, 6),
            "side": "sell",
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if order_type == "limit" and limit_price:
            payload["limit_price"] = round(float(limit_price), 2)
        if order_type == "stop" and stop_price:
            payload["stop_price"] = round(float(stop_price), 2)
        order = self._submit_order(payload, client_order_id=client_order_id)
        if order:
            logger.info(
                "SELL %s: %.6f shares via %s order %s",
                ticker,
                qty,
                order_type,
                order["order_id"],
            )
        return order

    @staticmethod
    def pre_execution_checklist(
        ticker: str,
        dollar_amount: float,
        portfolio_value: float,
        open_positions: int,
        daily_volume: float | None,
        kill_switch_msg: str | None,
        max_position_pct: float = 0.08,
        max_open_positions: int = 5,
        min_daily_volume: int = 500_000,
    ) -> tuple[bool, str]:
        if kill_switch_msg:
            return False, f"Kill switch active: {kill_switch_msg}"
        if dollar_amount > portfolio_value * max_position_pct:
            return (
                False,
                f"Position ${dollar_amount:.0f} exceeds {max_position_pct:.0%} of portfolio",
            )
        if open_positions >= max_open_positions:
            return False, f"At max positions ({max_open_positions})"
        if daily_volume is not None and daily_volume < min_daily_volume:
            return False, f"Volume {daily_volume:.0f} below {min_daily_volume:,}"
        return True, "Checklist passed"
