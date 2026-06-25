from __future__ import annotations

import logging
import os
from typing import Any

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
    return {
        "order_id": order["id"],
        "ticker": order.get("symbol", ""),
        "side": order.get("side", ""),
        "type": order.get("type", ""),
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


def _marketable_limit_price(side: str, reference_price: float, slippage_bps: int = 20) -> float:
    multiplier = 1 + slippage_bps / 10_000 if side == "buy" else 1 - slippage_bps / 10_000
    return round(reference_price * multiplier, 2)


class AlpacaExecutor:
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
                "cash": float(acct["cash"]),
                "portfolio_value": float(acct["portfolio_value"]),
                "buying_power": float(acct["buying_power"]),
                "day_trade_count": int(acct.get("day_trade_count", 0)),
            }
        except Exception as exc:
            logger.error("Failed to fetch account info: %s", exc)
            return None

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
                },
                timeout=10,
            )
            resp.raise_for_status()
            orders = resp.json()
            return [_order_summary(order) for order in orders]
        except Exception as exc:
            logger.error("Failed to fetch recent orders: %s", exc)
            return []

    def execute_buy(self, ticker: str, dollar_amount: float) -> dict[str, Any] | None:
        return self.execute_buy_limit(ticker, dollar_amount, limit_price=None)

    def execute_buy_limit(
        self,
        ticker: str,
        dollar_amount: float,
        limit_price: float | None,
    ) -> dict[str, Any] | None:
        try:
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
            resp = requests.post(
                f"{ALPACA_BASE}/v2/orders",
                headers=_headers(),
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            order = resp.json()
            logger.info(
                "BUY %s: $%.2f via %s order %s",
                ticker,
                dollar_amount,
                order_type,
                order["id"],
            )
            return _order_summary(order)
        except Exception as exc:
            logger.error("Buy order failed for %s: %s", ticker, exc)
            return None

    def execute_sell(self, ticker: str, qty: float) -> dict[str, Any] | None:
        return self.execute_sell_market(ticker, qty)

    def execute_sell_market(self, ticker: str, qty: float) -> dict[str, Any] | None:
        return self._submit_sell(ticker, qty, order_type="market", limit_price=None)

    def execute_sell_limit(
        self,
        ticker: str,
        qty: float,
        limit_price: float | None,
    ) -> dict[str, Any] | None:
        if not limit_price or limit_price <= 0:
            return self.execute_sell_market(ticker, qty)
        return self._submit_sell(ticker, qty, order_type="limit", limit_price=limit_price)

    def execute_stop_sell(
        self,
        ticker: str,
        qty: float,
        stop_price: float,
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
        )

    def _submit_sell(
        self,
        ticker: str,
        qty: float,
        order_type: str,
        limit_price: float | None,
        stop_price: float | None = None,
        time_in_force: str = "day",
    ) -> dict[str, Any] | None:
        try:
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
            resp = requests.post(
                f"{ALPACA_BASE}/v2/orders",
                headers=_headers(),
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            order = resp.json()
            logger.info(
                "SELL %s: %.6f shares via %s order %s",
                ticker,
                qty,
                order_type,
                order["id"],
            )
            return _order_summary(order)
        except Exception as exc:
            logger.error("Sell order failed for %s: %s", ticker, exc)
            return None

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
