from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Literal

import yfinance as yf

from tradingagents.alerts import AlertManager
from tradingagents.dataflows.congressional_data import get_conviction_watchlist
from tradingagents.dataflows.manipulation_detector import detect_manipulation
from tradingagents.execution.alpaca_executor import AlpacaExecutor
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.risk.survival_rules import SurvivalRules, validate_trade
from tradingagents.risk.performance_tracker import PerformanceTracker
from tradingagents.state_store import StrategyStateStore
from tradingagents.default_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

executor = AlpacaExecutor()
kill_switch = SurvivalRules()
tracker = PerformanceTracker(
    DEFAULT_CONFIG.get("data_cache_dir", "data") + "/performance.db"
)
state_store = StrategyStateStore(
    DEFAULT_CONFIG.get("data_cache_dir", "data") + "/strategy_state.db"
)
alerts = AlertManager()

ExecutionMode = Literal["dry-run", "shadow", "live"]

STOP_LOSS_PCT = -0.07
TAKE_PROFIT_PCT = 0.12
MAX_HOLD_TRADING_DAYS = 10
MANIPULATION_SELL_THRESHOLD = 0.85
LIMIT_SLIPPAGE_BPS = 20
SHADOW_SLIPPAGE_BPS = 10
MAX_UNIVERSE_SIZE = 20
GRAPH_ANALYSTS = ["congressional", "market", "social", "news", "fundamentals"]
GRAPH_BUY_RATINGS = {"Buy", "Overweight"}

TECHNICAL_SCREEN_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "AVGO",
    "AMD",
    "TSLA",
    "JPM",
    "LLY",
    "UNH",
    "XOM",
    "COST",
    "NFLX",
    "CRM",
    "ORCL",
    "ADBE",
    "QCOM",
    "INTC",
    "RTX",
    "LMT",
    "BA",
    "CVX",
]

SECTOR_EXPANSION = {
    "semiconductors": ["NVDA", "AMD", "AVGO", "QCOM", "INTC"],
    "defense": ["LMT", "RTX", "NOC", "GD", "BA"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "megacap_tech": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
}

LEVERAGED_ETFS = {
    "TQQQ",
    "SQQQ",
    "SOXL",
    "SOXS",
    "SPXL",
    "SPXS",
    "UPRO",
    "UVXY",
    "LABU",
    "LABD",
}


def _latest_scalar(data, column: str) -> float | None:
    if column not in data:
        return None

    values = data[column]
    if hasattr(values, "iloc"):
        latest = values.iloc[-1]
        if hasattr(latest, "iloc"):
            latest = latest.iloc[-1]
        return float(latest)
    return None


def _mean_scalar(values) -> float:
    mean_value = values.mean()
    if hasattr(mean_value, "iloc"):
        mean_value = mean_value.iloc[-1]
    return float(mean_value)


def _marketable_limit_price(side: str, price: float, slippage_bps: int = LIMIT_SLIPPAGE_BPS) -> float:
    if price <= 0:
        return 0.0
    multiplier = 1 + slippage_bps / 10_000 if side == "buy" else 1 - slippage_bps / 10_000
    return round(price * multiplier, 2)


def _shadow_fill_price(side: str, price: float) -> float:
    if price <= 0:
        return 0.0
    multiplier = 1 + SHADOW_SLIPPAGE_BPS / 10_000 if side == "buy" else 1 - SHADOW_SLIPPAGE_BPS / 10_000
    return round(price * multiplier, 4)


def _trading_days_between(entry_iso: str | None, now: datetime | None = None) -> int:
    if not entry_iso:
        return 0
    try:
        entry = datetime.fromisoformat(entry_iso.replace("Z", "+00:00")).date()
    except ValueError:
        return 0
    today = (now or datetime.now(timezone.utc)).date()
    if today <= entry:
        return 0
    days = 0
    cursor = entry
    while cursor < today:
        cursor = cursor.fromordinal(cursor.toordinal() + 1)
        if cursor.weekday() < 5:
            days += 1
    return days


def _is_within_next_trading_days(target: datetime, trading_days: int = 5) -> bool:
    now = datetime.now(target.tzinfo or timezone.utc)
    if target < now:
        return False
    days = 0
    cursor = now.date()
    while days < trading_days:
        cursor = cursor.fromordinal(cursor.toordinal() + 1)
        if cursor.weekday() < 5:
            days += 1
    return target.date() <= cursor


def _next_earnings_date(ticker: str) -> datetime | None:
    try:
        calendar = yf.Ticker(ticker).calendar
    except Exception as exc:
        logger.debug("Earnings calendar check failed for %s: %s", ticker, exc)
        return None
    if calendar is None:
        return None
    raw = None
    if isinstance(calendar, dict):
        raw = calendar.get("Earnings Date") or calendar.get("EarningsDate")
    elif hasattr(calendar, "loc") and "Earnings Date" in calendar.index:
        raw = calendar.loc["Earnings Date"]
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    if hasattr(raw, "iloc"):
        raw = raw.iloc[0]
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _hard_exclusion_reason(ticker: str) -> str | None:
    if ticker in LEVERAGED_ETFS:
        return "leveraged ETF excluded"
    earnings = _next_earnings_date(ticker)
    if earnings and _is_within_next_trading_days(earnings, trading_days=5):
        return f"earnings within 5 trading days: {earnings.date().isoformat()}"
    return None


def _exit_reason(
    entry_price: float,
    current_price: float,
    entry_date: str | None,
    manipulation_score: float,
) -> str | None:
    if entry_price <= 0 or current_price <= 0:
        return None
    return_pct = (current_price - entry_price) / entry_price
    if return_pct <= STOP_LOSS_PCT:
        return "stop_loss"
    if return_pct >= TAKE_PROFIT_PCT:
        return "take_profit"
    if _trading_days_between(entry_date) >= MAX_HOLD_TRADING_DAYS:
        return "time_exit"
    if manipulation_score >= MANIPULATION_SELL_THRESHOLD:
        return "manipulation_spike"
    return None


def _stop_price(entry_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    return round(entry_price * (1 + STOP_LOSS_PCT), 2)


def _latest_close_and_volume(ticker: str, period: str = "5d") -> tuple[float, float | None]:
    data = yf.download(
        ticker,
        period=period,
        progress=False,
        auto_adjust=False,
    )
    if data.empty:
        return 0.0, None
    return _latest_scalar(data, "Close") or 0.0, _latest_scalar(data, "Volume")


def _technical_candidates(max_candidates: int = 8) -> list[str]:
    candidates: list[str] = []
    for ticker in TECHNICAL_SCREEN_TICKERS:
        try:
            data = yf.download(
                ticker,
                period="3mo",
                progress=False,
                auto_adjust=False,
            )
            if data.empty or len(data) < 30:
                continue
            close = _latest_scalar(data, "Close") or 0.0
            volume = _latest_scalar(data, "Volume") or 0.0
            avg_volume = _mean_scalar(data["Volume"].tail(30))
            ma20 = _mean_scalar(data["Close"].tail(20))
            if close > ma20 and avg_volume > 0 and volume / avg_volume >= 2:
                candidates.append(ticker)
        except Exception as exc:
            logger.debug("Technical screen failed for %s: %s", ticker, exc)
        if len(candidates) >= max_candidates:
            break
    return candidates


def _sector_expansion_candidates(watchlist_tickers: list[str]) -> list[str]:
    expanded: list[str] = []
    watch = set(watchlist_tickers)
    for tickers in SECTOR_EXPANSION.values():
        if len(watch.intersection(tickers)) >= 2:
            for ticker in tickers:
                if ticker not in watch and ticker not in expanded:
                    expanded.append(ticker)
    return expanded


def _build_dynamic_universe(watchlist_tickers: list[str], max_size: int = MAX_UNIVERSE_SIZE) -> list[str]:
    universe: list[str] = []
    for ticker in watchlist_tickers + _technical_candidates() + _sector_expansion_candidates(watchlist_tickers):
        if _hard_exclusion_reason(ticker):
            continue
        if ticker not in universe:
            universe.append(ticker)
        if len(universe) >= max_size:
            break
    return universe


def _reconcile_splits() -> list[dict]:
    adjustments = []
    for position in state_store.open_positions():
        ticker = position["ticker"]
        entry_date = position.get("entry_date")
        if not entry_date:
            continue
        try:
            entry_dt = datetime.fromisoformat(entry_date.replace("Z", "+00:00")).date()
            splits = yf.Ticker(ticker).splits
        except Exception as exc:
            logger.warning("Corporate action check failed for %s: %s", ticker, exc)
            continue
        if splits is None or splits.empty:
            continue
        for ex_date, factor in splits.items():
            ex_date_str = ex_date.date().isoformat() if hasattr(ex_date, "date") else str(ex_date)
            try:
                ex_dt = datetime.fromisoformat(ex_date_str).date()
            except ValueError:
                continue
            if ex_dt <= entry_dt or float(factor) <= 0:
                continue
            if state_store.apply_split(position["id"], ticker, ex_date_str, float(factor)):
                adjustment = {
                    "ticker": ticker,
                    "action": "split",
                    "ex_date": ex_date_str,
                    "factor": float(factor),
                }
                adjustments.append(adjustment)
                logger.info("Applied split adjustment: %s", adjustment)
    return adjustments


def _evaluate_and_execute_sells(
    mode: ExecutionMode,
    decisions: list[dict],
) -> tuple[int, int]:
    executed = 0
    simulated = 0
    for position in state_store.open_positions():
        ticker = position["ticker"]
        qty = float(position.get("broker_quantity") or position.get("quantity") or 0)
        if qty <= 0:
            continue
        close, _ = _latest_close_and_volume(ticker)
        if close <= 0:
            decisions.append({"ticker": ticker, "decision": "sell-skip", "reason": "no price"})
            continue
        manipulation = detect_manipulation(ticker)
        manipulation_score = float(manipulation.get("score", 0) or manipulation.get("manipulation_score", 0) or 0)
        reason = _exit_reason(
            entry_price=float(position.get("entry_price") or 0),
            current_price=close,
            entry_date=position.get("entry_date"),
            manipulation_score=manipulation_score,
        )
        if not reason:
            continue

        if reason == "stop_loss":
            alerts.critical(
                "Stop-loss triggered",
                f"{ticker} triggered a stop-loss exit.",
                {"price": close, "quantity": qty, "mode": mode},
            )

        if mode == "shadow":
            fill_price = _shadow_fill_price("sell", close)
            state_store.record_sell(
                position_id=int(position["id"]),
                ticker=ticker,
                quantity=qty,
                fill_price=fill_price,
                reason=reason,
                mode=mode,
            )
            simulated += 1
            decisions.append(
                {"ticker": ticker, "decision": "shadow-sell", "reason": reason, "price": fill_price}
            )
            continue

        order = (
            executor.execute_sell_market(ticker, qty)
            if reason == "stop_loss"
            else executor.execute_sell_limit(ticker, qty, _marketable_limit_price("sell", close))
        )
        if order:
            state_store.upsert_order(order, mode)
            filled_qty = float(order.get("filled_qty") or 0)
            if filled_qty > 0:
                fill_price = float(order.get("filled_avg_price") or close)
                state_store.record_sell(
                    position_id=int(position["id"]),
                    ticker=ticker,
                    quantity=filled_qty,
                    fill_price=fill_price,
                    reason=reason,
                    mode=mode,
                    order=order,
                )
            else:
                state_store.mark_position_closing(int(position["id"]))
            executed += 1
            decisions.append(
                {
                    "ticker": ticker,
                    "decision": "live-sell-submitted",
                    "reason": reason,
                    "order_id": order.get("order_id"),
                    "status": order.get("status"),
                }
            )
        else:
            alerts.critical(
                "Sell order failed",
                f"{ticker} exit trigger could not submit a sell order.",
                {"reason": reason, "mode": mode},
            )
            decisions.append(
                {"ticker": ticker, "decision": "sell-order-failed", "reason": reason}
            )
    return executed, simulated


def _ensure_native_stop_orders(decisions: list[dict]) -> int:
    created = 0
    for position in state_store.positions_needing_stop_orders():
        ticker = position["ticker"]
        qty = float(position.get("broker_quantity") or 0)
        stop_price = _stop_price(float(position.get("entry_price") or 0))
        if qty <= 0 or stop_price <= 0:
            continue
        order = executor.execute_stop_sell(ticker, qty, stop_price)
        if order:
            state_store.upsert_order(order, "live")
            state_store.set_stop_order(int(position["id"]), order["order_id"])
            created += 1
            decisions.append(
                {
                    "ticker": ticker,
                    "decision": "native-stop-submitted",
                    "order_id": order["order_id"],
                    "stop_price": stop_price,
                    "qty": qty,
                }
            )
        else:
            alerts.critical(
                "Native stop order failed",
                f"{ticker} has a live broker position but no native stop order.",
                {"quantity": qty, "stop_price": stop_price},
            )
            decisions.append(
                {
                    "ticker": ticker,
                    "decision": "native-stop-failed",
                    "stop_price": stop_price,
                    "qty": qty,
                }
            )
    return created


def _normalize_tickers(tickers: list[str]) -> list[str]:
    seen = set()
    normalized = []
    for ticker in tickers:
        symbol = ticker.strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            normalized.append(symbol)
    return normalized


def _select_target_tickers(
    requested_tickers: list[str] | None,
    watchlist_tickers: list[str],
    allow_manual_tickers: bool = True,
) -> tuple[list[str], list[str]]:
    watchlist_tickers = _normalize_tickers(watchlist_tickers)
    if not requested_tickers:
        return watchlist_tickers, []

    requested = _normalize_tickers(requested_tickers)
    if allow_manual_tickers:
        return requested, []

    allowed = set(watchlist_tickers)
    target_tickers = [ticker for ticker in requested if ticker in allowed]
    skipped_tickers = [ticker for ticker in requested if ticker not in allowed]
    return target_tickers, skipped_tickers


def _resolve_mode(mode: str | None, dry_run: bool) -> ExecutionMode:
    if mode is None:
        return "dry-run" if dry_run else "live"

    normalized = mode.strip().lower().replace("_", "-")
    if normalized not in {"dry-run", "shadow", "live"}:
        raise ValueError("mode must be one of: dry-run, shadow, live")
    return normalized  # type: ignore[return-value]


def _create_analysis_graph() -> TradingAgentsGraph:
    config = dict(DEFAULT_CONFIG)
    # Scheduler runs are one-shot per ticker, so checkpoint resumes add state
    # complexity without much benefit here.
    config["checkpoint_enabled"] = False
    return TradingAgentsGraph(
        GRAPH_ANALYSTS,
        config=config,
        debug=False,
    )


def _run_graph_analysis(
    graph: TradingAgentsGraph,
    ticker: str,
    trade_date: str,
) -> tuple[str, dict]:
    final_state, rating = graph.propagate(ticker, trade_date)
    return rating, {
        "market_report": final_state.get("market_report", ""),
        "congressional_report": final_state.get("congressional_report", ""),
        "sentiment_report": final_state.get("sentiment_report", ""),
        "news_report": final_state.get("news_report", ""),
        "fundamentals_report": final_state.get("fundamentals_report", ""),
        "final_trade_decision": final_state.get("final_trade_decision", ""),
    }


def _position_pct_for_rating(rating: str) -> float:
    if rating == "Buy":
        return float(DEFAULT_CONFIG.get("scheduler_buy_position_pct", 0.02))
    if rating == "Overweight":
        return float(DEFAULT_CONFIG.get("scheduler_overweight_position_pct", 0.01))
    return 0.0


def run_cycle(
    tickers: list[str] | None = None,
    min_conviction: int = 6,
    lookback_days: int = 45,
    dry_run: bool = True,
    allow_manual_tickers: bool = True,
    mode: str | None = None,
):
    cycle_start = time.time()
    execution_mode = _resolve_mode(mode, dry_run)
    logger.info(
        "Cycle starting at %s (mode=%s)",
        datetime.now().isoformat(),
        execution_mode,
    )

    if execution_mode != "dry-run":
        market_open, clock, clock_error = executor.require_market_open()
        if not market_open:
            logger.info("Market unavailable; skipping %s cycle: %s", execution_mode, clock_error)
            if clock is None:
                alerts.critical(
                    "Trading cycle skipped",
                    "Could not verify Alpaca market clock, so the bot failed closed.",
                    {"mode": execution_mode},
                )
            return {
                "mode": execution_mode,
                "status": "skipped",
                "reason": clock_error,
                "clock": clock,
            }

    watchlist = get_conviction_watchlist(
        lookback_days=lookback_days, min_score=min_conviction
    )
    watchlist_tickers = (
        watchlist["ticker"].unique().tolist() if not watchlist.empty else []
    )
    if tickers:
        target_tickers, skipped_tickers = _select_target_tickers(
            tickers,
            watchlist_tickers,
            allow_manual_tickers=allow_manual_tickers,
        )
    else:
        target_tickers, skipped_tickers = _build_dynamic_universe(watchlist_tickers), []
    if skipped_tickers:
        logger.info(
            "Manual ticker(s) not on conviction watchlist skipped: %s",
            ", ".join(skipped_tickers),
        )

    cycle_id: int | None = None
    decisions: list[dict] = []
    broker_portfolio: list[dict] = []
    blocked_tickers: dict[str, str] = {}
    signals_fired = 0
    trades_executed = 0
    simulated_trades = 0
    graph = _create_analysis_graph()

    if execution_mode != "dry-run":
        cycle_id = state_store.start_cycle(execution_mode, target_tickers)
        broker_portfolio, portfolio_error = executor.get_portfolio_checked()
        if portfolio_error is not None:
            alerts.critical(
                "Broker reconciliation failed",
                "Could not fetch Alpaca positions, so the cycle failed closed.",
                {"mode": execution_mode, "error": portfolio_error},
            )
            if cycle_id is not None:
                state_store.complete_cycle(
                    cycle_id,
                    decisions=[],
                    portfolio=[],
                    status="failed",
                    error=portfolio_error,
                )
            return {
                "mode": execution_mode,
                "status": "failed",
                "reason": portfolio_error,
            }

        open_orders = executor.get_open_orders()
        state_store.reconcile_broker_state(
            positions=broker_portfolio,
            orders=open_orders,
            mode=execution_mode,
        )
        split_adjustments = _reconcile_splits()
        for adjustment in split_adjustments:
            decisions.append(
                {
                    "ticker": adjustment["ticker"],
                    "decision": "corporate-action-adjustment",
                    "action": adjustment["action"],
                    "factor": adjustment["factor"],
                    "ex_date": adjustment["ex_date"],
                }
            )
        sell_executed, sell_simulated = _evaluate_and_execute_sells(
            execution_mode,
            decisions,
        )
        broker_portfolio, _ = executor.get_portfolio_checked()
        if execution_mode == "live":
            _ensure_native_stop_orders(decisions)
        if execution_mode == "shadow":
            broker_portfolio = [
                p for p in broker_portfolio
                if p["ticker"] not in {
                    d["ticker"] for d in decisions if d.get("decision") == "shadow-sell"
                }
            ]
        trades_executed += sell_executed
        simulated_trades += sell_simulated
        blocked_tickers = state_store.active_blocked_tickers()

    for ticker in target_tickers:
        if ticker in blocked_tickers:
            logger.info("%s skipped: %s", ticker, blocked_tickers[ticker])
            decisions.append(
                {"ticker": ticker, "decision": "skip", "reason": blocked_tickers[ticker]}
            )
            continue
        exclusion = _hard_exclusion_reason(ticker)
        if exclusion:
            logger.info("%s skipped: %s", ticker, exclusion)
            decisions.append({"ticker": ticker, "decision": "skip", "reason": exclusion})
            continue

        manipulation = detect_manipulation(ticker)
        if manipulation["recommendation"] == "REJECT":
            logger.info("%s rejected by manipulation detector", ticker)
            decisions.append(
                {"ticker": ticker, "decision": "reject", "reason": "manipulation"}
            )
            continue

        try:
            data = yf.download(
                ticker,
                period="5d",
                progress=False,
                auto_adjust=False,
            )
            if data.empty:
                continue
            daily_volume = _latest_scalar(data, "Volume")
            close = _latest_scalar(data, "Close") or 0.0
        except Exception as exc:
            logger.warning("yfinance failed for %s: %s", ticker, exc)
            decisions.append(
                {"ticker": ticker, "decision": "skip", "reason": f"yfinance: {exc}"}
            )
            continue

        account = executor.get_account_info()
        if not account:
            portfolio_value = 10_000
        else:
            portfolio_value = account["portfolio_value"]

        kill_msg = kill_switch.check_kill_switch(portfolio_value)
        if kill_msg:
            logger.warning("Kill switch: %s", kill_msg)
            decisions.append(
                {"ticker": ticker, "decision": "stop", "reason": f"kill switch: {kill_msg}"}
            )
            break

        open_positions = len(broker_portfolio) if execution_mode != "dry-run" else len(executor.get_portfolio())
        trade_date = datetime.now().strftime("%Y-%m-%d")
        try:
            rating, analysis = _run_graph_analysis(graph, ticker, trade_date)
        except Exception as exc:
            logger.warning("Trading graph failed for %s: %s", ticker, exc)
            decisions.append(
                {
                    "ticker": ticker,
                    "decision": "skip",
                    "reason": f"graph: {exc}",
                }
            )
            continue

        if rating not in GRAPH_BUY_RATINGS:
            decisions.append(
                {
                    "ticker": ticker,
                    "decision": "skip",
                    "reason": f"graph_rating={rating}",
                    "rating": rating,
                    "final_trade_decision": analysis["final_trade_decision"],
                }
            )
            continue

        signals_fired += 1

        position_size = portfolio_value * _position_pct_for_rating(rating)
        valid, msg = validate_trade(
            ticker=ticker,
            position_size=position_size,
            portfolio_value=portfolio_value,
            open_positions=open_positions,
            daily_volume=daily_volume,
            max_position_pct=float(DEFAULT_CONFIG.get("scheduler_max_position_pct", 0.03)),
            max_open_positions=int(DEFAULT_CONFIG.get("scheduler_max_open_positions", 5)),
            min_daily_volume=float(DEFAULT_CONFIG.get("scheduler_min_daily_volume", 500_000)),
        )

        if not valid:
            logger.info("%s skipped: %s", ticker, msg)
            decisions.append({"ticker": ticker, "decision": "skip", "reason": msg})
            continue

        if execution_mode == "dry-run":
            logger.info(
                "DRY RUN buy signal for %s: %s, $%.2f at %.2f",
                ticker,
                rating,
                position_size,
                close,
            )
            simulated_trades += 1
            decisions.append(
                {
                    "ticker": ticker,
                    "decision": "buy-signal",
                    "mode": execution_mode,
                    "notional": position_size,
                    "price": close,
                    "rating": rating,
                    "final_trade_decision": analysis["final_trade_decision"],
                }
            )
            continue
        if execution_mode == "shadow":
            logger.info(
                "SHADOW buy signal for %s: %s, $%.2f at %.2f",
                ticker,
                rating,
                position_size,
                close,
            )
            state_store.record_buy_signal(
                ticker=ticker,
                entry_price=_shadow_fill_price("buy", close),
                position_size=position_size,
                mode=execution_mode,
                reason="shadow_entry_signal",
            )
            simulated_trades += 1
            decisions.append(
                {
                    "ticker": ticker,
                    "decision": "shadow-buy",
                    "notional": position_size,
                    "price": _shadow_fill_price("buy", close),
                    "rating": rating,
                    "final_trade_decision": analysis["final_trade_decision"],
                }
            )
            continue
        else:
            order = executor.execute_buy_limit(
                ticker,
                position_size,
                _marketable_limit_price("buy", close),
            )

        if order:
            state_store.upsert_order(order, execution_mode)
            position_id = state_store.record_buy_signal(
                ticker=ticker,
                entry_price=close,
                position_size=position_size,
                mode=execution_mode,
                order=order,
            )
            filled_qty = float(order.get("filled_qty") or 0)
            if filled_qty > 0:
                stop_price = _stop_price(float(order.get("filled_avg_price") or close))
                stop_order = executor.execute_stop_sell(ticker, filled_qty, stop_price)
                if stop_order:
                    state_store.upsert_order(stop_order, execution_mode)
                    state_store.set_stop_order(position_id, stop_order["order_id"])
                    decisions.append(
                        {
                            "ticker": ticker,
                            "decision": "native-stop-submitted",
                            "order_id": stop_order["order_id"],
                            "stop_price": stop_price,
                            "qty": filled_qty,
                        }
                    )
                else:
                    alerts.critical(
                        "Native stop order failed",
                        f"{ticker} buy filled but stop order submission failed.",
                        {"quantity": filled_qty, "stop_price": stop_price},
                    )
            trades_executed += 1
            tracker.log_trade_entry(
                ticker=ticker,
                entry_date=datetime.now().strftime("%Y-%m-%d"),
                entry_price=close,
                position_size=position_size,
                conviction_score=None,
            )
            decisions.append(
                {
                    "ticker": ticker,
                    "decision": "live-buy-submitted",
                    "order_id": order.get("order_id"),
                    "status": order.get("status"),
                    "rating": rating,
                    "final_trade_decision": analysis["final_trade_decision"],
                }
            )
        else:
            decisions.append(
                {"ticker": ticker, "decision": "order-failed", "reason": "execute_buy returned None"}
            )

    portfolio_value = 0
    account = executor.get_account_info()
    if account:
        portfolio_value = account["portfolio_value"]

    tracker.log_cycle(
        tickers_scanned=len(target_tickers),
        signals_fired=signals_fired,
        trades_executed=trades_executed,
        portfolio_value=portfolio_value,
        daily_pnl=0,
    )

    if cycle_id is not None:
        state_store.complete_cycle(
            cycle_id,
            decisions=decisions,
            portfolio=broker_portfolio,
        )

    elapsed = time.time() - cycle_start
    logger.info(
        "Cycle complete: %d tickers, %d signals, %d executed, %d simulated in %.1fs",
        len(target_tickers),
        signals_fired,
        trades_executed,
        simulated_trades,
        elapsed,
    )
    return {
        "mode": execution_mode,
        "status": "complete",
        "tickers": len(target_tickers),
        "signals": signals_fired,
        "executed": trades_executed,
        "simulated": simulated_trades,
        "decisions": decisions,
    }


def replay_cycle(cycle_id: int) -> dict:
    cycle = state_store.get_cycle(cycle_id)
    if cycle is None:
        raise ValueError(f"Cycle {cycle_id} not found")
    return {
        "cycle_id": cycle_id,
        "mode": cycle.get("mode"),
        "status": cycle.get("status"),
        "started_at": cycle.get("started_at"),
        "completed_at": cycle.get("completed_at"),
        "tickers": cycle.get("tickers", []),
        "decisions": cycle.get("decisions", []),
        "portfolio": cycle.get("portfolio", []),
        "error": cycle.get("error"),
    }


def start_scheduler(
    interval_minutes: int = 15,
    dry_run: bool = True,
    mode: str | None = None,
    tickers: list[str] | None = None,
    allow_manual_tickers: bool = True,
):
    try:
        import schedule
    except ImportError:
        logger.error("schedule library not installed")
        return

    execution_mode = _resolve_mode(mode, dry_run)

    def protected_cycle():
        try:
            return run_cycle(
                tickers=tickers,
                dry_run=(execution_mode == "dry-run"),
                mode=execution_mode,
                allow_manual_tickers=allow_manual_tickers,
            )
        except Exception as exc:
            logger.exception("Trading cycle crashed")
            alerts.critical(
                "Trading cycle crashed",
                "An unhandled exception occurred inside run_cycle.",
                {"mode": execution_mode, "error": str(exc)},
            )
            return None

    schedule.every(interval_minutes).minutes.do(protected_cycle)
    logger.info(
        "Scheduler started: every %d minutes (mode=%s)",
        interval_minutes,
        execution_mode,
    )

    while True:
        schedule.run_pending()
        time.sleep(1)
