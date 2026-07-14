from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yfinance as yf

from tradingagents.alerts import AlertManager
from tradingagents.dataflows.congressional_data import get_conviction_watchlist
from tradingagents.dataflows.manipulation_detector import detect_manipulation
from tradingagents.execution.alpaca_executor import AlpacaExecutor
from tradingagents.execution.order_monitor import OrderUpdateMonitor
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.risk.survival_rules import validate_trade
from tradingagents.risk.scorecard import Scorecard, ScorecardGate
from tradingagents.risk.release_gate import (
    REQUIRED_RELEASE_CHECKS,
    release_strategy_config,
    strategy_fingerprint,
)
from tradingagents.state_store import StrategyStateStore
from tradingagents.runtime import SingleInstanceLock
from tradingagents.strategy.rules import (
    exit_reason as strategy_exit_reason,
    position_pct_for_rating,
    stop_price as strategy_stop_price,
    strategy_rules_from_config,
    take_profit_price as strategy_take_profit_price,
)
from tradingagents.default_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

executor = AlpacaExecutor()
state_store = StrategyStateStore(
    DEFAULT_CONFIG.get("data_cache_dir", "data") + "/strategy_state.db"
)
alerts = AlertManager()

ExecutionMode = Literal["dry-run", "shadow", "paper", "real"]
REAL_MONEY_CONFIRMATION = "ENABLE REAL MONEY"

STOP_LOSS_PCT = float(DEFAULT_CONFIG.get("stop_loss_pct", -0.07))
TAKE_PROFIT_PCT = float(DEFAULT_CONFIG.get("take_profit_pct", 0.12))
MAX_HOLD_TRADING_DAYS = int(DEFAULT_CONFIG.get("max_hold_trading_days", 10))
MANIPULATION_SELL_THRESHOLD = float(
    DEFAULT_CONFIG.get("manipulation_sell_threshold", 0.85)
)
LIMIT_SLIPPAGE_BPS = int(DEFAULT_CONFIG.get("limit_slippage_bps", 20))
SHADOW_SLIPPAGE_BPS = int(DEFAULT_CONFIG.get("shadow_slippage_bps", 10))
MAX_UNIVERSE_SIZE = 20
GRAPH_ANALYSTS = ["congressional", "market", "social", "news", "fundamentals"]
GRAPH_BUY_RATINGS = {"Buy", "Overweight"}
STRATEGY_RULES = strategy_rules_from_config(DEFAULT_CONFIG)
APPROVED_FREE_GOOGLE_MODELS = frozenset(
    {
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    }
)
APPROVED_FREE_GROQ_MODELS = frozenset(
    {"meta-llama/llama-4-scout-17b-16e-instruct"}
)

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


def _validate_free_llm_config() -> str | None:
    """Reject paid or unknown hosted models in autonomous scheduler cycles."""
    provider = str(DEFAULT_CONFIG.get("llm_provider", "")).strip().lower()
    if provider == "ollama":
        return None
    if provider not in {"google", "groq"}:
        return (
            "Automated cycles require a zero-cost supported LLM provider "
            "(google, groq, or ollama)"
        )

    configured_models = {
        str(DEFAULT_CONFIG.get("quick_think_llm", "")).strip().lower(),
        str(DEFAULT_CONFIG.get("deep_think_llm", "")).strip().lower(),
    }
    approved_models = (
        APPROVED_FREE_GOOGLE_MODELS
        if provider == "google"
        else APPROVED_FREE_GROQ_MODELS
    )
    rejected = sorted(configured_models - approved_models)
    if rejected:
        return (
            f"Free-only mode rejected unapproved {provider.title()} model(s): "
            f"{', '.join(rejected)}"
        )
    return None


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


def _market_data_stale(data, max_calendar_days: int = 5) -> bool:
    if data is None or getattr(data, "empty", True):
        return True
    try:
        latest = data.index[-1]
        latest_date = latest.date() if hasattr(latest, "date") else datetime.fromisoformat(str(latest)).date()
    except (IndexError, TypeError, ValueError):
        return True
    return (datetime.now(timezone.utc).date() - latest_date).days > max_calendar_days


def _broker_reference_price(ticker: str, side: str) -> tuple[float, str | None]:
    snapshot, error = executor.get_stock_snapshot_checked(ticker)
    if error or snapshot is None:
        return 0.0, error or "Alpaca IEX snapshot unavailable"
    try:
        observed = datetime.fromisoformat(
            str(snapshot.get("timestamp")).replace("Z", "+00:00")
        )
        if observed.tzinfo is None:
            raise ValueError
        age_minutes = (datetime.now(timezone.utc) - observed).total_seconds() / 60
    except (TypeError, ValueError):
        return 0.0, "Alpaca IEX snapshot timestamp was invalid"
    max_age = int(DEFAULT_CONFIG.get("market_snapshot_max_age_minutes", 5))
    if age_minutes < -1 or age_minutes > max_age:
        return 0.0, f"Alpaca IEX snapshot was {age_minutes:.1f} minutes old"

    bid = float(snapshot.get("bid", 0) or 0)
    ask = float(snapshot.get("ask", 0) or 0)
    last = float(snapshot.get("last", 0) or 0)
    if bid > 0 and ask > 0:
        if ask < bid:
            return 0.0, "Alpaca IEX quote was crossed"
        midpoint = (bid + ask) / 2
        spread_pct = (ask - bid) / midpoint if midpoint else 1.0
        if spread_pct > float(DEFAULT_CONFIG.get("max_quote_spread_pct", 0.02)):
            return 0.0, f"Alpaca IEX spread was too wide ({spread_pct:.2%})"
    price = ask if side == "buy" else bid
    if price <= 0:
        price = last
    if price <= 0:
        return 0.0, "Alpaca IEX snapshot had no usable execution price"
    return price, None


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


def _manipulation_risk(result: dict) -> float:
    """Read the detector's canonical field while accepting older result shapes."""
    return float(
        result.get("manipulation_risk", 0)
        or result.get("manipulation_score", 0)
        or result.get("score", 0)
        or 0
    )


def _portfolio_value_for_mode(mode: ExecutionMode, account: dict | None) -> float | None:
    """Use fixed capital only in dry-run; broker-backed modes fail closed."""
    if account:
        value = float(account.get("portfolio_value", 0) or 0)
        return value if value > 0 else None
    if mode == "dry-run":
        return 10_000.0
    return None


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
    return strategy_exit_reason(
        entry_price=entry_price,
        current_price=current_price,
        holding_days=_trading_days_between(entry_date),
        manipulation_score=manipulation_score,
        rules=STRATEGY_RULES,
    )


def _stop_price(entry_price: float) -> float:
    return strategy_stop_price(entry_price, STRATEGY_RULES)


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
) -> tuple[int, int, int]:
    submitted = 0
    confirmed_fills = 0
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
        if mode in {"paper", "real"}:
            broker_price, price_error = _broker_reference_price(ticker, "sell")
            if price_error:
                state_store.record_health_event(
                    "warning",
                    "market_data",
                    "Broker exit pricing snapshot unavailable",
                    {"ticker": ticker, "error": price_error},
                )
                decisions.append(
                    {"ticker": ticker, "decision": "sell-skip", "reason": price_error}
                )
                continue
            close = broker_price
        manipulation = detect_manipulation(ticker)
        manipulation_score = _manipulation_risk(manipulation)
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

        if not executor.cancel_open_sell_orders(ticker):
            state_store.record_health_event(
                "critical",
                "protective_orders",
                "Could not cancel protective sell orders before an explicit exit",
                {"ticker": ticker, "reason": reason, "mode": mode},
            )
            alerts.critical(
                "Protective-order cancellation failed",
                f"{ticker} exit was not submitted because existing sell orders could not be canceled.",
                {"reason": reason, "mode": mode},
            )
            decisions.append(
                {"ticker": ticker, "decision": "sell-skip", "reason": "cancel_failed"}
            )
            continue

        sell_client_id = _client_order_id(
            "sell",
            ticker,
            datetime.now().strftime("%Y-%m-%d"),
            f"{position['id']}:{reason}",
        )
        order = (
            executor.execute_sell_market(ticker, qty, client_order_id=sell_client_id)
            if reason == "stop_loss"
            else executor.execute_sell_limit(
                ticker,
                qty,
                _marketable_limit_price("sell", close),
                client_order_id=sell_client_id,
            )
        )
        if order:
            updates = state_store.record_order_tree(order, mode, reason=reason)
            confirmed_fills += sum(
                1 for update in updates if float(update.get("fill_delta", 0)) > 0
            )
            filled_qty = float(order.get("filled_qty") or 0)
            if filled_qty + 1e-9 < qty:
                state_store.mark_position_closing(int(position["id"]))
            submitted += 1
            decisions.append(
                {
                    "ticker": ticker,
                    "decision": "broker-sell-submitted",
                    "reason": reason,
                    "order_id": order.get("order_id"),
                    "status": order.get("status"),
                }
            )
        else:
            state_store.record_health_event(
                "critical",
                "execution",
                "Triggered sell order could not be submitted",
                {"ticker": ticker, "reason": reason, "mode": mode},
            )
            alerts.critical(
                "Sell order failed",
                f"{ticker} exit trigger could not submit a sell order.",
                {"reason": reason, "mode": mode},
            )
            decisions.append(
                {"ticker": ticker, "decision": "sell-order-failed", "reason": reason}
            )
    return submitted, confirmed_fills, simulated


def _ensure_native_stop_orders(decisions: list[dict], mode: ExecutionMode = "paper") -> int:
    created = 0
    for position in state_store.positions_needing_stop_orders():
        ticker = position["ticker"]
        qty = float(position.get("broker_quantity") or 0)
        entry_price = float(position.get("entry_price") or 0)
        stop_price = _stop_price(entry_price)
        take_profit_price = strategy_take_profit_price(entry_price, STRATEGY_RULES)
        if qty <= 0 or stop_price <= 0:
            continue
        order = executor.execute_oco_sell(
            ticker,
            qty,
            take_profit_price,
            stop_price,
            _client_order_id("protect", ticker, str(position.get("entry_date") or ""), str(position["id"])),
        )
        if order:
            state_store.record_order_tree(order, mode, reason="protective_oco")
            protection_id = _stop_leg_id(order) or order["order_id"]
            state_store.set_stop_order(int(position["id"]), protection_id)
            created += 1
            decisions.append(
                {
                    "ticker": ticker,
                    "decision": "native-oco-submitted",
                    "order_id": order["order_id"],
                    "stop_price": stop_price,
                    "take_profit_price": take_profit_price,
                    "qty": qty,
                }
            )
        else:
            state_store.record_health_event(
                "critical",
                "protective_orders",
                "Broker position has no verified native protective order",
                {"ticker": ticker, "quantity": qty, "stop_price": stop_price},
            )
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
        return "dry-run" if dry_run else "paper"

    normalized = mode.strip().lower().replace("_", "-")
    if normalized == "live":
        normalized = "paper"
    if normalized not in {"dry-run", "shadow", "paper", "real"}:
        raise ValueError("mode must be one of: dry-run, shadow, paper, real")
    return normalized  # type: ignore[return-value]


def _validate_broker_mode(
    mode: ExecutionMode,
    account: dict | None,
    real_money_confirmation: str | None,
) -> str | None:
    if mode in {"dry-run", "shadow"}:
        return None
    if not executor.is_official_endpoint():
        return "Broker-backed modes require an official HTTPS Alpaca endpoint"
    if account is None:
        return "Alpaca account data is unavailable"
    if account.get("account_blocked") or account.get("trading_blocked"):
        return "Alpaca account is blocked from trading"
    if account.get("trade_suspended_by_user"):
        return "Alpaca trading is suspended by the account owner"
    if mode == "paper":
        if not executor.is_paper_endpoint():
            return "Paper mode refuses to use a real-money Alpaca endpoint"
        return None

    if executor.is_paper_endpoint():
        return "Real mode refuses to use Alpaca's paper endpoint"
    if not bool(DEFAULT_CONFIG.get("allow_real_money", False)):
        return "TRADINGAGENTS_ALLOW_REAL_MONEY is not enabled"
    if str(DEFAULT_CONFIG.get("llm_provider", "")).lower() not in {"google", "groq", "ollama"}:
        return "Real mode requires a zero-cost supported LLM provider (google, groq, or ollama)"
    expected_account = str(DEFAULT_CONFIG.get("expected_real_account_id", "")).strip()
    if not expected_account or expected_account != str(account.get("account_id", "")):
        return "Real Alpaca account ID does not match the configured expected account"
    if float(DEFAULT_CONFIG.get("max_real_money_notional", 0) or 0) <= 0:
        return "TRADINGAGENTS_MAX_REAL_MONEY_NOTIONAL must be positive"
    if real_money_confirmation != REAL_MONEY_CONFIRMATION:
        return f"Real mode requires the exact confirmation phrase: {REAL_MONEY_CONFIRMATION}"
    report_path = Path(str(DEFAULT_CONFIG.get("real_money_validation_report", "")))
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"A valid real-money release report is required at {report_path}"
    if report.get("approved") is not True:
        return "The real-money release report is not approved"
    report_checks = report.get("checks")
    if (
        not isinstance(report_checks, dict)
        or not REQUIRED_RELEASE_CHECKS.issubset(report_checks)
        or not all(value is True for value in report_checks.values())
    ):
        return "The real-money release report contains a failed or invalid gate"
    if str(report.get("account_id", "")) != expected_account:
        return "The release report was generated for a different account"
    if int(report.get("schema_version", 0)) != 2:
        return "The real-money release report schema is obsolete"
    strategy_key = _scorecard_strategy_key()
    if str(report.get("strategy_key", "")) != strategy_key:
        return "The release report was generated for a different strategy version"
    expected_fingerprint = strategy_fingerprint(
        strategy_key,
        release_strategy_config(DEFAULT_CONFIG),
    )
    if str(report.get("strategy_fingerprint", "")) != expected_fingerprint:
        return "The release report no longer matches the current strategy configuration"
    try:
        generated_at = datetime.fromisoformat(str(report["generated_at"]))
        if generated_at.tzinfo is None:
            raise ValueError
        age_hours = (datetime.now(timezone.utc) - generated_at).total_seconds() / 3600
    except (KeyError, TypeError, ValueError):
        return "The release report has an invalid generation timestamp"
    max_age = int(DEFAULT_CONFIG.get("release_report_max_age_hours", 24))
    if age_hours < 0 or age_hours > max_age:
        return f"The release report is older than {max_age} hours"
    return None


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
    return position_pct_for_rating(rating, STRATEGY_RULES)


def _scorecard_path() -> Path:
    return Path(DEFAULT_CONFIG.get("data_cache_dir", "data")) / "scorecard.db"


def _create_scorecard() -> Scorecard:
    return Scorecard(
        _scorecard_path(),
        horizon_days=int(DEFAULT_CONFIG.get("scorecard_horizon_days", 10)),
        stop_loss_pct=float(DEFAULT_CONFIG.get("scorecard_stop_loss_pct", -0.05)),
        warmup_position_pct=float(DEFAULT_CONFIG.get("scorecard_warmup_position_pct", 0.005)),
        tier1_position_pct=float(DEFAULT_CONFIG.get("scorecard_tier1_position_pct", 0.01)),
        tier2_position_pct=float(DEFAULT_CONFIG.get("scorecard_tier2_position_pct", 0.02)),
        min_resolved_decisions=int(DEFAULT_CONFIG.get("scorecard_min_resolved_decisions", 30)),
        tier2_min_decisions=int(DEFAULT_CONFIG.get("scorecard_tier2_min_decisions", 60)),
        benchmark_ticker=DEFAULT_CONFIG.get("benchmark_ticker") or "SPY",
    )


def _scorecard_strategy_key() -> str:
    version = DEFAULT_CONFIG.get("strategy_version", "unknown")
    provider = DEFAULT_CONFIG.get("llm_provider", "unknown")
    quick = DEFAULT_CONFIG.get("quick_think_llm", "unknown")
    deep = DEFAULT_CONFIG.get("deep_think_llm", "unknown")
    return f"full_graph:{version}:{provider}:{quick}:{deep}"


def _client_order_id(action: str, ticker: str, trade_date: str, context: str) -> str:
    """Stable Alpaca idempotency key for one strategy action."""
    raw = f"{action}|{ticker.upper()}|{trade_date}|{context}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:28]
    return f"ta-{action[:6]}-{digest}"


def _stop_leg_id(order: dict) -> str | None:
    for leg in order.get("legs") or []:
        if leg.get("type") in {"stop", "stop_limit", "trailing_stop"}:
            return str(leg.get("order_id") or "") or None
    return None


def _scorecard_fields(gate: ScorecardGate) -> dict:
    return {
        "scorecard_status": gate.status,
        "scorecard_allowed_pct": gate.allowed_position_pct,
        "scorecard_resolved_decisions": gate.resolved_decisions,
        "scorecard_avg_alpha_pct": gate.avg_alpha_pct,
        "scorecard_max_drawdown_pct": gate.max_drawdown_pct,
    }


def run_cycle(
    tickers: list[str] | None = None,
    min_conviction: int | None = None,
    lookback_days: int | None = None,
    dry_run: bool = True,
    allow_manual_tickers: bool = True,
    mode: str | None = None,
    real_money_confirmation: str | None = None,
):
    cycle_start = time.time()
    execution_mode = _resolve_mode(mode, dry_run)
    min_conviction = int(
        min_conviction
        if min_conviction is not None
        else DEFAULT_CONFIG.get("congressional_min_conviction_score", 6)
    )
    lookback_days = int(
        lookback_days
        if lookback_days is not None
        else DEFAULT_CONFIG.get("congressional_lookback_days", 45)
    )
    logger.info(
        "Cycle starting at %s (mode=%s)",
        datetime.now().isoformat(),
        execution_mode,
    )

    llm_config_error = _validate_free_llm_config()
    if llm_config_error:
        state_store.record_health_event(
            "critical",
            "llm_config",
            llm_config_error,
        )
        return {
            "mode": execution_mode,
            "status": "failed",
            "reason": llm_config_error,
            "submitted": 0,
            "executed": 0,
            "simulated": 0,
            "decisions": [],
        }

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
    orders_submitted = 0
    confirmed_fills = 0
    simulated_trades = 0
    graph_successes = 0
    graph_failures = 0
    graph = _create_analysis_graph()
    scorecard = _create_scorecard()
    scorecard_strategy_key = _scorecard_strategy_key()
    if not watchlist_tickers:
        state_store.record_health_event(
            "warning",
            "congressional_data",
            "Congressional watchlist was empty; continuing only because it is an optional signal",
        )
    try:
        scorecard.resolve_due_outcomes()
    except Exception as exc:
        logger.warning("Scorecard outcome resolution failed: %s", exc)

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

        open_orders = executor.get_recent_orders(limit=100, status="all")
        state_store.reconcile_broker_state(
            positions=broker_portfolio,
            orders=open_orders,
            mode=execution_mode,
        )
        activities, activities_error = executor.get_account_activities_checked()
        if activities_error:
            state_store.record_health_event(
                "warning",
                "account_activities",
                "Could not refresh broker fee/fill activities",
                {"error": activities_error},
            )
        else:
            state_store.reconcile_account_activities(activities)
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
        sell_submitted, sell_fills, sell_simulated = _evaluate_and_execute_sells(
            execution_mode,
            decisions,
        )
        broker_portfolio, _ = executor.get_portfolio_checked()
        if execution_mode in {"paper", "real"}:
            _ensure_native_stop_orders(decisions, execution_mode)
        if execution_mode == "shadow":
            broker_portfolio = [
                p for p in broker_portfolio
                if p["ticker"] not in {
                    d["ticker"] for d in decisions if d.get("decision") == "shadow-sell"
                }
            ]
        orders_submitted += sell_submitted
        confirmed_fills += sell_fills
        simulated_trades += sell_simulated
        blocked_tickers = state_store.active_blocked_tickers()

    account = executor.get_account_info()
    portfolio_value = _portfolio_value_for_mode(execution_mode, account)
    if portfolio_value is None:
        error = "Could not verify a positive Alpaca portfolio value"
        alerts.critical(
            "Trading cycle stopped",
            "Account data was unavailable or invalid, so new buying was disabled.",
            {"mode": execution_mode},
        )
        if cycle_id is not None:
            state_store.complete_cycle(
                cycle_id,
                decisions=decisions,
                portfolio=broker_portfolio,
                status="failed",
                error=error,
            )
        return {
            "mode": execution_mode,
            "status": "failed",
            "reason": error,
            "decisions": decisions,
        }

    mode_error = _validate_broker_mode(
        execution_mode, account, real_money_confirmation
    )
    if mode_error:
        state_store.record_health_event(
            "critical", "execution_mode", mode_error, {"mode": execution_mode}
        )
        if cycle_id is not None:
            state_store.complete_cycle(
                cycle_id,
                decisions=decisions,
                portfolio=broker_portfolio,
                status="failed",
                error=mode_error,
            )
        return {
            "mode": execution_mode,
            "status": "failed",
            "reason": mode_error,
            "decisions": decisions,
        }

    risk_gate = None
    if execution_mode != "dry-run":
        risk_gate = state_store.evaluate_persistent_risk(
            equity=portfolio_value,
            cash=float(account.get("cash", 0) if account else 0),
            buying_power=float(account.get("buying_power", 0) if account else 0),
            mode=execution_mode,
            daily_limit_pct=float(DEFAULT_CONFIG.get("daily_drawdown_limit", 0.005)),
            weekly_limit_pct=float(DEFAULT_CONFIG.get("weekly_drawdown_limit", 0.01)),
            total_limit_pct=float(DEFAULT_CONFIG.get("total_drawdown_limit", 0.03)),
        )
        if risk_gate["halted"]:
            error = f"Persistent risk halt: {risk_gate['reason']}"
            alerts.critical("Trading halted", error, risk_gate)
            if cycle_id is not None:
                state_store.complete_cycle(
                    cycle_id,
                    decisions=decisions,
                    portfolio=broker_portfolio,
                    status="halted",
                    error=error,
                )
            return {
                "mode": execution_mode,
                "status": "halted",
                "reason": error,
                "risk": risk_gate,
                "decisions": decisions,
            }

    if execution_mode in {"paper", "real"}:
        unprotected = state_store.positions_needing_stop_orders()
        if unprotected:
            tickers_without_protection = [str(position["ticker"]) for position in unprotected]
            error = "Live position protection could not be verified"
            alerts.critical(
                "New buying disabled",
                error,
                {"tickers": ",".join(tickers_without_protection)},
            )
            if cycle_id is not None:
                state_store.complete_cycle(
                    cycle_id,
                    decisions=decisions,
                    portfolio=broker_portfolio,
                    status="failed",
                    error=error,
                )
            return {
                "mode": execution_mode,
                "status": "failed",
                "reason": error,
                "unprotected_tickers": tickers_without_protection,
                "decisions": decisions,
            }

    open_positions = (
        state_store.reserved_open_position_count()
        if execution_mode != "dry-run"
        else len(executor.get_portfolio())
    )

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
        if manipulation.get("recommendation") == "UNKNOWN" and execution_mode in {"paper", "real"}:
            reason = "social manipulation data unavailable"
            state_store.record_health_event(
                "warning", "stocktwits", reason, {"ticker": ticker, "error": manipulation.get("error")}
            )
            decisions.append({"ticker": ticker, "decision": "skip", "reason": reason})
            continue
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
                decisions.append(
                    {"ticker": ticker, "decision": "skip", "reason": "market_data_unavailable"}
                )
                state_store.record_health_event(
                    "warning", "market_data", "No price data returned", {"ticker": ticker}
                )
                continue
            if _market_data_stale(data):
                decisions.append(
                    {"ticker": ticker, "decision": "skip", "reason": "market_data_stale"}
                )
                state_store.record_health_event(
                    "warning", "market_data", "Stale price data", {"ticker": ticker}
                )
                continue
            daily_volume = _latest_scalar(data, "Volume")
            close = _latest_scalar(data, "Close") or 0.0
        except Exception as exc:
            logger.warning("yfinance failed for %s: %s", ticker, exc)
            decisions.append(
                {"ticker": ticker, "decision": "skip", "reason": f"yfinance: {exc}"}
            )
            continue

        trade_date = datetime.now().strftime("%Y-%m-%d")
        try:
            rating, analysis = _run_graph_analysis(graph, ticker, trade_date)
        except Exception as exc:
            graph_failures += 1
            error_text = str(exc)
            if len(error_text) > 500:
                error_text = error_text[:497] + "..."
            logger.warning("Trading graph failed for %s: %s", ticker, error_text)
            state_store.record_health_event(
                "error",
                "llm_graph",
                "Trading graph analysis failed",
                {"ticker": ticker, "error": error_text},
            )
            decisions.append(
                {
                    "ticker": ticker,
                    "decision": "skip",
                    "reason": f"graph: {error_text}",
                }
            )
            continue
        graph_successes += 1

        try:
            scorecard.record_decision(
                strategy_key=scorecard_strategy_key,
                ticker=ticker,
                trade_date=trade_date,
                rating=rating,
                model_provider=DEFAULT_CONFIG.get("llm_provider"),
                quick_model=DEFAULT_CONFIG.get("quick_think_llm"),
                deep_model=DEFAULT_CONFIG.get("deep_think_llm"),
                mode=execution_mode,
                entry_price=close,
                final_trade_decision=analysis["final_trade_decision"],
                strategy_version=str(DEFAULT_CONFIG.get("strategy_version", "unknown")),
                config={
                    "rules": STRATEGY_RULES.__dict__,
                    "scorecard_horizon_days": DEFAULT_CONFIG.get("scorecard_horizon_days"),
                    "analysts": GRAPH_ANALYSTS,
                },
            )
        except Exception as exc:
            logger.warning("Scorecard logging failed for %s: %s", ticker, exc)

        gate = scorecard.gate_for_strategy(scorecard_strategy_key)

        if rating not in GRAPH_BUY_RATINGS:
            decisions.append(
                {
                    "ticker": ticker,
                    "decision": "skip",
                    "reason": f"graph_rating={rating}",
                    "rating": rating,
                    "final_trade_decision": analysis["final_trade_decision"],
                    **_scorecard_fields(gate),
                }
            )
            continue

        if gate.allowed_position_pct <= 0:
            logger.info("%s skipped: scorecard blocked new buys (%s)", ticker, gate.reason)
            decisions.append(
                {
                    "ticker": ticker,
                    "decision": "skip",
                    "reason": "scorecard_blocked",
                    "rating": rating,
                    "final_trade_decision": analysis["final_trade_decision"],
                    **_scorecard_fields(gate),
                }
            )
            continue

        signals_fired += 1

        base_position_pct = _position_pct_for_rating(rating)
        position_pct = min(base_position_pct, gate.allowed_position_pct)
        if execution_mode == "real":
            position_pct = min(
                position_pct,
                float(DEFAULT_CONFIG.get("real_position_pct", 0.005)),
            )
        position_size = portfolio_value * position_pct
        if execution_mode == "real":
            position_size = min(
                position_size,
                float(DEFAULT_CONFIG.get("max_real_money_notional", 0) or 0),
                max(
                    0.0,
                    float(account.get("cash", 0) if account else 0)
                    - state_store.reserved_buy_notional(),
                ),
            )
            current_exposure = sum(
                max(0.0, float(position.get("market_value", 0) or 0))
                for position in broker_portfolio
            ) + state_store.reserved_buy_notional()
            max_exposure = portfolio_value * float(
                DEFAULT_CONFIG.get("real_max_exposure_pct", 0.015)
            )
            if current_exposure + position_size > max_exposure:
                decisions.append(
                    {
                        "ticker": ticker,
                        "decision": "skip",
                        "reason": "real_money_exposure_cap",
                    }
                )
                continue
            if position_size <= 0:
                decisions.append(
                    {"ticker": ticker, "decision": "skip", "reason": "no_unreserved_cash"}
                )
                continue
        valid, msg = validate_trade(
            ticker=ticker,
            position_size=position_size,
            portfolio_value=portfolio_value,
            open_positions=open_positions,
            daily_volume=daily_volume,
            max_position_pct=float(DEFAULT_CONFIG.get("scheduler_max_position_pct", 0.03)),
            max_open_positions=(
                int(DEFAULT_CONFIG.get("real_max_open_positions", 1))
                if execution_mode == "real"
                else int(DEFAULT_CONFIG.get("scheduler_max_open_positions", 5))
            ),
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
                    "position_pct": position_pct,
                    "price": close,
                    "rating": rating,
                    "final_trade_decision": analysis["final_trade_decision"],
                    **_scorecard_fields(gate),
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
                    "position_pct": position_pct,
                    "price": _shadow_fill_price("buy", close),
                    "rating": rating,
                    "final_trade_decision": analysis["final_trade_decision"],
                    **_scorecard_fields(gate),
                }
            )
            open_positions += 1
            continue
        else:
            execution_price, price_error = _broker_reference_price(ticker, "buy")
            if price_error:
                state_store.record_health_event(
                    "warning",
                    "market_data",
                    "Broker entry pricing snapshot unavailable",
                    {"ticker": ticker, "error": price_error},
                )
                decisions.append(
                    {"ticker": ticker, "decision": "skip", "reason": price_error}
                )
                continue
            order = executor.execute_buy_bracket(
                ticker,
                position_size,
                _marketable_limit_price("buy", execution_price),
                _stop_price(execution_price),
                strategy_take_profit_price(execution_price, STRATEGY_RULES),
                _client_order_id("buy", ticker, trade_date, scorecard_strategy_key),
            )

        if order:
            updates = state_store.record_order_tree(
                order, execution_mode, reason="graph_entry"
            )
            position_id = next(
                (int(item["position_id"]) for item in updates if item.get("position_id")),
                0,
            )
            protection_id = _stop_leg_id(order)
            if position_id and protection_id:
                state_store.set_stop_order(position_id, protection_id)
            orders_submitted += 1
            confirmed_fills += sum(
                1 for update in updates if float(update.get("fill_delta", 0)) > 0
            )
            open_positions += 1
            decisions.append(
                {
                    "ticker": ticker,
                    "decision": "broker-buy-submitted",
                    "order_id": order.get("order_id"),
                    "status": order.get("status"),
                    "rating": rating,
                    "final_trade_decision": analysis["final_trade_decision"],
                    "position_pct": position_pct,
                    **_scorecard_fields(gate),
                }
            )
        else:
            state_store.record_health_event(
                "warning",
                "execution",
                "Buy order could not be submitted",
                {"ticker": ticker, "mode": execution_mode},
            )
            decisions.append(
                {"ticker": ticker, "decision": "order-failed", "reason": "execute_buy returned None"}
            )

    cycle_status = "complete"
    cycle_error = None
    if graph_failures:
        if graph_successes == 0:
            cycle_status = "failed"
            cycle_error = (
                f"All {graph_failures} attempted graph analyses failed; "
                "check LLM model access and quota"
            )
        else:
            cycle_status = "degraded"
            cycle_error = (
                f"{graph_failures} graph analyses failed and "
                f"{graph_successes} completed"
            )
    if execution_mode in {"paper", "real"}:
        _ensure_native_stop_orders(decisions, execution_mode)
        final_unprotected = state_store.positions_needing_stop_orders()
        if final_unprotected:
            tickers_without_protection = [
                str(position["ticker"]) for position in final_unprotected
            ]
            cycle_status = "failed"
            cycle_error = (
                "Cycle ended with unprotected broker positions: "
                + ", ".join(tickers_without_protection)
            )
            state_store.record_health_event(
                "critical",
                "protective_orders",
                cycle_error,
                {"mode": execution_mode},
            )

    if cycle_id is not None:
        state_store.complete_cycle(
            cycle_id,
            decisions=decisions,
            portfolio=broker_portfolio,
            status=cycle_status,
            error=cycle_error,
        )

    elapsed = time.time() - cycle_start
    logger.info(
        "Cycle complete: %d tickers, %d signals, %d submitted, %d confirmed fills, %d simulated in %.1fs",
        len(target_tickers),
        signals_fired,
        orders_submitted,
        confirmed_fills,
        simulated_trades,
        elapsed,
    )
    return {
        "mode": execution_mode,
        "status": cycle_status,
        "tickers": len(target_tickers),
        "signals": signals_fired,
        "submitted": orders_submitted,
        "executed": confirmed_fills,
        "simulated": simulated_trades,
        "analysis_failures": graph_failures,
        "reason": cycle_error,
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
    interval_minutes: int | None = None,
    dry_run: bool = True,
    mode: str | None = None,
    tickers: list[str] | None = None,
    allow_manual_tickers: bool = True,
    run_immediately: bool = True,
    daily_at: str = "08:45",
    timezone_name: str = "America/Chicago",
    real_money_confirmation: str | None = None,
):
    if interval_minutes is not None and interval_minutes < 1:
        raise ValueError("interval_minutes must be at least 1")
    try:
        datetime.strptime(daily_at, "%H:%M")
    except ValueError as exc:
        raise ValueError("daily_at must use 24-hour HH:MM format") from exc

    try:
        import schedule
    except ImportError as exc:
        raise RuntimeError("schedule library is required to run the bot") from exc

    execution_mode = _resolve_mode(mode, dry_run)

    def protected_cycle():
        try:
            return run_cycle(
                tickers=tickers,
                dry_run=(execution_mode == "dry-run"),
                mode=execution_mode,
                allow_manual_tickers=allow_manual_tickers,
                real_money_confirmation=real_money_confirmation,
            )
        except Exception as exc:
            logger.exception("Trading cycle crashed")
            alerts.critical(
                "Trading cycle crashed",
                "An unhandled exception occurred inside run_cycle.",
                {"mode": execution_mode, "error": str(exc)},
            )
            return None

    scheduler = schedule.Scheduler()
    if interval_minutes is not None:
        scheduler.every(interval_minutes).minutes.do(protected_cycle)
        schedule_description = f"every {interval_minutes} minutes"
    else:
        for weekday in ("monday", "tuesday", "wednesday", "thursday", "friday"):
            getattr(scheduler.every(), weekday).at(daily_at, timezone_name).do(protected_cycle)
        schedule_description = f"weekdays at {daily_at} ({timezone_name})"
    logger.info("Scheduler started: %s (mode=%s)", schedule_description, execution_mode)

    lock = SingleInstanceLock(
        Path(DEFAULT_CONFIG.get("data_cache_dir", "data")) / "trading-bot.lock"
    )
    monitor = (
        OrderUpdateMonitor(state_store, execution_mode)
        if execution_mode in {"paper", "real"}
        else None
    )
    lock.acquire()
    try:
        if monitor:
            monitor.start()
        if run_immediately:
            protected_cycle()
        while True:
            scheduler.run_pending()
            time.sleep(1)
    finally:
        if monitor:
            monitor.stop()
        lock.release()
