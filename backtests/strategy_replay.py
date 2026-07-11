from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from tradingagents.strategy.rules import (
    StrategyRules,
    position_pct_for_rating,
    stop_price,
    take_profit_price,
)


@dataclass(frozen=True)
class ReplayConfig:
    initial_cash: float = 10_000.0
    fee_pct: float = 0.001
    slippage_bps: int = 10
    scorecard_size_cap: float = 0.005


def summarize_replay_results(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build cross-ticker validation metrics without pretending it is one portfolio."""
    usable = {ticker: result for ticker, result in results.items() if result}
    if not usable:
        return {
            "ticker_count": 0,
            "tickers": [],
            "num_trades": 0,
            "total_return_pct": 0.0,
            "benchmark_return_pct": 0.0,
            "alpha_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "start_date": None,
            "end_date": None,
            "by_ticker": {},
        }
    values = list(usable.values())
    starts = [str(item["start_date"]) for item in values if item.get("start_date")]
    ends = [str(item["end_date"]) for item in values if item.get("end_date")]
    return {
        "ticker_count": len(usable),
        "tickers": sorted(usable),
        "num_trades": sum(int(item.get("num_trades", 0)) for item in values),
        "total_return_pct": round(
            sum(float(item.get("total_return_pct", 0)) for item in values) / len(values),
            2,
        ),
        "benchmark_return_pct": round(
            sum(float(item.get("benchmark_return_pct", 0)) for item in values) / len(values),
            2,
        ),
        "alpha_pct": round(
            sum(float(item.get("alpha_pct", 0)) for item in values) / len(values),
            2,
        ),
        "max_drawdown_pct": round(
            min(float(item.get("max_drawdown_pct", 0)) for item in values),
            2,
        ),
        "start_date": min(starts) if starts else None,
        "end_date": max(ends) if ends else None,
        "by_ticker": usable,
    }


def replay_graph_decisions(
    prices: pd.DataFrame,
    decisions: list[dict[str, Any]],
    *,
    rules: StrategyRules | None = None,
    config: ReplayConfig | None = None,
) -> dict[str, Any]:
    """Replay stored graph ratings with next-bar execution and live strategy exits."""
    rules = rules or StrategyRules()
    config = config or ReplayConfig()
    frame = prices.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
    frame = frame.sort_values("Date").reset_index(drop=True)
    decision_by_date = {
        pd.to_datetime(item["trade_date"], errors="coerce").normalize(): item
        for item in decisions
        if pd.notna(pd.to_datetime(item.get("trade_date"), errors="coerce"))
    }

    cash = config.initial_cash
    qty = 0.0
    entry_price = 0.0
    entry_cost = 0.0
    entry_date: pd.Timestamp | None = None
    holding_days = 0
    pending_rating: str | None = None
    trades: list[dict[str, Any]] = []
    equity_curve: list[float] = []

    for _, row in frame.iterrows():
        date = row["Date"]
        open_price = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        if min(open_price, high, low, close) <= 0:
            continue

        if qty > 0:
            holding_days += 1
            stop = stop_price(entry_price, rules)
            target = take_profit_price(entry_price, rules)
            exit_price = 0.0
            reason = None
            if low <= stop:
                # Conservative ordering if both stop and target are touched.
                exit_price = min(open_price, stop)
                reason = "stop_loss"
            elif high >= target:
                exit_price = max(open_price, target)
                reason = "take_profit"
            elif holding_days >= rules.max_hold_trading_days:
                exit_price = open_price
                reason = "time_exit"
            if reason:
                fill = exit_price * (1 - config.slippage_bps / 10_000)
                proceeds = qty * fill * (1 - config.fee_pct)
                cash += proceeds
                trades.append(
                    {
                        "entry_date": entry_date.date().isoformat() if entry_date is not None else "",
                        "exit_date": date.date().isoformat(),
                        "entry_price": entry_price,
                        "exit_price": fill,
                        "quantity": qty,
                        "pnl": proceeds - entry_cost,
                        "reason": reason,
                    }
                )
                qty = 0.0
                entry_price = 0.0
                entry_cost = 0.0
                entry_date = None
                holding_days = 0

        if qty == 0 and pending_rating and float(row["Volume"]) >= rules.min_daily_volume:
            pct = min(position_pct_for_rating(pending_rating, rules), config.scorecard_size_cap)
            notional = cash * pct
            fill = open_price * (1 + config.slippage_bps / 10_000)
            cost = notional * (1 + config.fee_pct)
            if pct > 0 and cost <= cash:
                qty = notional / fill
                cash -= cost
                entry_price = fill
                entry_cost = cost
                entry_date = date
                holding_days = 0
            pending_rating = None

        decision = decision_by_date.get(date)
        if decision:
            rating = str(decision.get("rating", "Hold"))
            pending_rating = rating if rating in {"Buy", "Overweight"} else None

        equity_curve.append(cash + qty * close)

    if qty > 0 and not frame.empty:
        last = frame.iloc[-1]
        fill = float(last["Close"]) * (1 - config.slippage_bps / 10_000)
        proceeds = qty * fill * (1 - config.fee_pct)
        cash += proceeds
        trades.append(
            {
                "entry_date": entry_date.date().isoformat() if entry_date is not None else "",
                "exit_date": last["Date"].date().isoformat(),
                "entry_price": entry_price,
                "exit_price": fill,
                "quantity": qty,
                "pnl": proceeds - entry_cost,
                "reason": "end_of_test",
            }
        )
        if equity_curve:
            equity_curve[-1] = cash

    equity = pd.Series(equity_curve, dtype="float64")
    rolling_max = equity.cummax() if not equity.empty else equity
    drawdown = ((equity - rolling_max) / rolling_max).min() * 100 if not equity.empty else 0.0
    returns = equity.pct_change().dropna()
    sharpe = (
        returns.mean() / returns.std() * (252**0.5)
        if not returns.empty and returns.std() > 0
        else 0.0
    )
    if frame.empty:
        benchmark_return = 0.0
    else:
        benchmark_entry = float(frame.iloc[0]["Open"]) * (
            1 + config.slippage_bps / 10_000
        )
        benchmark_qty = config.initial_cash / (
            benchmark_entry * (1 + config.fee_pct)
        )
        benchmark_exit = float(frame.iloc[-1]["Close"]) * (
            1 - config.slippage_bps / 10_000
        )
        benchmark_final = benchmark_qty * benchmark_exit * (1 - config.fee_pct)
        benchmark_return = (benchmark_final / config.initial_cash - 1) * 100
    total_return = (cash / config.initial_cash - 1) * 100
    wins = sum(1 for trade in trades if float(trade["pnl"]) > 0)
    return {
        "initial_cash": config.initial_cash,
        "final_equity": round(float(cash), 2),
        "total_return_pct": round(total_return, 2),
        "benchmark_return_pct": round(benchmark_return, 2),
        "alpha_pct": round(total_return - benchmark_return, 2),
        "max_drawdown_pct": round(float(drawdown), 2),
        "sharpe": round(float(sharpe), 2),
        "num_trades": len(trades),
        "win_rate_pct": round((wins / len(trades)) * 100, 2) if trades else 0.0,
        "start_date": frame.iloc[0]["Date"].date().isoformat() if not frame.empty else None,
        "end_date": frame.iloc[-1]["Date"].date().isoformat() if not frame.empty else None,
        "trades": trades,
    }
